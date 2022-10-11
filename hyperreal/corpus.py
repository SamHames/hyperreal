"""
For the purposes of hyperreal, a corpus is a collection of documents. A corpus
object is a Data Access Object (DAO), that is responsible for representing and
accessing documents stored elsewhere. The corpus is also responsible for
describing how those documents are to be represented in the index.

This module describes the protocol a Corpus class needs to implement, and also
shows concrete implementation examples.

"""

import abc
from collections import defaultdict
import datetime as dt
import gzip
import html
import json
import re
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse
from xml.etree import ElementTree

from dateutil.parser import isoparse
from jinja2 import Template
from markupsafe import Markup

from hyperreal import utilities
from hyperreal.db_utilities import connect_sqlite, dict_factory


@runtime_checkable
class BaseCorpus(Protocol):
    """
    - provides access to documents, and describes how to index them.
    - needs to be picklable and safely enable concurrent read access
    - Designed to enable small batches of work, and avoid representing entire
      collections in memory/work on collections much larger than memory.
    - Designed to enable downstream concurrent computing on those small batches.
    """

    CORPUS_TYPE: str

    @abc.abstractmethod
    def docs(self, doc_keys=None):
        """
        Return an iterator of key-document pairs matching the given keys.

        If doc_keys is None, all documents will be iterated over.

        """
        pass

    @abc.abstractmethod
    def keys(self):
        """An iterator of all document keys present."""
        pass

    @abc.abstractmethod
    def index(self, doc):
        """
        Returns a mapping of:

            {
                "field1": [value1, value2],
                "field2": [value]
            }

        Values need not be deduplicated: the indexer will take care of that
        for the boolean query construction, and leaving duplicates allows for
        things like Bag of Feature counts.

        """
        pass

    @abc.abstractmethod
    def serialize(self):
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, state):
        pass

    def close(self):
        pass

    def __getitem__(self, doc_key):
        return self.docs([doc_key])

    def __iter__(self):
        return self.docs(doc_keys=None)


class SqliteBackedCorpus(BaseCorpus):
    def __init__(self, db_path):
        """
        A helper class for creating corpuses backed by SQLite databases.

        This handles some basic things like saving the database path and ensuring
        that the corpus object is picklable for multiprocessing.

        You will still need to:

        - Add the CORPUS_TYPE class attribute
        - Define the docs, keys and index methods

        """

        self.db_path = db_path
        self._db = None

    @property
    def db(self):
        if self._db is None:
            self._db = connect_sqlite(self.db_path, row_factory=dict_factory)

        return self._db

    def __getstate__(self):
        return self.db_path

    def __setstate__(self, db_path):
        self.__init__(db_path)

    def serialize(self):
        return self.db_path

    @classmethod
    def deserialize(cls, data):
        return cls(data)

    def close(self):
        if self._db is not None:
            self._db.close()


class PlainTextSqliteCorpus(SqliteBackedCorpus):

    CORPUS_TYPE = "PlainTextSqliteCorpus"

    def replace_docs(self, texts):
        """Replace the existing documents with texts."""
        self.db.execute("pragma journal_mode=WAL")
        self.db.execute("savepoint add_texts")

        self.db.execute("drop table if exists doc")
        self.db.execute(
            """
            create table doc(
                doc_id integer primary key,
                text not null
            )
            """
        )

        self.db.execute("delete from doc")
        self.db.executemany(
            "insert or ignore into doc(text) values(?)", ([t] for t in texts)
        )

        self.db.execute("release add_texts")

    def docs(self, doc_keys=None):

        self.db.execute("savepoint docs")

        # Note that it's valid to pass an empty sequence of doc_keys,
        # so we need to check sentinel explicitly.
        if doc_keys is None:
            doc_keys = self.keys()

        for key in doc_keys:
            doc = list(
                self.db.execute("select doc_id, text from doc where doc_id = ?", [key])
            )[0]
            yield key, doc

        self.db.execute("release docs")

    def keys(self):
        return (r["doc_id"] for r in self.db.execute("select doc_id from doc"))

    def index(self, doc):
        return {
            "text": utilities.tokens(doc["text"]),
        }

    def render_docs_html(self, doc_keys):
        """Return the given documents as HTML."""
        return [(key, doc["text"]) for key, doc in self.docs(doc_keys=doc_keys)]


class StackExchangeCorpus(SqliteBackedCorpus):

    CORPUS_TYPE = "StackExchangeCorpus"

    def add_site_data(self, posts_file, comments_file, users_file, site_url):
        """
        Add the data for a specific stackexchange site to the model.

        The site_url is used for differentiating sites, and constructing links
        to relevant pages on the correct site.

        Existing rows in the database will be replaced, so it is possible to
        update with newer data, however if data is missing from the archive
        dump that was present earlier, the state of the history may be
        inconsistent. It may be better to recreate the entire state from
        scratch in this case.

        """
        self.db.execute("pragma journal_mode=WAL")
        self.db.executescript(
            """
            create table if not exists Site(
                site_id integer primary key,
                site_url unique
            );

            create table if not exists Post(
                -- doc_id is a surrogate key necessary for indexing.
                -- The natural key is (site_id, Id).
                doc_id integer primary key,
                site_id integer references Site,
                Id integer,
                OwnerUserId integer,
                AcceptedAnswerId integer references Post,
                ContentLicense text,
                ParentId integer references Post,
                Title text default '',
                FavoriteCount integer,
                Score integer,
                CreationDate datetime,
                ViewCount integer,
                Body text,
                LastActivityDate datetime,
                CommentCount integer,
                PostType text,
                unique(site_id, Id),
                foreign key (site_id, OwnerUserId) references User(site_id, Id)
            );

            create table if not exists comment(
                CreationDate datetime,
                ContentLicense text,
                Score integer,
                Text text,
                UserId integer,
                PostId integer,
                Id integer,
                site_id integer references Site,
                primary key (site_id, Id),
                foreign key (site_id, UserId) references User(site_id, Id),
                foreign key (site_id, PostId) references Post(site_id, Id)
            );

            create index if not exists post_comment on comment(site_id, PostId);

            create table if not exists User(
                AboutMe text,
                CreationDate datetime,
                Location text,
                ProfileImageUrl text,
                WebsiteUrl text,
                AccountId integer,
                Reputation integer,
                Id integer,
                Views integer,
                UpVotes integer,
                DownVotes integer,
                DisplayName text,
                LastAccessDate datetime,
                site_id integer references Site,
                primary key (site_id, Id)
            );

            create table if not exists PostTag(
                site_id integer references Site,
                PostId integer,
                Tag text,
                primary key (site_id, PostId, Tag),
                foreign key (site_id, PostId) references Post
            );

            """
        )

        try:
            self.db.execute("begin")

            # Process Posts, which includes both questions and answers.
            tag_splitter = re.compile("<|>|<>")

            self.db.execute(
                "insert or ignore into Site(site_url) values(?)", [site_url]
            )
            site_id = list(
                self.db.execute(
                    "select site_id from Site where site_url = ?", [site_url]
                )
            )[0]["site_id"]

            tree = ElementTree.iterparse(posts_file, events=("end",))
            post_types = {"1": "Question", "2": "Answer"}

            for event, elem in tree:

                # We only consider questions and answers - SX uses other post types
                # to describe wiki's, tags, moderator nominations and more.
                if elem.attrib.get("PostTypeId") not in ("1", "2"):
                    elem.clear()
                    continue

                doc = defaultdict(lambda: None)
                doc.update(elem.attrib)
                doc["PostType"] = post_types[elem.attrib["PostTypeId"]]
                doc["site_id"] = site_id

                self.db.execute(
                    """
                    replace into post values (
                        (select doc_id from Post where (site_id, Id) = (:site_id, :Id)),
                        :site_id,
                        :Id,
                        :OwnerUserId,
                        :AcceptedAnswerId,
                        :ContentLicense,
                        :ParentId,
                        :Title,
                        :FavoriteCount,
                        :Score,
                        :CreationDate,
                        :ViewCount,
                        :Body,
                        :LastActivityDate,
                        :CommentCount,
                        :PostType
                    )
                    """,
                    doc,
                )

                tag_insert = (
                    (site_id, elem.attrib["Id"], t)
                    for t in tag_splitter.split(elem.attrib.get("Tags", ""))
                    if t
                )

                self.db.executemany("replace into PostTag values(?, ?, ?)", tag_insert)

                # This is important when using iterparse to free memory from
                # processed nodes in the tree.
                elem.clear()

            tree = ElementTree.iterparse(comments_file, events=("end",))

            for event, elem in tree:

                doc = defaultdict(lambda: None)
                doc.update(elem.attrib)
                doc["site_id"] = site_id

                self.db.execute(
                    """
                    replace into comment values (
                        :CreationDate,
                        :ContentLicense,
                        :Score,
                        :Text,
                        :UserId,
                        :PostId,
                        :Id,
                        :site_id
                    )
                    """,
                    doc,
                )
                elem.clear()

            tree = ElementTree.iterparse(users_file, events=("end",))

            for event, elem in tree:

                doc = defaultdict(lambda: None)
                doc.update(elem.attrib)
                doc["site_id"] = site_id

                self.db.execute(
                    """
                    replace into user values (
                        :AboutMe,
                        :CreationDate,
                        :Location,
                        :ProfileImageUrl,
                        :WebsiteUrl,
                        :AccountId,
                        :Reputation,
                        :Id,
                        :Views,
                        :UpVotes,
                        :DownVotes,
                        :DisplayName,
                        :LastAccessDate,
                        :site_id
                    )
                    """,
                    doc,
                )
                elem.clear()

            self.db.execute("commit")

        except Exception as e:
            self.db.execute("rollback")
            raise

    def docs(self, doc_keys=None):
        self.db.execute("savepoint docs")

        try:
            # Note that it's valid to pass an empty sequence of doc_keys,
            # so we need to check sentinel explicitly.
            if doc_keys is None:
                doc_keys = self.keys()

            for key in doc_keys:

                doc = list(
                    self.db.execute(
                        """
                        SELECT
                            site_url,
                            site_id,
                            Id,
                            Title,
                            Body,
                            -- Used to retrieve tags for the root question, as
                            -- the tags are not present on answers, only questions.
                            coalesce(ParentId, Id) as TagPostId,
                            coalesce(
                                (
                                    select DisplayName
                                    from User
                                    where
                                        (User.site_id, User.Id) =
                                        (Post.site_id, Post.OwnerUserId)
                                ),
                                '<Deleted User>'
                            ) as DisplayName,
                            CreationDate
                        from Post
                        inner join Site using(site_id)
                        where Post.doc_id = ?
                        """,
                        [key],
                    )
                )[0]

                # Use the tags on the question, not the (absent) tags on the answer.
                doc["Tags"] = {
                    r["Tag"]
                    for r in self.db.execute(
                        "select Tag from PostTag where (site_id, PostId) = (:site_id, :TagPostId)",
                        doc,
                    )
                }

                # Note we're indexing by AccountId, which is stable across all SX sites,
                # not the local user ID.
                doc["UserComments"] = list(
                    self.db.execute(
                        """
                        SELECT
                            coalesce(
                                (
                                    select DisplayName
                                    from User
                                    where
                                        (User.site_id, User.Id) =
                                        (comment.site_id, comment.UserId)
                                ),
                                '<Deleted User>'
                            ) as DisplayName,
                            Text
                        from comment
                        where (comment.site_id, comment.PostId) = (:site_id, :Id)
                        """,
                        doc,
                    )
                )

                yield key, doc

        finally:
            self.db.execute("release docs")

    TEMPLATE = Template(
        """
        <details>
            <summary>{{ base_fields["PostType"] }} from {{ base_fields["site_url"] }}: "{{ base_fields["QuestionTitle"] }}"</summary>

            <a href="{{ '{}/questions/{}'.format(base_fields["site_url"], base_fields["Id"])  }}">Live Link</a>

            {{ base_fields["Body"] }}

            <p>
                <small>
                    Copyright {{ base_fields["ContentLicense"]}} by
                    {% if base_fields["OwnerUserId"] %}
                    <a href="{{ '{}/users/{}'.format(base_fields["site_url"], base_fields["OwnerUserId"]) }}">
                        {{ base_fields["DisplayName"] }}
                    </a>
                    {% else %}
                    <Deleted User>
                    {% endif %}
                </small>
            </p>

            <details>
                <summary>Tags:</summary>
                <ul>
                    {{ tags }}
                </ul>
            </details>

            <details>
                <summary>Comments:</summary>
                <ul>
                    {% for comment in user_comments %}
                        <li>{{ comment["Text"] }}
                            <small>
                                Copyright {{ comment["ContentLicense"]}} by
                                {% if comment["UserId"] %}
                                <a href="{{ '{}/users/{}'.format(comment["site_url"], comment["UserId"]) }}">
                                    {{ comment["DisplayName"] }}
                                </a>
                                {% else %}
                                <Deleted User>
                                {% endif %}
                            </small>
                        </li>
                    {% endfor %}
                </ul>
            </details>
        </details>
        """
    )

    def _render_doc_key(self, key):

        base_fields = list(
            self.db.execute(
                """
                select
                    Post.site_id,
                    site_url,
                    Post.Id,
                    -- Used to retrieve tags for the root question, as
                    -- the tags are not present on answers, only questions.
                    coalesce(Post.ParentId, Post.Id) as TagPostId,
                    Post.OwnerUserId,
                    (
                        select DisplayName
                        from User
                        where
                            (User.site_id, User.Id) =
                            (Post.site_id, Post.OwnerUserId)
                    ) as DisplayName,
                    post.ContentLicense,
                    post.PostType,
                    post.Body,
                    coalesce(Post.Title, parent.Title) as QuestionTitle,
                    coalesce(Post.ParentId, Post.Id) as TagPostId
                from Post
                inner join site using(site_id)
                left outer join Post parent
                    on (parent.site_id, parent.Id) =
                        (Post.site_id, Post.ParentId)
                where Post.doc_id = ?
                """,
                [key],
            )
        )[0]

        tags = "\n".join(
            f"<li>  { r['Tag'] }"
            for r in self.db.execute(
                "select Tag from PostTag where (site_id, PostId) = (:site_id, :TagPostId)",
                base_fields,
            )
        )

        user_comments = list(
            self.db.execute(
                """
                select
                    site_url,
                    UserId,
                    (
                        select DisplayName
                        from User
                        where
                            (User.site_id, User.Id) =
                            (comment.site_id, comment.UserId)
                    ) as DisplayName,
                    Text,
                    ContentLicense
                from comment
                inner join site using(site_id)
                where (site_id, PostId) = (:site_id, :Id)
                """,
                base_fields,
            )
        )

        return Markup(
            self.TEMPLATE.render(
                base_fields=base_fields, tags=tags, user_comments=user_comments
            )
        )

    def render_docs_html(self, doc_keys):
        self.db.execute("savepoint render_docs_html")

        docs = [(key, self._render_doc_key(key)) for key in doc_keys]

        self.db.execute("release render_docs_html")

        return docs

    def keys(self):
        return (r["doc_id"] for r in self.db.execute("select doc_id from Post"))

    def index(self, doc):

        indexed = {
            "UserPosting": set([doc["DisplayName"]]),
            # Note tokenise different components separately, so there are
            # sentinels included for long distance bigrams.
            "Post": utilities.tokens((doc["Title"] or "")) +
            # Todo: this is pretty basic, in the future may want to pull out
            # some markup into a separate field, like for code.
            [
                t
                for l in utilities.text_from_html(doc["Body"] or "")
                for t in utilities.tokens(l)
            ],
            "Tag": doc["Tags"],
            # Comments from deleted users remain, but have no UserId associated.
            "UserCommenting": {u["DisplayName"] for u in doc["UserComments"]},
            "Comment": [
                t for c in doc["UserComments"] for t in utilities.tokens(c["Text"])
            ],
            "Site": set([doc["site_url"]]),
        }

        created_at = isoparse(doc["CreationDate"])
        rounded_times = utilities.round_datetime(created_at)
        for granularity, dt in rounded_times.items():
            # Don't bother indexing at minute/hour granularity - just day/month/year
            if granularity in ("minute", "hour"):
                continue
            indexed[f"created_{granularity}"] = [dt.isoformat()]

        return indexed


class TwittersphereCorpus(SqliteBackedCorpus):

    """
    A corpus of tweets based on the relational database representation from
    the Twittersphere tool:
    https://github.com/QUT-Digital-Observatory/twittersphere

    Modelling Notes
    ---------------

    Each document in this corpus is a tweet object.

    Only tweets that have been collected directly in response to a query such
    as a user timeline or search query will be included - tweets that are
    only included because they are referenced by other Tweets are excluded.

    The latest collected version of each tweet and user object is used: if
    a user has tweets collected at multiple times, the latest version of their
    user profile, including username will be usef.

    Tweets are indexed by:

    - The time bucket they were created in (minute/hour/day/month/year)
    - @mentions
    - #hashtags
    - Included external URLs (that is, links to Twitter itself such as in quote tweets are removed)
    - Domains of included external URLs
    - The text of the tweet, after removing hashtags, mentions and URLs
    - The user who created the tweet
    - The ID of the conversation thread the tweet is part of
    - The ID of the tweet being replied to, retweeted, or quoted
    - If retrievable, the user being replied to, retweeted or quoted

    Retweets are treated as references to the original tweet, not as verbatim
    republication. Thus the only entries indexed in this case are:

    - When the retweet happened
    - The tweet ID of the tweet being retweeted
    - The user ID of the account posting the tweet being retweeted
    - The user who is retweeting the tweet

    """

    CORPUS_TYPE = "TwittersphereCorpus"

    def docs(self, doc_keys=None):
        self.db.execute("savepoint docs")

        try:
            # Note that it's valid to pass an empty sequence of doc_keys,
            # so we need to check sentinel explicitly.
            if doc_keys is None:
                doc_keys = self.keys()

            for tweet_id in doc_keys:

                doc = list(
                    self.db.execute(
                        """
                        SELECT
                            *
                        from tweet_latest
                        where tweet_id = ?
                        """,
                        [tweet_id],
                    )
                )[0]

                doc["hashtags"] = {
                    r["tag"]
                    for r in self.db.execute(
                        "select lower(hashtag) as tag from tweet_hashtag where tweet_id = ?",
                        [tweet_id],
                    )
                }

                doc["mentions"] = {
                    r["mentioned_username"]
                    for r in self.db.execute(
                        """
                        select
                            lower(mentioned_username) as mentioned_username
                        from tweet_mention
                        where tweet_id = ?
                        """,
                        [tweet_id],
                    )
                }

                # Extract links outside of Twitter. The API does also return
                # many internal links, including media, quote tweets, and
                # occasionally people sharing direct links to other tweets
                # rather than using platform affordances.
                doc["external_urls"] = {
                    r["full_url"]
                    for r in self.db.execute(
                        """
                        select
                            coalesce(unwound_url, expanded_url, url) as full_url
                        from tweet_latest
                        inner join tweet_url using(tweet_id, retrieved_at)
                        inner join url using(url, retrieved_at)
                        where tweet_id = ?
                            and full_url not like 'https://twitter.com/%'
                        """,
                        [tweet_id],
                    )
                }

                doc["external_domains"] = {
                    urlparse(url).hostname for url in doc["external_urls"]
                }

                doc["user"] = self._retrieve_user(doc["user_id"])

                # Retrieve the user profiles of related tweets, if present.
                # Note that this is not retrievable in all cases - especially
                # for replies and quote tweets, the underlying reply/quote
                # tweet may no longer exist, and therefore won't be present.
                for key in (
                    "replied_to_tweet_id",
                    "retweeted_tweet_id",
                    "quoted_tweet_id",
                ):
                    if doc[key] is not None:
                        key_user = list(
                            self.db.execute(
                                """
                                select user_id, username
                                from user_at_time
                                where (user_id, retrieved_at) = (
                                    select user_id, retrieved_at
                                    from tweet_latest
                                    where tweet_id = ?
                                )
                                """,
                                [doc[key]],
                            )
                        )

                        if key_user:
                            doc[key.replace("tweet_id", "user")] = key_user[0]

                yield tweet_id, doc

        finally:
            self.db.execute("release docs")

    def _retrieve_user(self, user_id):

        return list(
            self.db.execute("select * from user_latest where user_id = ?", [user_id])
        )[0]

    TEMPLATE = Template(
        """
        <div>
            <span>
            <img src={{ doc["user"]["profile_image_url"] }}>
            <a href="https://twitter.com/{{ doc["user"]["username"] }}">@{{ doc["user"]["username"] }}</a>
            {{ doc["user"]["name"] }} on <a href="https://twitter.com/{{ doc["user"]["username"] }}/status/{{ doc["tweet_id"] }}">
                <time datetime="{{ doc['created_at'] }}">{{ doc['created_at'] }}</time>
            </a>
            </span>

            <p>{{ doc["text"] }}</p>

            <details>
                <summary>Metadata</summary>
                <dl>
                    <dt>Likes</dt><dd>{{ doc["like_count"] }}</dd>
                    <dt>Replies</dt><dd>{{ doc["reply_count"] }}</dd>
                    <dt>Quote Tweets</dt><dd>{{ doc["quote_count"] }}</dd>
                    <dt>Retweets</dt><dd>{{ doc["retweet_count"] }}</dd>
                    <dt>Retrieved at</dt><dd>{{ doc["retrieved_at"] }}</dd>
                </dl>
            </details>
        </div

        """
    )

    def render_docs_html(self, doc_keys):
        self.db.execute("savepoint render_docs_html")

        docs = [
            (key, Markup(self.TEMPLATE.render(doc=doc)))
            for key, doc in self.docs(doc_keys=doc_keys)
        ]

        self.db.execute("release render_docs_html")

        return docs

    def keys(self):
        return (
            r["tweet_id"]
            for r in self.db.execute(
                "select tweet_id from directly_collected_tweet order by tweet_id"
            )
        )

    def index(self, doc):

        # Round down datetime to various degrees of granularity
        created_at = isoparse(doc["created_at"])
        rounded_times = utilities.round_datetime(created_at)

        # Note that we only index the full set of items for original tweets
        # Retweets are indexed as a pointer to the original tweet only.
        if doc["retweeted_tweet_id"] is None:
            tokens = utilities.social_media_tokens(html.unescape(doc["text"]))

            indexed = {
                "text": tokens,
                "#": {h.lower() for h in doc["hashtags"]},
                "@": {m.lower() for m in doc["mentions"]},
                "conversation_id": [doc["conversation_id"]],
                "by": [doc["user"]["username"]],
                "url": doc["external_urls"],
                "domain": doc["external_domains"],
                "quoted_tweet_id": [doc["quoted_tweet_id"]],
                "quoted_user": [doc.get("quoted_user", {}).get("username", None)],
                "replied_to_tweet_id": [doc["replied_to_tweet_id"]],
                "replied_to_user": [
                    doc.get("replied_to_user", {}).get("username", None)
                ],
            }
        else:
            indexed = {
                "by": [doc["user"]["username"]],
                "retweeted_tweet_id": [doc["retweeted_tweet_id"]],
                "retweeted_user": [doc.get("retweeted_user", {}).get("username", None)],
            }

        for granularity, dt in rounded_times.items():
            indexed[f"created_{granularity}"] = [dt.isoformat()]

        return indexed
