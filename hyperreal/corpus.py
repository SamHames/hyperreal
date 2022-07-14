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
import gzip
import json
import re
from typing import Protocol, runtime_checkable
from xml.etree import ElementTree

from jinja2 import Template
from markupsafe import Markup

import hyperreal.utilities
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
            "text": hyperreal.utilities.tokens(doc["text"]),
        }

    def render_docs_html(self, doc_keys):
        """Return the given documents as HTML."""
        return [(key, doc["text"]) for key, doc in self.docs(doc_keys=doc_keys)]


class StackExchangeCorpus(SqliteBackedCorpus):

    CORPUS_TYPE = "StackExchangeCorpus"

    def replace_docs(self, posts_file, comments_file, users_file):
        """Completely replace the content of the corpus with the content of these files."""
        self.db.execute("pragma journal_mode=WAL")
        self.db.executescript(
            """
            drop table if exists Post;
            create table Post(
                Id integer primary key,
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
                PostType text
            );

            drop table if exists comment;
            create table comment(
                CreationDate datetime,
                ContentLicense text,
                Score integer,
                Text text,
                UserId integer references User(Id),
                PostId integer references Post(Id),
                Id integer primary key
            );

            create index post_comment on comment(PostId);

            drop table if exists User;
            create table User(
                AboutMe text,
                CreationDate datetime,
                Location text,
                ProfileImageUrl text,
                WebsiteUrl text,
                AccountId integer,
                Reputation integer,
                Id integer primary key,
                Views integer,
                UpVotes integer,
                DownVotes integer,
                DisplayName text,
                LastAccessDate datetime
            );

            drop table if exists PostTag;
            create table PostTag(
                PostId integer references Post,
                Tag text,
                primary key (PostId, Tag)
            );

            """
        )

        try:
            self.db.execute("begin")

            # Process Posts, which includes both questions and answers.
            tag_splitter = re.compile("<|>|<>")

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

                self.db.execute(
                    """
                    insert into post values (
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
                    (elem.attrib["Id"], t)
                    for t in tag_splitter.split(elem.attrib.get("Tags", ""))
                    if t
                )

                self.db.executemany("insert into PostTag values(?, ?)", tag_insert)

                # This is important when using iterparse to free memory from
                # processed nodes in the tree.
                elem.clear()

            tree = ElementTree.iterparse(comments_file, events=("end",))

            for event, elem in tree:

                doc = defaultdict(lambda: None)
                doc.update(elem.attrib)

                self.db.execute(
                    """
                    insert into comment values (
                        :CreationDate,
                        :ContentLicense,
                        :Score,
                        :Text,
                        :UserId,
                        :PostId,
                        :Id
                    )
                    """,
                    doc,
                )
                elem.clear()

            tree = ElementTree.iterparse(users_file, events=("end",))

            for event, elem in tree:

                doc = defaultdict(lambda: None)
                doc.update(elem.attrib)

                self.db.execute(
                    """
                    insert into user values (
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
                        :LastAccessDate
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
                        select 
                            Title,
                            Body,
                            -- Used to retrieve tags for the root question, as
                            -- the tags are not present on answers, only questions.
                            coalesce(ParentId, Id) as TagPostId
                        from post 
                        where post.Id = ?
                        """,
                        [key],
                    )
                )[0]

                # Use the tags on the question, not the (absent) tags on the answer.
                doc["Tags"] = [
                    r["Tag"]
                    for r in self.db.execute(
                        "select Tag from PostTag where PostId = ?", [doc["TagPostId"]]
                    )
                ]

                # Use the tags on the question, not the (absent) tags on the answer.
                doc["UserComments"] = list(
                    self.db.execute(
                        "select UserId, Text from comment where PostId = ?",
                        [key],
                    )
                )

                yield key, doc

        finally:
            self.db.execute("release docs")

    TEMPLATE = Template(
        """
        <details>
            <summary>{{ base_fields["PostType"] }}: "{{ base_fields["QuestionTitle"] }}"</summary>
            
            {{ base_fields["Body"] }}

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
                        <li>{{ comment["Text"] }}</li>
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
                    post.PostType, 
                    post.Body,
                    coalesce(Post.Title, parent.Title) as QuestionTitle
                from Post 
                left outer join Post parent 
                    on parent.Id = Post.ParentId
                where Post.Id = ?
                """,
                [key],
            )
        )[0]

        tags = "\n".join(
            f"<li>  { r['Tag'] }"
            for r in self.db.execute("select Tag from PostTag where PostId = ?", [key])
        )

        user_comments = list(
            self.db.execute(
                "select UserId, Text from comment where PostId = ?",
                [key],
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
        return (r["Id"] for r in self.db.execute("select Id from post"))

    def index(self, doc):
        return {
            "Post": hyperreal.utilities.tokens(
                hyperreal.utilities.strip_tags(
                    (doc["Title"] or "") + " " + (doc["Body"] or "")
                )
            ),
            "Tag": doc["Tags"],
            # Comments from deleted users remain, but have no UserId associated.
            "UsersCommenting": [
                u["UserId"] for u in doc["UserComments"] if u["UserId"] is not None
            ],
            "Comment": [
                t
                for c in doc["UserComments"]
                for t in hyperreal.utilities.tokens(c["Text"])
            ],
        }
