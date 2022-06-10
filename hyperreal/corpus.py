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
from typing import Protocol, runtime_checkable

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
        self.db = connect_sqlite(self.db_path)
        self.db.execute("pragma journal_mode=WAL")

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
        self.db.close()


class TidyTweetCorpus(SqliteBackedCorpus):

    CORPUS_TYPE = "TidyTweetCorpus"

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
                        "select text, author_id, created_at from tweet where id = ?",
                        [key],
                    )
                )[0]
                yield key, doc

        finally:
            self.db.execute("release docs")

    def keys(self):
        return (
            row[0]
            for row in self.db.execute(
                "select distinct id from tweet where directly_collected=1"
            )
        )

    def index(self, doc):
        return {
            "text": hyperreal.utilities.social_media_tokens(doc[0]),
            "author_id": [doc[1]],
            "created_at_utc_day": [doc[2][:10]],
        }


class PlainTextSqliteCorpus(SqliteBackedCorpus):

    CORPUS_TYPE = "PlainTextSqliteCorpus"

    def replace_docs(self, texts):
        """Replace the existing documents with texts."""
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

        try:
            # Note that it's valid to pass an empty sequence of doc_keys,
            # so we need to check sentinel explicitly.
            if doc_keys is None:
                doc_keys = self.keys()

            for key in doc_keys:
                doc = list(
                    self.db.execute(
                        "select text from doc where doc_id = ?",
                        [key],
                    )
                )[0][0]
                yield key, doc

        finally:
            self.db.execute("release docs")

    def keys(self):
        return (r[0] for r in self.db.execute("select doc_id from doc"))

    def index(self, doc):
        return {
            "text": hyperreal.utilities.tokens(doc),
        }

    def render_docs_html(self, doc_keys):
        """Return the given documents as HTML."""
        return list(self.docs(doc_keys=doc_keys))


class CirrusSearchWikiCorpus(SqliteBackedCorpus):

    CORPUS_TYPE = "CirrusSearchWikiCorpus"

    def ingest(self, filepath):

        self.db.executescript(
            """
            pragma foreign_keys=1;

            create table if not exists page(
                id text primary key,
                create_timestamp datetime,
                timestamp datetime,
                incoming_links integer,
                popularity_score float,
                text_bytes integer,
                text text,
                title text
            );

            create table if not exists page_external_link(
                id integer references page(id) on delete cascade,
                link text,
                primary key (id, link)
            );

            create table if not exists page_category(
                id integer references page(id) on delete cascade,
                category text,
                primary key (id, category)
            );

            create table if not exists page_outgoing_link(
                id integer references page(id) on delete cascade,
                link text,
                primary key (id, link)
            );
            """
        )

        try:
            self.db.execute("begin")

            docs_added = 0
            batch_docs_added = 0

            with gzip.open(filepath, "rt") as f:

                for page in f:
                    page_id = json.loads(page)["index"]["_id"]
                    content = defaultdict(lambda: None)
                    content.update(json.loads(next(f)))
                    content["id"] = page_id

                    self.db.execute(
                        """
                        replace into page values(
                            :id,
                            :create_timestamp,
                            :timestamp,
                            :incoming_links,
                            :popularity_score,
                            :text_bytes,
                            :text,
                            :title
                        )
                        """,
                        content,
                    )

                    if content["external_link"]:
                        self.db.executemany(
                            "insert into page_external_link values(?, ?)",
                            ((page_id, link) for link in content["external_link"]),
                        )

                    if content["category"]:
                        self.db.executemany(
                            "insert into page_category values(?, ?)",
                            ((page_id, link) for link in content["category"]),
                        )

                    if content["outgoing_link"]:
                        self.db.executemany(
                            "insert into page_outgoing_link values(?, ?)",
                            ((page_id, link) for link in content["outgoing_link"]),
                        )

                    docs_added += 1
                    batch_docs_added += 1

                    if batch_docs_added == 100000:
                        self.db.execute("commit")
                        batch_docs_added = 0
                        print(docs_added)
                        self.db.execute("begin")

            self.db.execute("commit")

        except Exception as e:
            self.db.execute("rollback")
            print(page, content)
            raise

    def docs(self, doc_keys=None):
        self.db.execute("savepoint docs")

        try:
            # Note that it's valid to pass an empty sequence of doc_keys,
            # so we need to check sentinel explicitly.
            if doc_keys is None:
                doc_keys = self.keys()

            for key in doc_keys:
                doc = list(self.db.execute("select * from page where id = ?", [key]))[0]

                doc["external_link"] = [
                    r["link"]
                    for r in list(
                        self.db.execute(
                            "select link from page_external_link where id = ?", [key]
                        )
                    )
                ]

                doc["category"] = [
                    r["category"]
                    for r in list(
                        self.db.execute(
                            "select category from page_category where id = ?", [key]
                        )
                    )
                ]

                doc["outgoing_link"] = [
                    r["link"]
                    for r in list(
                        self.db.execute(
                            "select link from page_outgoing_link where id = ?", [key]
                        )
                    )
                ]

                yield key, doc

        finally:
            self.db.execute("release docs")

    def keys(self):
        return (r["id"] for r in self.db.execute("select id from page"))

    def index(self, doc):
        return {
            "text": hyperreal.utilities.tokens(doc.get("text", "")),
            "title": hyperreal.utilities.tokens(doc.get("title", "")),
            "external_link": doc.get("external_link", []),
            "category": doc.get("category", []),
            "outgoing_link": doc.get("outgoing_link", []),
        }
