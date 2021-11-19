"""
For the purposes of hyperreal, a corpus is a collection of documents. A corpus
object is a Data Access Object (DAO), that is responsible for representing and
accessing documents stored elsewhere. The corpus is also responsible for
describing how those documents are to be represented in the index.

This module describes the protocol a Corpus class needs to implement, and also
shows concrete implementation examples.

"""

import sqlite3 as lite
from typing import Protocol

import utilities


class Corpus(Protocol):
    """
    - provides access to documents, and describes how to index them.
    - needs to be picklable and safely enable concurrent read access
    - Designed to enable small batches of work, and avoid representing entire
      collections in memory/work on collections much larger than memory.
    - Designed to enable downstream concurrent computing on those small batches.
    """

    @abstractmethod
    def docs(self, doc_keys=None, raise_on_missing=True):
        """
        Return an iterator of key-document pairs matching the given keys.

        If doc_keys is None, all documents will be iterated over.

        If a key is passed that isn't present, the default behaviour is to
        raise a KeyError, however in some applications it is advantageous to
        skip that.

        """
        pass

    @abstractmethod
    def keys(self):
        """An iterator of all document keys present."""
        pass

    @abstractmethod
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

    def __getitem__(self, doc_key):
        return self.docs([doc_key], raise_on_missing=True)

    def __iter__(self):
        return self.docs(doc_keys=None)


class TidyTweetCorpus(Corpus):
    def __init__(self, db_path):

        self.db_path = db_path
        self.db = lite.connect(self.db_path, isolation_level=None)
        self.db.row_factory = utilities.dict_factory

    def __getstate__(self):
        return self.db_path

    def __setstate__(self, db_path):
        self.__init__(db_path)

    def docs(self, doc_keys=None, raise_on_missing=True):
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
        return self.db.execute("select distinct id from tweet")

    def index(self, doc):
        return {
            "author_id": [doc["author_id"]],
            "created_at": [doc["created_at"]],
            "text": utilities.social_media_tokens(doc["text"]),
        }
