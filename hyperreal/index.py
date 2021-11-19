"""
An Index is a boolean inverted index, mapping features in documents
to document keys.

"""

import multiprocessing as mp
import sqlite3 as lite


class Index:
    def __init__(self, db_path, corpus, migrate=True):
        """ """
        self.db_path = db_path
        self.db = lite.connect(self.db_path, isolation_level=None)
        self.corpus = corpus

        for statement in """
            pragma synchronous=NORMAL;
            pragma foreign_keys=ON;
            pragma journal_mode=WAL;
            """.split(
            ";"
        ):
            self.db.execute(statement)

        if migrate:
            self.db.executescript(
                """
                create table if not exists key_document (
                    doc_id integer primary key,
                    doc_key unique
                );

                create table if not exists inverted_index (
                    field text,
                    value,
                    docs_count integer not null,
                    doc_ids blob not null,
                    primary key (field, value)
                );

                create index if not exists docs_counts on inverted_index(docs_count);
                """
            )

    def index(self, raise_on_missing=True):
        """
        Indexes the corpus, using the provided function or the corpus default.

        This method will index the entire corpus from scratch. If the corpus has
        already been indexed, it will be atomically replaced.

        """
        self.db.execute("savepoint reindex")

        try:

        except Exception:
            self.db.execute("rollback to reindex")
        finally:
            self.db.execute("release reindex")

        pass
