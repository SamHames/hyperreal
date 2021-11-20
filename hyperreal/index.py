"""
An Index is a boolean inverted index, mapping field, value tuples in documents
to document keys.

"""

import collections
import heapq
import multiprocessing as mp
import os
import sqlite3 as lite
import tempfile

from pyroaring import BitMap, FrozenBitMap


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
                create table if not exists doc_key (
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

    def index(
        self,
        raise_on_missing=True,
        n_cpus=None,
        batch_key_size=1000,
        max_batch_entries=50_000_000,
    ):
        """
        Indexes the corpus, using the provided function or the corpus default.

        This method will index the entire corpus from scratch. If the corpus has
        already been indexed, it will be atomically replaced.

        Implementation notes:

        - aims to only load and process small batches in parallel in the worker
          threads
        - provide back pressure, to prevent too much state being held in main
          memory at the same time.

        """
        self.db.execute("savepoint reindex")

        n_cpus = n_cpus or mp.cpu_count()

        mp_context = mp.get_context("spawn")
        manager = mp_context.Manager()

        # Note the size limit here - this means we won't materialise too many
        # key objects in memory.
        in_queue = manager.Queue(n_cpus * 10)

        try:
            # We're still inside a transaction here, so processes reading from
            # the index won't see any of these changes until the release at
            # the end.
            self.db.execute("delete from doc_key")

            # we will associate document keys to internal document ids
            # sequentially
            doc_keys = enumerate(self.corpus.keys())

            batch_doc_ids = []
            batch_doc_keys = []
            batch_size = 0

            with tempfile.TemporaryDirectory() as tempdir:

                temporary_db_paths = [
                    os.path.join(tempdir, str(i)) for i in range(n_cpus)
                ]

                # Start all of the worker processes.
                workers = [
                    mp_context.Process(
                        target=_index_docs,
                        args=(
                            self.corpus,
                            in_queue,
                            temp_db_path,
                            max_batch_entries,
                        ),
                    )
                    for temp_db_path in temporary_db_paths
                ]

                for w in workers:
                    w.start()

                for doc_key in doc_keys:

                    self.db.execute("insert into doc_key values(?, ?)", doc_key)
                    batch_doc_ids.append(doc_key[0])
                    batch_doc_keys.append(doc_key[1])
                    batch_size += 1

                    if batch_size >= batch_key_size:
                        in_queue.put((batch_doc_ids, batch_doc_keys))
                        batch_doc_ids = []
                        batch_doc_keys = []
                        batch_size = 0
                else:
                    if batch_doc_ids:
                        in_queue.put((batch_doc_ids, batch_doc_keys))

                for w in workers:
                    in_queue.put(None)

                for w in workers:
                    w.join()

                temp_dbs = [
                    lite.connect(temp_db_path) for temp_db_path in temporary_db_paths
                ]

                queries = [
                    db.execute("select * from inverted_index order by field, value")
                    for db in temp_dbs
                ]

                to_merge = heapq.merge(*queries)

                self.__write_merged_segments(to_merge)

        except Exception:
            self.db.execute("rollback to reindex")
            raise

        finally:
            # Make sure to nicely cleanup all of the multiprocessing bits and bobs.
            self.db.execute("release reindex")
            manager.shutdown()

    def __write_merged_segments(self, to_merge):

        self.db.execute("delete from inverted_index")

        current_field, current_value, doc_ids = next(to_merge)
        current_docs = BitMap.deserialize(doc_ids)

        for field, value, doc_ids in to_merge:
            if (field, value) == (current_field, current_value):
                # still aggregating this field...
                current_docs |= BitMap.deserialize(doc_ids)
                to_send = False
            else:
                # We've hit something new...
                # output_queue.put((current_field, current_value, current_docs))
                self.db.execute(
                    "insert into inverted_index values(?, ?, ?, ?)",
                    [
                        current_field,
                        current_value,
                        len(current_docs),
                        current_docs.serialize(),
                    ],
                )
                (current_field, current_value) = (field, value)
                current_docs = BitMap.deserialize(doc_ids)
                to_send = True
        else:
            if to_send:
                self.db.execute(
                    "insert into inverted_index values(?, ?, ?, ?)",
                    [
                        current_field,
                        current_value,
                        len(current_docs),
                        current_docs.serialize(),
                    ],
                )

    def simple_index(self, max_batch_entries=200_000_000):

        self.db.execute("savepoint simple_index")

        self.db.execute(
            "create temporary table inverted_index_segment(field, value, doc_ids)"
        )
        self.db.execute("delete from doc_key")

        batch = collections.defaultdict(lambda: collections.defaultdict(BitMap))
        batch_entries = 0

        def write_batch():
            for field, values in batch.items():
                self.db.executemany(
                    "insert into inverted_index_segment values(?, ?, ?)",
                    (
                        (field, value, doc_ids.serialize())
                        for value, doc_ids in values.items()
                    ),
                )
            return 0, collections.defaultdict(lambda: collections.defaultdict(BitMap))

        for doc_id, (doc_key, doc) in enumerate(self.corpus):

            self.db.execute("insert into doc_key values(?, ?)", (doc_id, doc_key))

            features = self.corpus.index(doc)

            for field, values in features.items():

                set_values = set(values)
                batch_entries += len(set_values)

                for value in set_values:
                    batch[field][value].add(doc_id)

            if batch_entries >= max_batch_entries:
                batch_entries, batch = write_batch()

        else:
            batch_entries, batch = write_batch()

        to_merge = self.db.execute(
            "select * from inverted_index_segment order by field, value"
        )

        self.__write_merged_segments(to_merge)

        self.db.execute("release simple_index")


def _index_docs(corpus, input_queue, temp_db_path, max_batch_entries):

    local_db = lite.connect(temp_db_path, isolation_level=None)
    # Note that we create an in memory database, but use a temporary table anyway,
    # because an in-memory database can't spill to disk.
    local_db.execute("begin")
    local_db.execute("create table inverted_index(field, value, doc_ids)")

    batch_entries = 0

    # This is {field: {value: doc_ids, value2: doc_ids}}
    batch = collections.defaultdict(lambda: collections.defaultdict(BitMap))

    # Index documents until we go over max_batch_entries, then flush that
    # to a local temporary table.
    for doc_ids, doc_keys in iter(input_queue.get, None):

        docs = corpus.docs(doc_keys=doc_keys)

        for doc_id, (_, doc) in zip(doc_ids, docs):
            features = corpus.index(doc)
            for field, values in features.items():

                set_values = set(values)
                batch_entries += len(set_values)

                for value in set_values:
                    batch[field][value].add(doc_id)

            if batch_entries >= max_batch_entries:
                for field, values in batch.items():
                    local_db.executemany(
                        "insert into inverted_index values(?, ?, ?)",
                        (
                            (field, value, doc_ids.serialize())
                            for value, doc_ids in sorted(values.items())
                        ),
                    )
                batch = collections.defaultdict(lambda: collections.defaultdict(BitMap))
                batch_entries = 0

    else:
        for field, values in batch.items():
            local_db.executemany(
                "insert into inverted_index values(?, ?, ?)",
                (
                    (field, value, doc_ids.serialize())
                    for value, doc_ids in values.items()
                ),
            )

    # TODO: Merge the components locally first?
    local_db.execute("create index field_values on inverted_index(field, value)")
    local_db.execute("commit")


if __name__ == "__main__":
    import corpus

    c = corpus.PlainTextSqlite("big.db")

    i = Index("big_index.db", c)

    i.index(n_cpus=6)
    # i.simple_index()
