"""
An Index is a boolean inverted index, mapping field, value tuples in documents
to document keys.

"""

import collections
import heapq
import multiprocessing as mp
import sqlite3 as lite

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

    def index(self, raise_on_missing=True, n_cpus=4, batch_key_size=1000):
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

        manager = mp.Manager()
        mp_context = mp.get_context("spawn")

        # Note the size limit here - this means we won't materialise too many
        # key objects in memory.
        in_queue = manager.Queue(1000)

        # Create bounded output queues for each process. The bound puts
        # is for backpressure.
        out_queues = [manager.Queue(10) for _ in range(n_cpus)]

        try:
            # we will associate document keys to internal document ids
            # sequentially
            doc_keys = enumerate(corpus.keys())

            # We're still inside a transaction here, so processes reading from
            # the index won't see any of these changes until the release at
            # the end.
            self.db.execute("delete from doc_key")

            doc_ids = []
            doc_keys = []
            batch_size = 0

            # Start all of the worker processes.
            workers = [
                mp_context.Process(
                    target=_index_docs, args=(self.corpus, in_queue, out_queue)
                )
                for out_queue in out_queues
            ]

            for w in workers:
                w.start()

            for doc_key in doc_keys:
                self.db.execute("insert into doc_key values(?, ?)", doc_key)
                doc_ids.append(doc_key[0])
                doc_keys.append(doc_key[1])
                batch_size += 1

                if batch_size >= batch_key_size:
                    in_queue.put((doc_ids, doc_keys))
                    doc_ids = []
                    doc_keys = []
                    batch_size = 0
            else:
                if doc_ids:
                    in_queue.put((doc_ids, doc_keys))

            # Wait for everything to be indexed, then start aggregating and saving the results.
            to_merge = heapq.merge(iter(out_queue, None) for out_queue in out_queues)

            self.db.execute("delete from inverted_index")

            current_field, current_value, current_docs = next(to_merge)

            for field, value, doc_ids in segments:
                if (field, value) == (current_field, current_value):
                    # still aggregating this field...
                    current_docs |= doc_ids
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
                    current_docs = doc_ids
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

        except Exception:
            self.db.execute("rollback to reindex")

        finally:
            # Make sure to nicely cleanup all of the multiprocessing bits and bobs.
            self.db.execute("release reindex")

            for w in workers:
                w.join()

            in_queue.close()
            for q in out_queues:
                q.close()


def _index_docs(corpus, input_queue, output_queue, max_batch_entries):

    local_db = lite.connect(":memory:", isolation_level=None)
    # Note that we create an in memory database, but use a temporary table anyway,
    # because an in-memory database can't spill to disk.
    local_db.execute("create temporary table inverted_index (field, values, doc_ids)")

    batch_entries = 0

    # This is {field: {value: doc_ids, value2: doc_ids}}
    batch = collections.defaultdict(lambda: collections.defaultdict(BitMap))

    # Index documents until we go over max_batch_entries, then flush that
    # to a local temporary table.
    for doc_ids, doc_keys in iter(input_queue, None):

        docs = corpus.docs(doc_keys=doc_keys)

        for doc_id, doc in zip(doc_ids, docs):
            features = corpus.index(doc)
            for field, values in features.items():

                batch_entries += len(values)

                for value in values:
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

    # Once we have the pieces processed, we aggregate locally, then return
    # them to the main process via the output_queue for final aggregation
    segments = local_db.execute(
        "select field, value, doc_ids from inverted_index order by field, value"
    )

    current_field, current_value, doc_ids = next(segments)
    current_docs = BitMap.deserialize(doc_ids)

    for field, value, doc_ids in segments:
        if (field, value) == (current_field, current_value):
            # still aggregating this field...
            current_docs |= BitMap.deserialize(doc_ids)
            to_send = False
        else:
            # We've hit something new...
            output_queue.put((current_field, current_value, current_docs))
            (current_field, current_value) = (field, value)
            current_docs = BitMap.deserialize(doc_ids)
            to_send = True
    else:
        if to_send:
            output_queue.put((current_field, current_value, current_docs))

    output_queue.put(None)
    output_queue.close()
