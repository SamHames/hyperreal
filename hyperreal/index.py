"""
An Index is a boolean inverted index, mapping field, value tuples in documents
to document keys.

"""

import array
import collections
from collections.abc import Sequence, Iterable
import concurrent.futures as cf
from functools import wraps
import heapq
import logging
import math
import multiprocessing as mp
import os
import random
import sqlite3
import tempfile
from typing import Any, Union, Hashable, Optional

from pyroaring import BitMap, FrozenBitMap, AbstractBitMap

from hyperreal import db_utilities, corpus, _index_schema, utilities


logger = logging.getLogger(__name__)


class CorpusMissingError(AttributeError):
    pass


class IndexingError(AttributeError):
    pass


FeatureKey = tuple[str, Any]
FeatureKeyOrId = Union[FeatureKey, int]
FeatureIdAndKey = tuple[int, str, Any]
BitSlice = list[BitMap]


def requires_corpus(func):
    """
    Mark method as requiring a corpus object.

    Raises an AttributeError if no corpus object is present on this index.

    """

    @wraps(func)
    def wrapper_func(self, *args, **kwargs):
        if self.corpus is None:
            raise CorpusMissingError(
                "A Corpus must be provided to the index for this functionality."
            )

        return func(self, *args, **kwargs)

    return wrapper_func


def atomic(writes=False):
    """
    Wrap the decorated interaction with SQLite in a transaction or savepoint.

    Uses savepoints - if no enclosing transaction is present, this will create
    one, if a transaction is in progress, this will be nested as a non durable
    savepoint within that transaction.

    By default, transactions are considered readonly - set this to false to
    mark when changes happen so that housekeeping functions can run at the
    end of a transaction.

    """

    def atomic_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            try:
                self._transaction_level += 1
                self.db.execute(f'savepoint "{func.__name__}"')

                results = func(*args, **kwargs)

                if writes:
                    self._changed = True

                return results

            except Exception:
                self.logger.exception("Error executing index method.")
                # Rewind to the previous savepoint, then release it
                # This is necessary to behave nicely whether we are operating
                # inside a larger transaction or just in autocommit mode.
                self.db.execute(f'rollback to "{func.__name__}"')
                raise

            finally:
                self._transaction_level -= 1

                # We've decremented to the final transaction level and are about
                # to commit.
                if self._transaction_level == 0 and self._changed:
                    self.logger.info(f"Changes detected - updating clusters.")
                    # Note that this will check for changed queries, and will
                    # therefore be a noop if there aren't any.
                    self._update_changed_clusters()
                    self._changed = False

                self.db.execute(f'release "{func.__name__}"')

        return wrapper

    return atomic_wrapper


class Index:
    def __init__(self, db_path, corpus=None, pool=None, random_seed=None):
        """
        The corpus object is optional - if not provided certain operations such
        as retrieving or rendering documents won't be possible.

        A concurrent.futures pool may be provided to control concurrency
        across different operations. If not provided, a pool will be initialised
        using within a `spawn` mpcontext.

        Note that the index is structured so that db_path is the only necessary
        state, and can always be reinitialised from just that path.

        A random seed can be specified - this will be used with the standard
        library's random module to fix the seed state + enable some kinds of
        reproducibility. Note that this isn't guaranteed to be consistent
        across Python versions.

        """
        self.db_path = db_path
        self.db = db_utilities.connect_sqlite(self.db_path)
        self.random = random.Random(random_seed)

        self._pool = pool

        for statement in """
            pragma synchronous=NORMAL;
            pragma foreign_keys=ON;
            pragma journal_mode=WAL;
            """.split(
            ";"
        ):
            self.db.execute(statement)

        migrated = _index_schema.migrate(self.db)

        if migrated:
            self.db.execute("begin")
            self._update_changed_clusters()
            self.db.execute("commit")

        self.corpus = corpus

        # For tracking the state of nested transactions. This is incremented
        # everytime a savepoint is entered with the @atomic() decorator, and
        # decremented on leaving. Housekeeping functions will run when leaving
        # the last savepoint by committing a transaction.
        self._transaction_level = 0
        self._changed = False

        # Set up a context specific adapter for this index.
        self.logger = logging.LoggerAdapter(logger, {"index_db_path": self.db_path})

    @property
    def pool(self):
        """Lazily initialised multiprocessing pool if none is provided on init."""
        if self._pool is None:
            self._pool = cf.ProcessPoolExecutor(mp_context=mp.get_context("spawn"))

        return self._pool

    @classmethod
    def is_index_db(cls, db_path):
        """Returns True if a db exists at db_path and is an index db."""
        try:
            db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            return (
                list(db.execute("pragma application_id"))[0][0]
                == _index_schema.MAGIC_APPLICATION_ID
            )
        except sqlite3.OperationalError:
            return False

    def close(self):
        self.db.close()

    @atomic()
    def __getitem__(self, key: Union[FeatureKey, int, slice]) -> BitMap:
        """key can either be a feature_id integer, a (field, value) tuple, or a slice of (field, value)'s indicating a range."""

        if isinstance(key, int):
            try:
                return list(
                    self.db.execute(
                        "select doc_ids from inverted_index where feature_id = ?",
                        [key],
                    )
                )[0][0]
            except IndexError:
                return BitMap()

        elif isinstance(key, tuple):
            try:
                return list(
                    self.db.execute(
                        "select doc_ids from inverted_index where (field, value) = (?, ?)",
                        key,
                    )
                )[0][0]
            except IndexError:
                return BitMap()

        elif isinstance(key, slice):
            if key.step is not None and key.step != 1:
                raise ValueError("Only stepsize of 1 is supported for slicing.")
            if key.start is None or key.stop is None:
                raise ValueError("Neither the start or end values can be None.")
            if key.start[0] != key.stop[0]:
                raise ValueError("Slicing is only supported on a single field.")

            results = BitMap()

            matching = self.db.execute(
                """
                select doc_ids
                from inverted_index
                where (field, value) >= (?, ?)
                    and (field, value) < (?, ?)
                """,
                (*key.start, *key.stop),
            )

            for (q,) in matching:
                results |= q

            return results

    def __setitem__(self, key: tuple[str, Any], doc_ids: AbstractBitMap) -> None:
        """
        Create a new feature in the index.

        Note:

        - Existing features are immutable and cannot be changed.
        - Features added in this way are not currently preserved on reindexing.

        """

        try:
            self.db.execute(
                "insert into inverted_index(field, value, docs_count, doc_ids) values(?, ?, ?, ?)",
                [*key, len(doc_ids), doc_ids],
            )
        except sqlite3.IntegrityError:
            raise KeyError(f"Feature {key} already exists.")

    def lookup_feature(self, feature_id: int) -> tuple[str, Any]:
        """Lookup the (field, value) for this feature by feature_id."""

        results = list(
            self.db.execute(
                "select field, value from inverted_index where feature_id = ?",
                [feature_id],
            )
        )

        if results:
            return results[0]
        else:
            raise KeyError(f"Feature with id '{feature_id}' not found.")

    def lookup_feature_id(self, key: tuple[str, Any]) -> int:
        """Lookup the (field, value) for this feature by feature_id."""

        results = list(
            self.db.execute(
                "select feature_id from inverted_index where (field, value) = (?, ?)",
                key,
            )
        )

        if results:
            return results[0][0]
        else:
            raise KeyError(f"Feature with key '{key}' not found.")

    @requires_corpus
    def index(
        self,
        doc_batch_size=1000,
        merge_batches=10,
        working_dir=None,
        skipgram_window_size=0,
        skipgram_min_docs=3,
    ):
        """
        Indexes the corpus, using the provided function or the corpus default.

        This method will index the entire corpus from scratch. If the corpus has
        already been indexed, it will be atomically replaced.

        By default, a temporary directory will be created to store temporary
        files which will be cleaned up at the end of the process. If
        `working_dir` is provided, temporary files will be stored there and
        cleaned up as processing continues, but the directory itself won't be
        cleaned up at the end of the process.

        Implementation notes:

        - aims to only load and process small batches in parallel in the worker
          threads: documents will be streamed through so that memory is used
          only for storing the incremental index results
        - limits the number of batches in flight at the same time
        - incrementally merges background batches to a single file
        - new index content is created in the background, and indexed content is
          written to the index in the background.

        """

        workers = self.pool._max_workers

        try:
            self.db.execute("begin")
        except sqlite3.OperationalError:
            raise IndexingError(
                "The `index` method can't be called in a nested transaction."
            )

        if merge_batches > 10:
            raise ValueError(f"merge_batches must be <= 10.")

        try:
            tempdir = working_dir or tempfile.TemporaryDirectory()
            current_temp_file_counter = 0

            def get_next_temp_file():
                nonlocal current_temp_file_counter
                temp_file_path = os.path.join(
                    tempdir.name, str(current_temp_file_counter)
                )
                current_temp_file_counter += 1
                return temp_file_path

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

            futures = set()
            to_merge = collections.defaultdict(list)

            self.logger.info("Beginning indexing.")

            for doc_key in doc_keys:
                self.db.execute("insert into doc_key values(?, ?)", doc_key)
                batch_doc_ids.append(doc_key[0])
                batch_doc_keys.append(doc_key[1])
                batch_size += 1

                if batch_size >= doc_batch_size:
                    self.logger.debug("Dispatching batch for indexing.")

                    # Dispatch the batch
                    futures.add(
                        self.pool.submit(
                            _index_docs,
                            self.corpus,
                            batch_doc_ids,
                            batch_doc_keys,
                            get_next_temp_file(),
                            skipgram_window_size,
                        )
                    )
                    batch_doc_ids = []
                    batch_doc_keys = []
                    batch_size = 0

                    if len(futures) >= workers:
                        done, futures = cf.wait(futures, return_when="FIRST_COMPLETED")

                        for f in done:
                            try:
                                segment_path, level = f.result()
                                to_merge[level].append(segment_path)
                            except Exception:
                                self.logger.exception(
                                    "Caught indexing exception from background."
                                )
                                raise

                        # Note that there's an SQLite limit to how many db's
                        # we can attach at once, so we can't dispatch more than
                        # to_merge at any time, even if there's more available.
                        for level, paths in list(to_merge.items()):
                            if len(paths) >= merge_batches:
                                futures.add(
                                    self.pool.submit(
                                        _merge_indices,
                                        get_next_temp_file(),
                                        level + 1,
                                        *paths[:merge_batches],
                                    )
                                )
                                self.logger.debug(
                                    f"Dispatching batches {paths[:merge_batches]}for merging."
                                )
                                to_merge[level] = paths[merge_batches:]

            # Dispatch the final batch.
            if batch_doc_keys:
                self.logger.debug("Dispatching final batch for indexing.")
                futures.add(
                    self.pool.submit(
                        _index_docs,
                        self.corpus,
                        batch_doc_ids,
                        batch_doc_keys,
                        get_next_temp_file(),
                        skipgram_window_size,
                    )
                )

            self.logger.info("Waiting for batches to complete.")

            # Finalise all segment indexing
            while futures:
                done, futures = cf.wait(futures, return_when="FIRST_COMPLETED")

                for future in done:
                    segment_path, level = future.result()
                    to_merge[level].append(segment_path)

                for level, paths in list(to_merge.items()):
                    if len(paths) >= merge_batches:
                        futures.add(
                            self.pool.submit(
                                _merge_indices,
                                get_next_temp_file(),
                                level + 1,
                                *paths[:merge_batches],
                            )
                        )
                        to_merge[level] = paths[merge_batches:]

            self.logger.info("Merging batches.")
            # The merge strategy is different now: if we are below merge_batches left, just go
            # straight to that, otherwise dispatch as many chunks as possible
            to_merge = [p for paths in to_merge.values() for p in paths]

            while len(to_merge) > 1:
                futures = set()
                merge_blocks = math.ceil(len(to_merge) / merge_batches)
                # Now to try to dispatch more similar chunks of work, regardless of level.

                for i in range(merge_blocks):
                    futures.add(
                        self.pool.submit(
                            _merge_indices,
                            get_next_temp_file(),
                            level + 1,
                            *to_merge[i::merge_blocks],
                        )
                    )

                to_merge = []
                for f in cf.as_completed(futures):
                    to_merge.append(f.result()[0])

            merge_file = to_merge[0]

            self.logger.info("Finalising indexing.")

            # Now merge back to the original index, preserving feature_ids
            # if this is a reindex operation.
            # Deleted features will be removed
            self.db.execute("attach ? as merged", [merge_file])
            to_detach = True

            # Zero out existing features, but don't reassign them
            self.db.execute(
                "update inverted_index set docs_count = 0, doc_ids = ?",
                [BitMap()],
            )

            # Ensure there's a feature_id for every field, value
            self.db.execute(
                """
                insert or ignore into inverted_index
                    select null, field, value, 0, ?
                    from merged.inverted_index_segment
                    -- Assign smaller feature_ids to more
                    -- frequent features.
                    order by docs_count desc
                """,
                [BitMap()],
            )
            # Update all features that are present in the new indexing.
            self.db.execute(
                """
                update inverted_index set
                    (docs_count, doc_ids) = (
                        select
                            docs_count,
                            doc_ids
                        from merged.inverted_index_segment
                        where (field, value) = (inverted_index.field, inverted_index.value)
                    )
                where (field, value) in (
                    select field, value
                    from merged.inverted_index_segment
                )
                """
            )

            # Update docs_counts in the clusters
            self.db.execute(
                """
                update feature_cluster set
                    docs_count = (
                        select docs_count
                        from inverted_index ii
                        where ii.feature_id = feature_cluster.feature_id
                    )
                """
            )

            self.db.execute("DELETE from skipgram_count")
            self.db.execute(
                """
                INSERT into main.skipgram_count
                select
                    (
                        select feature_id
                        from inverted_index ii
                        where (msc.field, msc.value_a) = (ii.field, ii.value)
                    ),
                    (
                        select feature_id
                        from inverted_index ii2
                        where (msc.field, msc.value_b) = (ii2.field, ii2.value)
                    ),
                    distance,
                    docs_count
                from merged.skipgram_count msc
                where docs_count >= ?
                """,
                [skipgram_min_docs],
            )

            # Write the field summary
            self.db.execute("delete from field_summary")
            self.db.execute(
                """
                insert into field_summary
                select
                    field,
                    count(*) as distinct_values,
                    min(value) as min_value,
                    max(value) as max_value
                from inverted_index
                group by field
                """
            )

            # Update all cluster stats based on new index stats
            self.db.execute(
                "insert into changed_cluster select cluster_id from cluster"
            )

            self.db.execute("commit")
            self.db.execute("detach merged")

            self.db.execute("begin")
            self._update_changed_clusters()
            self.db.execute("commit")

        except Exception:
            self.logger.exception("Indexing failure.")
            self.db.execute("rollback")
            raise

        finally:
            tempdir.cleanup()

        self.logger.info("Indexing completed.")

    @atomic()
    def convert_query_to_keys(self, query):
        """Generate the doc_keys one by one for the given query."""

        for doc_id in query:
            doc_key = list(
                self.db.execute(
                    "select doc_key from doc_key where doc_id = ?", [doc_id]
                )
            )[0][0]
            yield doc_key

    @requires_corpus
    def docs(self, query):
        """Retrieve the documents matching the given query set."""
        keys = self.convert_query_to_keys(query)
        return self.corpus.docs(doc_keys=keys)

    @requires_corpus
    def render_docs(self, query, random_sample_size=None):
        """
        Return the rendered representation of the docs matching this query.

        Optionally take a random sample of documents before rendering.
        """

        if random_sample_size is not None:
            q = len(query)
            if q > random_sample_size:
                query = BitMap(
                    query[i] for i in self.random.sample(range(q), random_sample_size)
                )

        doc_keys = self.convert_query_to_keys(query)
        return self.corpus.render_docs_html(doc_keys)

    def indexed_field_summary(self):
        """
        Return a summary tables of the indexed fields.

        """
        return list(self.db.execute("select * from field_summary"))

    @atomic(writes=True)
    def initialise_clusters(self, n_clusters, min_docs=1, include_fields=None):
        """
        Initialise the model with the given number of clusters.

        Features that retrieve at least `min_docs` are randomly assigned to
        one of the given clusters.

        `include_fields` can be specified to limit initialising the model to features
        from only the selected fields.

        """

        # Note - foreign key constraints handle most of the associated metadata,
        # we just do one extra step to avoid a circular trigger
        self.db.execute("delete from feature_cluster")
        self.db.execute("delete from cluster")

        self.db.execute("create temporary table if not exists include_field(field)")
        self.db.execute("delete from include_field")

        if include_fields:
            self.db.executemany(
                "insert into include_field values(?)", [[f] for f in include_fields]
            )
        else:
            self.db.execute("insert into include_field select field from field_summary")

        feature_ids = list(
            self.db.execute(
                """
                select
                    feature_id,
                    docs_count
                from inverted_index
                inner join include_field using(field)
                where docs_count >= ?
                -- Note: specify the ordering to ensure reproducibility, as these
                -- results will be shuffled.
                order by feature_id
                """,
                [min_docs],
            )
        )

        self.random.shuffle(feature_ids)

        clusters = ((i, feature_ids[i::n_clusters]) for i in range(n_clusters))

        self.db.executemany(
            """
            insert into feature_cluster(cluster_id, feature_id, docs_count)
                values(?, ?, ?)
            """,
            (
                (cluster_id, *feature)
                for cluster_id, features in clusters
                for feature in features
            ),
        )

        self.db.execute("drop table include_field")

        self.logger.info(f"Initialised new model with {n_clusters} clusters.")

    @atomic(writes=True)
    def delete_clusters(self, cluster_ids):
        """Delete the specified clusters."""
        # The cluster table will be automatically updated by the housekeeping functionality
        self.db.executemany(
            "delete from feature_cluster where cluster_id = ?",
            [[c] for c in cluster_ids],
        )

        self.logger.info(f"Deleted clusters {cluster_ids}.")

    @atomic(writes=True)
    def merge_clusters(self, cluster_ids):
        """Merge all clusters into the first cluster_id in the provided list."""

        merge_cluster_id = cluster_ids[0]

        for cluster_id in cluster_ids[1:]:
            self.db.execute(
                "update feature_cluster set cluster_id=? where cluster_id=?",
                [merge_cluster_id, cluster_id],
            )

        self.logger.info(f"Merged {cluster_ids} into {merge_cluster_id}.")

        return merge_cluster_id

    @atomic(writes=True)
    def delete_features(self, feature_ids):
        """Delete the given features from the model."""
        self.db.executemany(
            "delete from feature_cluster where feature_id=?",
            [[f] for f in feature_ids],
        )

        self.logger.info(f"Delete features {feature_ids}.")

    def next_cluster_id(self):
        """Returns a new cluster_id, guaranteed to be higher than anything already assigned."""
        next_cluster_id = list(
            self.db.execute(
                """
                select
                    coalesce(max(cluster_id), 0) + 1
                from cluster
                """
            )
        )[0][0]

        return next_cluster_id

    @atomic(writes=True)
    def create_cluster_from_features(self, feature_ids):
        """
        Create a new cluster from the provided set of features.

        The features must exist in the index.

        """

        next_cluster_id = self.next_cluster_id()

        self.db.executemany(
            """
            insert or ignore into feature_cluster(feature_id, cluster_id, docs_count)
                values (
                    ?1,
                    ?2,
                    (
                        select docs_count from inverted_index where feature_id = ?1
                    )
                )
            """,
            [(f, next_cluster_id) for f in feature_ids],
        )

        self.db.executemany(
            "update feature_cluster set cluster_id = ? where feature_id = ?",
            [(next_cluster_id, f) for f in feature_ids],
        )

        self.logger.info(f"Created cluster {next_cluster_id} from {feature_ids}.")

        return next_cluster_id

    @atomic(writes=True)
    def pin_features(self, feature_ids: Sequence[int], pinned: bool = True):
        """
        Pin (or unpin) the given features.

        Pinning a feature prevents it from being moved during the clustering
        feature. This can be used to preserve interesting combinations of
        features together in the same cluster.

        """

        self.db.executemany(
            "update feature_cluster set pinned = ? where feature_id = ?",
            ((pinned, f) for f in feature_ids),
        )

    @atomic(writes=True)
    def pin_clusters(self, cluster_ids: Sequence[int], pinned: bool = True):
        """
        Pin (or unpin) the given clusters.

        A pinned cluster will not be modified by the automated clustering
        routine. This can be used to preserve useful clusters and allow
        remaining unpinned clusters to be refined further.
        """
        self.db.executemany(
            "update cluster set pinned = ? where cluster_id = ?",
            ((pinned, c) for c in cluster_ids),
        )

    @property
    def cluster_ids(self):
        return [
            r[0]
            for r in self.db.execute(
                "select cluster_id from cluster order by cluster_id"
            )
        ]

    @atomic()
    def top_cluster_features(self, top_k=20):
        """Return the top_k features according to the number of matching documents."""

        cluster_docs = self.db.execute(
            """
            select cluster_id, docs_count
            from cluster
            order by docs_count desc
            """
        )

        clusters = [
            (cluster_id, docs_count, self.cluster_features(cluster_id, top_k=top_k))
            for cluster_id, docs_count in cluster_docs
        ]

        return clusters

    @atomic()
    def pivot_clusters_by_query(
        self, query, cluster_ids=None, top_k=20, scoring="jaccard"
    ):
        """
        Sort all clusters and features within clusters by similarity with the probe query.

        Args:
            query: the query object as a bitmap of document IDs
            cluster_ids: an optional sequence
            top_k: the number of top features to return in each cluster
            scoring: The similarity scoring function, currently "jaccard"
                and "chi_squared" are supported.

        """

        cluster_ids = cluster_ids or self.cluster_ids

        if scoring == "jaccard":
            work = [
                (self.db_path, query, cluster_id, top_k) for cluster_id in cluster_ids
            ]
            pivoted = (
                r for r in self.pool.map(_pivot_cluster_by_query_jaccard, work) if r[1]
            )
        elif scoring == "chi_squared":
            N = list(self.db.execute("select count(*) from doc_key"))[0][0]
            work = [
                (self.db_path, query, cluster_id, top_k, N)
                for cluster_id in cluster_ids
            ]
            pivoted = (
                r
                for r in self.pool.map(_pivot_cluster_by_query_chi_squared, work)
                if r[1]
            )

        clusters = [
            (r[0][1], r[0][0], r[1])
            for r in sorted(
                pivoted,
                reverse=True,
                key=lambda r: r[0],
            )
        ]

        return clusters

    def cluster_features(self, cluster_id, top_k=2**62):
        """
        Returns an impact ordered list of features for the given cluster.

        If top_k is specified, only the top_k most frequent features by
        document count are returned in descending order.

        """
        cluster_features = list(
            self.db.execute(
                f"""
                select
                    feature_id,
                    field,
                    value,
                    -- Note that docs_count is denormalised to allow
                    -- a per cluster sorting of document count.
                    fc.docs_count
                from feature_cluster fc
                inner join inverted_index ii using(feature_id)
                where cluster_id = ?
                order by fc.docs_count desc
                limit ?
                """,
                [cluster_id, top_k],
            )
        )

        return cluster_features

    @atomic()
    def union_bitslice(self, features: Sequence[FeatureKeyOrId]):
        """
        Return matching documents and accumulated bitslice for the given set
        of features.

        """

        bitmaps = (self[feature] for feature in features)
        return utilities.compute_bitslice(bitmaps)

    @atomic()
    def cluster_query(self, cluster_id):
        """
        Return matching documents and accumulated bitslice for cluster_id.

        If you only need the matching documents, the `cluster_docs` method is
        faster as it retrieves a precomputed set of documents.

        The matching documents are the documents that contain any terms from
        the cluster. The returned bitslice represents the accumulation of
        features matching across all features and can be used for ranking
        with `utilities.bstm`.

        """

        feature_ids = [r[0] for r in self.cluster_features(cluster_id)]

        return self.union_bitslice(feature_ids)

    def cluster_docs(self, cluster_id: int) -> AbstractBitMap:
        """Return the bitmap of documents covered by this cluster."""
        return list(
            self.db.execute(
                "select doc_ids from cluster where cluster_id = ?", [cluster_id]
            )
        )[0][0]

    def _refine_feature_groups(
        self,
        cluster_feature: dict[Hashable, set[int]],
        iterations: int = 10,
        group_test: bool = True,
        minimum_cluster_features: int = 1,
        pinned_features: Optional[Iterable[int]] = None,
        probe_query: Optional[AbstractBitMap] = None,
        max_clusters: Optional[int] = None,
        tolerance: float = 0.001,
    ) -> dict[Hashable, set[int]]:
        """
        Low level function for iteratively refining a feature clustering.

        cluster_features is a mapping from a cluster_key to a set of feature_ids.

        This is most useful if you want to explore specific clustering
        approaches without the constraint of the saved clusters.

        To expand the number of clusters in a model, empty cluster_ids can be
        provided and will be used for additional clusters made by splitting
        the largest clusters up.

        Pinned features will not be considered as candidates for moving.

        If a probe_query is provided, it will be intersected with all features
        for clustering: this is used to generate a clustering for a subset of
        the data. Note that features may not intersect with the probe query -
        clustering is not well defined in this case and should be used with
        care.

        max_clusters: specifies a maximum number of clusters for the model: if
        there are more clusters, the clusters that contribute least to the
        objective will be dissolved. Dissolves will be conducted evenly per
        iteration, rather than all at once.

        tolerance: specifies a termination tolerance. If the relative change
        in objective is less than this after an iteration, or if fewer than
        this fraction of non-pinned features moves during an iteration,
        terminate early.

        """

        max_clusters = max_clusters or len(cluster_feature)

        # If cluster_feature is empty, return immediately
        if not len(cluster_feature.keys()):
            return cluster_feature

        # Make sure to copy the input dict
        cluster_feature = {
            cluster_id: features.copy()
            for cluster_id, features in cluster_feature.items()
        }

        pinned_features = set(pinned_features) if pinned_features else set()
        # The set of clusters with pinned features - these will be used to
        # avoid interfering with pinned features when dividing or dissolving
        # clusters.
        clusters_with_pinned_features = {
            cluster_id
            for cluster_id, features in cluster_feature.items()
            if features & pinned_features
        }
        movable_features = {
            f
            for features in cluster_feature.values()
            for f in features
            if f not in pinned_features
        }

        # Use initial cluster ids to keep track of whether any clusters
        # are fully emptied out and can be used to split larger clusters up.
        assigned_cluster_ids = set(cluster_feature)

        # Work out how many low objective clusters to dissolve on each iteration.
        dissolve_clusters = len(assigned_cluster_ids) - max_clusters
        # Note that we try to structure it so the very last iteration does not dissolve
        # anything.
        dissolve_per_iteration = math.ceil(dissolve_clusters / max(1, iterations - 1))
        dissolve_cluster_ids = set()

        available_workers = self.pool._max_workers

        current_cluster_scores = {}
        changed_clusters = set(cluster_feature)

        prev_obj = 0

        for iteration in range(iterations):
            # Prepation phase: handle empty/too small clusters, but taking
            # into account possible dissolves as well. Note that we should
            # only drop a cluster_id that was dissolved on the last
            # iteration - this allows us to be incremental in our approach.
            current_cluster_ids = {
                cluster_id
                for cluster_id, values in cluster_feature.items()
                # Keep every cluster above the minimum size, or if it has a pinned feature
                if len(values) >= minimum_cluster_features or values & pinned_features
            }

            too_small_cluster_ids = assigned_cluster_ids - current_cluster_ids

            # Actually dissolve the clusters from the model.
            for cluster_id in too_small_cluster_ids:
                cluster_feature[cluster_id] = set()

            if too_small_cluster_ids:
                logger.info(f"Filling {len(too_small_cluster_ids)} too-small clusters.")

            # The logic here is to: split the largest cluster recursively until
            # we have enough clusters. Maybe in the future this could be a split
            # and refine clustering instead of a simple random split?
            while too_small_cluster_ids:
                # Split the biggest cluster in two.
                n_features, split_id = max(
                    (
                        (len(c_features), cluster_id)
                        for cluster_id, c_features in cluster_feature.items()
                    )
                )

                small_id = too_small_cluster_ids.pop()

                changed_clusters.add(small_id)
                changed_clusters.add(split_id)

                logger.debug(
                    f"Filling cluster {small_id} by splitting largest "
                    f"cluster {split_id} with {n_features} features."
                )

                # Account for pinned features when splitting clusters - also
                # account for large numbers of features in a cluster being pinned.
                to_split = list(cluster_feature[split_id] - pinned_features)
                sample = self.random.sample(
                    to_split, min(len(to_split), n_features // 2)
                )

                for feature in sample:
                    cluster_feature[split_id].discard(feature)
                    cluster_feature[small_id].add(feature)

            # Compute current feature assignments and contributions to each cluster
            futures = {
                self.pool.submit(
                    measure_feature_contribution_to_cluster,
                    self.db_path,
                    cluster,
                    features,
                    probe_query,
                )
                for cluster, features in cluster_feature.items()
                if features and cluster in changed_clusters
            }

            for f in cf.as_completed(futures):
                r = f.result()
                current_cluster_scores[r[0]] = r[1:]

            total_objective = sum(r[2] for r in current_cluster_scores.values())
            self.logger.info(
                f"Iteration {iteration}, current objective: {total_objective}"
            )

            # Dissolve target low objective clusters for this iteration
            if dissolve_clusters:
                n_dissolve = min(dissolve_clusters, dissolve_per_iteration)
                dissolve_clusters -= n_dissolve
                dissolve_cluster_ids = set(
                    sorted(
                        current_cluster_scores.keys() - clusters_with_pinned_features,
                        key=lambda x: current_cluster_scores[x][2],
                    )[:n_dissolve]
                )
                self.logger.info(
                    f"Dissolving {len(dissolve_cluster_ids)} low objective clusters"
                )
            else:
                # Note that we only check the objective if there are no more
                # clusters to dissolve: otherwise we could leave the index in
                # an unexpected state.
                obj_delta = total_objective - prev_obj
                prev_obj = total_objective
                if (obj_delta / total_objective) < tolerance:
                    self.logger.info(
                        "Terminating refinement due to small change in the objective."
                    )
                    break
                dissolve_cluster_ids = set()

            assigned_cluster_ids -= dissolve_cluster_ids

            # Group testing for which specific cluster to check against. Note
            # that features are not tested against the group containing their
            # current cluster.

            # Start by generating the randomised cluster groups, accounting
            # for the dissolving clusters
            cluster_ids = list(current_cluster_scores.keys() - dissolve_cluster_ids)
            self.random.shuffle(cluster_ids)

            n_batches = math.ceil(len(cluster_ids) ** 0.5)

            if group_test and n_batches > 2:
                cluster_groups = [
                    tuple(cluster_ids[i::n_batches]) for i in range(n_batches)
                ]

                futures = set()

                for group in cluster_groups:
                    group_features = {
                        f for c in group for f in current_cluster_scores[c][0]
                    }
                    # Note that we don't test adding the feature to it's own group!
                    comparison_features = {
                        f
                        for c in current_cluster_scores.keys() - set(group)
                        for f in current_cluster_scores[c][0]
                    } - pinned_features
                    futures.add(
                        self.pool.submit(
                            measure_add_features_to_cluster,
                            self.db_path,
                            group,
                            group_features,
                            comparison_features,
                            probe_query,
                        )
                    )

                # Accumulate the best group to do a detailed check against
                best_groups = {}

                for f in cf.as_completed(futures):
                    group_key, feature_array, delta_array = f.result()

                    for feature, delta in zip(feature_array, delta_array):
                        # Note that we can't use a defaultdict for this, as
                        # we would need to know the structure of group_key for
                        # comparison purposes.
                        try:
                            best_groups[feature] = max(
                                best_groups[feature], (delta, group_key)
                            )
                        except KeyError:
                            best_groups[feature] = (delta, group_key)

                # Individual cluster checks against the selected comparison group
                comparison_features = collections.defaultdict(list)
                for feature, (_, group_key) in best_groups.items():
                    for c in group_key:
                        comparison_features[c].append(feature)

            else:
                comparison_features = {
                    c: movable_features - set(current_cluster[0])
                    for c, current_cluster in current_cluster_scores.items()
                    if c not in dissolve_cluster_ids
                }

            # Implementation detail: we're going to make this deterministic,
            # by processing return values in the order than they were submitted,
            # and *not* in the order they complete!
            futures = []

            for c in sorted(comparison_features):
                futures.append(
                    self.pool.submit(
                        measure_add_features_to_cluster,
                        self.db_path,
                        c,
                        current_cluster_scores[c][0],
                        comparison_features[c],
                        probe_query,
                    )
                )

            current_feature_scores = {
                # This will track delta, proposed cluster, existing cluster
                f: (
                    (d, c, c)
                    if c not in dissolve_cluster_ids
                    # Randomly choose a base cluster if the cluster is being dissolved.
                    else (-math.inf, self.random.choice(cluster_ids), c)
                )
                for c, (features, deltas, _) in current_cluster_scores.items()
                for f, d in zip(features, deltas)
            }

            changed_features = set()
            changed_clusters = set()

            # Make sure to process moves for dissolved clusters
            for cluster_id in dissolve_cluster_ids:
                changed_features |= cluster_feature[cluster_id]

            # Implementation note: cf.as_completed(futures) might be faster,
            # but is not deterministic like we'd want.
            for f in futures:
                test_cluster, feature_array, delta_array = f.result()

                # Note that we're not accumulating to the best possible score here!
                # We accept any change that improves the cluster score with a certain
                # probability. Note also that the group testing means that we won't
                # see all possible moves, there might be better moves we haven't
                # tested.
                for f, d in zip(feature_array, delta_array):
                    score, _, existing_cluster = current_feature_scores[f]

                    # Accept improvements with only a 50% probability. If
                    # there are multiple improvement options a move is
                    # therefore more likely.
                    if d > score and self.random.random() < 0.5:
                        changed_features.add(f)
                        current_feature_scores[f] = (
                            d,
                            test_cluster,
                            existing_cluster,
                        )

            # Actually assign to the new clusters, having done all of that.
            for f in changed_features:
                _, new_cluster, old_cluster = current_feature_scores[f]
                changed_clusters.add(new_cluster)
                changed_clusters.add(old_cluster)
                cluster_feature[old_cluster].discard(f)
                cluster_feature[new_cluster].add(f)

            for cluster_id in dissolve_cluster_ids:
                del current_cluster_scores[cluster_id]
                del cluster_feature[cluster_id]

            if not dissolve_cluster_ids:
                if (len(changed_features) / len(movable_features)) < tolerance:
                    self.logger.info(
                        "Terminating refinement due to small number of feature moves."
                    )
                    break

            self.logger.info(
                f"Finished iteration {iteration + 1}/{iterations}, {len(changed_clusters)} changed clusters, {len(changed_features)} features"
            )

        return cluster_feature

    @atomic()
    def refine_clusters(
        self,
        iterations: int = 10,
        cluster_ids: Optional[Sequence[int]] = None,
        target_clusters: Optional[int] = None,
        minimum_cluster_features: int = 1,
    ):
        """
        Refine the feature clusters for the current model.

        Optionally provide a list of specific cluster_ids to refine.

        If target_clusters is larger than the current number of clusters in
        the model, the largest clusters by number of features will be split
        to reach the target. This can be used to split all or some selected
        clusters.

        """

        cluster_ids = set(cluster_ids or self.cluster_ids)

        if iterations < 1:
            raise ValueError(
                f"You must specificy at least one iteration, provided '{iterations}'."
            )

        # Establish forward reverse mappings of features to clusters and vice versa.
        cluster_feature = collections.defaultdict(set)
        pinned_features = set()

        for feature_id, cluster_id, pinned in self.db.execute(
            """
            select
                feature_id,
                cluster_id,
                feature_cluster.pinned
            from feature_cluster
            """
        ):
            if cluster_id in cluster_ids:
                cluster_feature[cluster_id].add(feature_id)
                if pinned:
                    pinned_features.add(feature_id)

        # If the target clusters is greater than the current clusters, create
        # empty clusters to expand into. The other case where there are
        # more clusters than desired will be handled in _refine_feature_groups
        target_clusters = target_clusters or len(cluster_feature)
        n_empty_required = target_clusters - len(cluster_feature)

        if n_empty_required > 0:
            next_cluster_id = self.next_cluster_id()

            for i in range(n_empty_required):
                cluster_feature[next_cluster_id + i] = set()

        # Remove pinned clusters from refinement. Note we do this after
        # creating empty clusters because a pinned cluster still needs to be
        # counted.
        pinned_clusters = {
            r[0] for r in self.db.execute("select cluster_id from cluster where pinned")
        }

        for cluster_id in pinned_clusters:
            del cluster_feature[cluster_id]

        cluster_feature = self._refine_feature_groups(
            cluster_feature,
            iterations=iterations,
            group_test=True,
            pinned_features=pinned_features,
            minimum_cluster_features=minimum_cluster_features,
            max_clusters=target_clusters,
        )

        # Serialise the actual results of the clustering!
        self._update_cluster_feature(cluster_feature)

    @atomic(writes=True)
    def _update_cluster_feature(self, cluster_feature):
        """
        Update the given cluster: feature mapping.

        Note that this only updates the provided clusters: it does not replace
        the entire state of the model. Also note that this can clobber
        cluster_ids if you're not careful.

        """
        self.db.executemany(
            """
            update feature_cluster set cluster_id = ?1 where feature_id = ?2
            """,
            (
                (cluster_id, feature_id)
                for cluster_id, features in cluster_feature.items()
                for feature_id in features
            ),
        )

    def _update_changed_clusters(self):
        """
        Refresh cluster union queries for changed clusters.

        This is usually called from the `atomic` decorator, or when reindexing.

        It is assumed that this is called inside a transaction.

        """

        # First update the feature counts, and remove empty clusters
        self.db.execute(
            """
            update cluster set feature_count = (
                select count(*) 
                from feature_cluster fc
                where fc.cluster_id=cluster.cluster_id
            )
            where cluster_id in (select cluster_id from changed_cluster)
            """
        )

        changed = self.db.execute("select cluster_id from changed_cluster")

        # Then update the union statistics for all of the clusters
        bg_args = (
            (
                self.db_path,
                cluster_param[0],
                [
                    row[0]
                    for row in self.db.execute(
                        """
                        select feature_id
                        from feature_cluster
                        where cluster_id = ?
                        """,
                        cluster_param,
                    )
                ],
            )
            for cluster_param in changed
        )

        # Note that the data here may not have been committed yet, so we have
        # to read and pass the feature_ids to the background ourselves.
        for cluster_id, query, weight in self.pool.map(_union_query, bg_args):
            self.db.execute(
                """
                update cluster set (docs_count, weight, doc_ids) = (?, ?, ?)
                where cluster_id = ?
                """,
                (len(query), weight, query, cluster_id),
            )

        self.db.execute("delete from cluster where feature_count = 0")
        self.db.execute("delete from changed_cluster")


def _index_docs(corpus, doc_ids, doc_keys, temp_db_path, skipgram_window_size):
    """Index all of the given docs into temp_db_path."""

    local_db = db_utilities.connect_sqlite(temp_db_path)

    try:
        local_db.execute("begin")

        # This is {field: {value: doc_ids, value2: doc_ids}}
        batch = collections.defaultdict(lambda: collections.defaultdict(BitMap))
        # The structure is (distance, field, value_a, value_b: count)
        skipgram_counts = [
            collections.defaultdict(
                lambda: collections.defaultdict(collections.Counter)
            )
            for _ in range(skipgram_window_size)
        ]

        docs = corpus.docs(doc_keys=doc_keys)

        for doc_id, (_, doc) in zip(doc_ids, docs):
            features = corpus.index(doc)
            for field, values in features.items():
                # Only find bigrams in sequences - non sequence types such as
                # a set don't make sense to do this.
                if skipgram_window_size > 0 and isinstance(
                    values, collections.abc.Sequence
                ):
                    bigrams = utilities.long_distance_bigrams(
                        values, skipgram_window_size
                    )
                    for item_a, item_b, distance in bigrams:
                        skipgram_counts[distance - 1][field][item_a][item_b] += 1

                set_values = set(values)

                for value in set_values:
                    batch[field][value].add(doc_id)

        local_db.execute(
            """
            CREATE table inverted_index_segment(
                field text,
                value,
                docs_count,
                doc_ids roaring_bitmap,
                primary key (field, value)
            ) without rowid
            """
        )
        for field, values in batch.items():
            local_db.executemany(
                "insert into inverted_index_segment values(?, ?, ?, ?)",
                (
                    (field, value, len(doc_ids), doc_ids)
                    for value, doc_ids in values.items()
                    if value is not None
                ),
            )

        local_db.execute(
            """
            CREATE table skipgram_count(
                field text,
                value_a,
                value_b,
                distance integer,
                docs_count integer,
                primary key (field, value_a, value_b, distance)
            )
            """
        )

        for i, f in enumerate(skipgram_counts):
            distance = i + 1
            for field, item_as in f.items():
                for item_a, item_bs in item_as.items():
                    local_db.executemany(
                        "INSERT into skipgram_count values(?, ?, ?, ?, ?)",
                        (
                            (field, item_a, item_b, distance, c)
                            for item_b, c in sorted(item_bs.items())
                        ),
                    )

        local_db.execute("commit")

    finally:
        local_db.close()

    return temp_db_path, 0


def _merge_indices(target_db_path, level, *to_merge):
    """Merge the given indices into target_db_path."""

    target_db = db_utilities.connect_sqlite(target_db_path)

    try:
        target_db.execute("begin")

        for i, db in enumerate(to_merge):
            target_db.execute(f"attach ? as to_merge_{i}", [db])

        target_db.execute(
            """
            CREATE table inverted_index_segment(
                field text,
                value,
                docs_count,
                doc_ids roaring_bitmap,
                primary key (field, value)
            ) without rowid
            """
        )

        union_all = "\nunion all\n".join(
            f"SELECT * from to_merge_{i}.inverted_index_segment"
            for i in range(len(to_merge))
        )

        target_db.execute(
            f"""
            INSERT into inverted_index_segment
                select
                    field,
                    value,
                    sum(docs_count),
                    roaring_union(doc_ids)
                from (
                    {union_all}
                )
                group by field, value
            """
        )

        union_all = "\nunion all\n".join(
            f"SELECT * from to_merge_{i}.skipgram_count" for i in range(len(to_merge))
        )

        target_db.execute(
            """
            CREATE table skipgram_count(
                field text,
                value_a,
                value_b,
                distance integer,
                docs_count integer,
                primary key (field, value_a, value_b, distance)
            )
            """
        )

        target_db.execute(
            f"""
            INSERT into skipgram_count
                select
                    field,
                    value_a,
                    value_b,
                    distance,
                    sum(docs_count)
                from (
                    {union_all}
                )
                group by 1, 2, 3, 4
            """
        )

        target_db.execute("commit")

    finally:
        target_db.close()
        for path in to_merge:
            os.remove(path)

    return target_db_path, level


def _pivot_cluster_by_query_jaccard(args):
    index_db_path, query, cluster_id, top_k = args
    index = Index(index_db_path)

    results = [(0, -1, "", "")] * top_k

    q = len(query)

    search_upper = index.db.execute(
        """
        select
            feature_cluster.feature_id,
            field,
            value,
            feature_cluster.docs_count,
            doc_ids
        from feature_cluster
        inner join inverted_index using(feature_id)
        where cluster_id = ?
            and feature_cluster.docs_count >= ?
        order by feature_cluster.docs_count
        """,
        [cluster_id, q],
    )

    for feature_id, field, value, docs_count, docs in search_upper:
        # Early break if the length threshold can't be reached.
        if q / docs_count <= results[0][0]:
            break

        heapq.heappushpop(
            results, (query.jaccard_index(docs), feature_id, field, value)
        )

    search_upper = index.db.execute(
        """
        select
            feature_cluster.feature_id,
            field,
            value,
            feature_cluster.docs_count,
            doc_ids
        from feature_cluster
        inner join inverted_index using(feature_id)
        where cluster_id = ?
            and feature_cluster.docs_count < ?
        order by feature_cluster.docs_count desc
        """,
        [cluster_id, q],
    )

    for feature_id, field, value, docs_count, docs in search_upper:
        # Early break if the length threshold can't be reached.
        if docs_count / q <= results[0][0]:
            break

        heapq.heappushpop(
            results, (query.jaccard_index(docs), feature_id, field, value)
        )

    results = sorted(
        ((*r[1:], r[0]) for r in results if r[0] > 0), reverse=True, key=lambda r: r[3]
    )

    # Finally compute the similarity of the query with the cluster_union.
    cluster_union = list(
        index.db.execute(
            "select doc_ids from cluster where cluster_id = ?", [cluster_id]
        )
    )[0][0]

    similarity = query.jaccard_index(cluster_union)

    index.close()

    return (similarity, cluster_id), results


def _pivot_cluster_by_query_chi_squared(args):
    index_db_path, query, cluster_id, top_k, N = args
    index = Index(index_db_path)

    results = [(0, -1, "", "")] * top_k

    q = len(query)

    search = index.db.execute(
        """
        select
            feature_cluster.feature_id,
            field,
            value,
            feature_cluster.docs_count,
            doc_ids
        from feature_cluster
        inner join inverted_index using(feature_id)
        where cluster_id = ?
        """,
        [cluster_id],
    )

    for feature_id, field, value, docs_count, docs in search:
        f = docs_count

        # These are the cells in the 2x2 contingency table
        A = query.intersection_cardinality(docs)
        B = q - A
        C = f - A
        D = N - q - f + A

        score = (((A * D) - (B * C)) ** 2) * N / ((A + B) * (C + D) * (B + D) * (A + C))

        heapq.heappushpop(results, (score, feature_id, field, value))

    results = sorted(
        ((*r[1:], r[0]) for r in results if r[1] >= 0), reverse=True, key=lambda r: r[3]
    )

    # Finally compute the similarity of the query with the cluster_union.
    cluster_union = list(
        index.db.execute(
            "select doc_ids from cluster where cluster_id = ?", [cluster_id]
        )
    )[0][0]

    similarity = results[0][-1]

    index.close()

    return (similarity, cluster_id), results


def measure_feature_contribution_to_cluster(
    index_db_path,
    group_key,
    feature_group: set[int],
    probe_query: Optional[AbstractBitMap],
):
    """
    Measure the contribution of each feature to this cluster.

    The contribution is the delta between the objective of the cluster without
    the feature and with the feature.

    This function also has the side effect of calculating the objective
    contribution for this exact cluster.

    """

    try:
        index = Index(index_db_path)
        index.db.execute("begin")

        # FIRST PHASE: compute the objective and minimal cover stats for the
        # current cluster.

        # The union of all docs covered by the cluster
        cluster_union = BitMap()
        # The set of all docs covered at least twice.
        # This will be used to work out which documents are only covered once.
        covered_twice = BitMap()

        hits = 0
        n_features = len(feature_group)

        # Construct the union of all cluster tokens, and also the set of
        # documents only covered by a single feature.
        for feature in feature_group:
            docs = index[feature]

            if probe_query:
                docs &= probe_query

            hits += len(docs)

            # Docs covered at least twice
            covered_twice |= cluster_union & docs
            # All docs now covered
            cluster_union |= docs

        only_once = cluster_union - covered_twice

        c = len(cluster_union)
        objective = hits / (c + n_features)

        # SECOND PHASE: compute the incremental change in objective from removing each
        # feature (alone) from the current cluster.
        # Note: using an array to only manage two objects worth of de/serialisation

        feature_array = array.array("q", feature_group)
        delta_array = array.array("d", (0 for _ in feature_group))

        # Features that are already in the cluster, so we need to calculate a remove operator.
        # Effectively we're counting the negative of the score for removing that feature
        # as the effect of adding it to the cluster.
        for i, feature in enumerate(feature_array):
            docs = index[feature]

            if probe_query:
                docs &= probe_query

            feature_hits = len(docs)

            old_hits = hits - feature_hits
            old_c = c - docs.intersection_cardinality(only_once)

            # It's okay for the cluster to become empty - we'll just prune it.
            if old_c:
                old_objective = old_hits / (old_c + n_features - 1)
                delta = objective - old_objective
            # If it would otherwise be a singleton cluster, just mark it as no change
            else:
                delta = 0

            delta_array[i] = delta

    finally:
        index.db.execute("commit")
        index.close()

    return group_key, feature_array, delta_array, objective


def measure_add_features_to_cluster(
    index_db_path,
    group_key,
    feature_group: set[int],
    add_features: set[int],
    probe_query: Optional[AbstractBitMap],
):
    """
    Measure the incremental objective change from adding add_features to this cluster.

    If any of add_features are in feature_group already, incorrect results will be returned!

    """

    try:
        index = Index(index_db_path)
        index.db.execute("begin")

        # PHASE 1: Current cluster objective and cover.

        # The union of all docs covered by the cluster
        cluster_union = BitMap()

        hits = 0
        n_features = len(feature_group)

        # Construct the union of all cluster tokens, and also the set of
        # documents only covered by a single feature.
        for feature in feature_group:
            docs = index[feature]

            if probe_query:
                docs &= probe_query

            hits += len(docs)

            # All docs now covered
            cluster_union |= docs

        c = len(cluster_union)
        objective = hits / (c + n_features)

        # PHASE 2: Incremental delta from adding new features to the cluster.
        # Note: using an array to only manage two objects worth of de/serialisation
        feature_array = array.array("q", sorted(add_features))
        delta_array = array.array("d", (0 for _ in feature_array))

        # All tokens that are adds (not already in the cluster)
        for i, feature in enumerate(feature_array):
            docs = index[feature]

            if probe_query:
                docs &= probe_query

            feature_hits = len(docs)

            new_hits = hits + feature_hits
            new_c = docs.union_cardinality(cluster_union)
            new_objective = new_hits / (new_c + n_features + 1)

            delta = new_objective - objective

            delta_array[i] = delta

    finally:
        index.db.execute("commit")
        index.close()

    return group_key, feature_array, delta_array


def _union_query(args):
    index_db_path, query_key, feature_ids = args

    try:
        index = Index(index_db_path)
        query = BitMap()
        weight = 0

        for feature_id in feature_ids:
            docs = index[feature_id]
            query |= docs
            weight += len(docs)

    finally:
        index.close()

    return query_key, query, weight
