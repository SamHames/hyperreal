"""
An Index is a boolean inverted index, mapping field, value tuples in documents
to document keys.

"""

import array
import collections
from collections.abc import Sequence
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

    Uses savepoints - if no encloding transaction is present, this will create
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

        # Note - foreign key constraints handle all of the associated
        # metadata.
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

        self.logger.info(f"Initialised new model with {n_clusters}.")

    @atomic(writes=True)
    def delete_clusters(self, cluster_ids):
        """Delete the specified clusters."""
        self.db.executemany(
            "delete from cluster where cluster_id = ?", [[c] for c in cluster_ids]
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

    @atomic(writes=True)
    def create_cluster_from_features(self, feature_ids):
        """
        Create a new cluster from the provided set of features.

        The features must exist in the index.

        """

        next_cluster_id = list(
            self.db.execute(
                """
            select
                coalesce(max(cluster_id), 0) + 1
            from cluster
            """
            )
        )[0][0]

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

    @property
    def cluster_ids(self):
        return [r[0] for r in self.db.execute("select cluster_id from cluster")]

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
            (cluster_id, docs_count, self.cluster_features(cluster_id, limit=top_k))
            for cluster_id, docs_count in cluster_docs
        ]

        return clusters

    @atomic()
    def pivot_clusters_by_query(self, query, cluster_ids=None, top_k=20):

        cluster_ids = cluster_ids or self.cluster_ids
        work = [(self.db_path, query, cluster_id, top_k) for cluster_id in cluster_ids]

        # Calculate and filter out anything that has no matches at all with the query.
        pivoted = (r for r in self.pool.map(_pivot_cluster_by_query, work) if r[1])

        clusters = [
            (r[0][1], r[0][0], r[1])
            for r in sorted(
                pivoted,
                reverse=True,
                key=lambda r: r[0],
            )
        ]

        return clusters

    def cluster_features(self, cluster_id, limit=2**62):
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
                [cluster_id, limit],
            )
        )

        return cluster_features

    @atomic()
    def cluster_query(self, cluster_id):
        """
        Return matching documents and accumulated bitslice for cluster_id.

        The matching documents are the documents that contain any terms from
        the cluster. The returned bitslice represents the accumulation of
        features matching across all features and can be used for ranking
        with `utilities.bstm`.

        """

        feature_ids = [r[0] for r in self.cluster_features(cluster_id)]

        future = self.pool.submit(_union_bitslice, [self.db_path, None, feature_ids])

        return future.result()[1:]

    def _calculate_assignments(self, group_features, group_checks):
        """
        Determine the assignments of features to clusters given these features groups.

        This is a building block of the various iterations of refinement operations.

        group_features: a mapping of group_keys to lists of features in that group.
        group_checks: a mapping of group keys to the features to be check against
            that group.

        """

        # Dispatch the comparisons that involve the most features first, as they will
        # likely take the longest.
        check_order = sorted(
            (
                (
                    self.db_path,
                    group_key,
                    features,
                    group_checks[group_key],
                )
                for group_key, features in group_features.items()
                if group_key in group_checks
            ),
            key=lambda x: len(x[2]) + len(x[3]),
            reverse=True,
        )

        futures = [
            self.pool.submit(measure_features_to_feature_group, *c) for c in check_order
        ]

        total_objective = 0
        assignments = collections.defaultdict(lambda: (-math.inf, -1))

        for f in cf.as_completed(futures):

            result = f.result()

            if result is None:
                continue
            else:
                group_key, check_features, delta, already_in, objective = result

            total_objective += objective

            for feature, delta in zip(check_features, delta):

                assignments[feature] = max(assignments[feature], (delta, group_key))

        return assignments, total_objective

    def _refine_feature_groups(
        self,
        cluster_feature: dict[Hashable, set[int]],
        iterations: int = 10,
        sub_iterations: int = 2,
        group_test: bool = False,
    ):
        """
        Low level function for iteratively refining a feature clustering.

        cluster_features is a mapping from a cluster_key to a set of feature_ids.

        This is most useful if you want to explore specific clustering
        approaches without the constraint of the saved clusters.

        """

        feature_cluster = {
            feature_id: cluster_id
            for cluster_id, features in cluster_feature.items()
            for feature_id in features
        }

        # Make sure to copy the input dict
        cluster_feature = {
            cluster_id: features.copy()
            for cluster_id, features in cluster_feature.items()
        }

        features = list(feature_cluster)

        # Prune empty clusters.
        cluster_ids = [
            cluster_id for cluster_id, features in cluster_feature.items() if features
        ]
        total_objective = 0

        available_workers = self.pool._max_workers

        for iteration in range(iterations):

            self.logger.info(f"Starting iteration {iteration + 1}/{iterations}")

            # We want a completely different order of feature checks on each
            # iteration. This randomised ordering is also used to break into
            # sub batches for checking.
            self.random.shuffle(features)

            # More subiterations --> Less perturbation of the model for each
            # set of features, at the cost of more time for each subiteration.
            for sub_iter in range(sub_iterations):

                futures: set[cf.Future] = set()
                # This approach ensures that each subiteration is of
                # approximately the same size with respect to the number of
                # features.
                moving_features = set(features[sub_iter::sub_iterations])

                # Just like the features, we want random assignments of
                # clusters to batches in each subiteration.
                self.random.shuffle(cluster_ids)

                # We may lose clusters if everything in that cluster is moved
                # somewhere else
                n_clusters = len(cluster_ids)

                # Group testing only two instances doesn't make sense,
                # so if we have plenty of CPU availability, we can skip
                # straight to dense comparisons.
                cluster_worker_ratio = n_clusters / available_workers

                if group_test and cluster_worker_ratio >= 2:

                    # The square root heuristic here let's us spend roughly
                    # the same time on the group tests and the detailed
                    # tests.
                    n_batches = math.ceil(n_clusters**0.5)

                    # Assemble random batches of clusters to check against.
                    group_features = {
                        tuple(cluster_ids[i::n_batches]): {
                            feature
                            for cluster_id in cluster_ids[i::n_batches]
                            for feature in cluster_feature[cluster_id]
                        }
                        for i in range(n_batches)
                    }

                    # Check all of the features against all of the batches
                    group_tests = {
                        group_key: moving_features for group_key in group_features
                    }

                    batch_assignments, _ = self._calculate_assignments(
                        group_features, group_tests
                    )

                    # Convert the group tests into individual cluster tests
                    cluster_tests = collections.defaultdict(set)

                    for feature, (_, check_keys) in batch_assignments.items():
                        # Test against the current cluster
                        cluster_tests[feature_cluster[feature]].add(feature)

                        # Test against each of the clusters in the best batches
                        for cluster_id in check_keys:
                            cluster_tests[cluster_id].add(feature)

                # Too few clusters, or group testing turned off: check all against all
                else:

                    cluster_tests = {
                        cluster_id: moving_features for cluster_id in cluster_ids
                    }

                previous_objective = total_objective

                # Compute the final assignments
                assignments, total_objective = self._calculate_assignments(
                    cluster_feature, cluster_tests
                )

                # Unpack the beam search to assign to the nearest cluster
                # Note on the last iteration, the assignments will be used
                # to track the nearest neighbours for other uses.
                for feature, (delta, cluster_id) in assignments.items():

                    cluster_feature[feature_cluster[feature]].discard(feature)
                    cluster_feature[cluster_id].add(feature)
                    feature_cluster[feature] = cluster_id

                # Prune emptied clusters from ids
                cluster_ids = [
                    cluster_id
                    for cluster_id, values in cluster_feature.items()
                    if len(values)
                ]

        # prune empty clusters before returning
        cluster_feature = {
            cluster_id: features
            for cluster_id, features in cluster_feature.items()
            if features
        }

        return cluster_feature

    def propose_cluster_split(
        self,
        cluster_id,
        k: Optional[int] = None,
        iterations: int = 10,
        sub_iterations: int = 2,
        group_test: bool = True,
    ):
        """Compute a proposed split of the given cluster.

        k specifies the number of a splits, if left at the default of None, it
        will be automatically split as the sqrt(number of features).

        """
        cluster_features = self.cluster_features(cluster_id)
        k = k or math.ceil(len(cluster_features) ** 0.5)

        feature_ids = [r[0] for r in cluster_features]
        self.random.shuffle(feature_ids)

        split_cluster_features = {i: set(feature_ids[i::k]) for i in range(k)}

        return self._refine_feature_groups(
            split_cluster_features, iterations, sub_iterations, group_test
        )

    @atomic(writes=True)
    def refine_clusters(
        self,
        iterations: int = 10,
        sub_iterations: int = 2,
    ):
        """
        Attempt to improve the model clustering of the features for `iterations`.

        """

        if iterations < 1:
            raise ValueError(
                f"You must specificy at least one iteration, provided '{iterations}'."
            )
        if sub_iterations < 1:
            raise ValueError(
                f"You must specificy at least one subiteration, provided '{sub_iterations}'."
            )

        # Establish forward reverse mappings of features to clusters and vice versa.
        cluster_feature = collections.defaultdict(set)

        for feature_id, cluster_id in self.db.execute(
            """
            select
                feature_id,
                cluster_id
            from feature_cluster
            """
        ):
            cluster_feature[cluster_id].add(feature_id)

        cluster_feature = self._refine_feature_groups(
            cluster_feature,
            iterations=iterations,
            sub_iterations=sub_iterations,
            group_test=True,
        )

        # Serialise the actual results of the clustering!
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

        changed = self.db.execute("select cluster_id from changed_cluster")
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


def _pivot_cluster_by_query(args):

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
        ((*r[1:], r[0]) for r in results if r[1] >= 0), reverse=True, key=lambda r: r[3]
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


def measure_features_to_feature_group(
    index_db_path, group_key, feature_group, comparison_features
):
    """
    Measure the objective for moving the given subset of features into the
    given clusters.

    First loads the given cluster features into the representative centroid, then
    measures the improvement if each of comparison features is added in turn.

    """

    try:
        index = Index(index_db_path)

        if not feature_group:
            return None

        n_features = len(feature_group)

        return_size = len(comparison_features)
        return_features = []
        # Note that we're using arrays here just to minimise the number of objects
        # we're returning and serialising here.
        result_delta = array.array("d", (0 for _ in range(return_size)))
        array_index = 0

        # The union of all docs covered by the cluster
        cluster_union = BitMap()
        # The set of all docs covered at least twice.
        # This will be used to work out which documents are only covered once.
        covered_twice = BitMap()

        hits = 0

        # Construct the union of all cluster tokens, and also the set of
        # documents only covered by a single feature.
        for feature in feature_group:

            docs = index[feature]

            hits += len(docs)

            # Docs covered at least twice
            covered_twice |= cluster_union & docs
            # All docs now covered
            cluster_union |= docs

        only_once = cluster_union - covered_twice

        c = len(cluster_union)
        objective = hits / (c + n_features)

        assignments = []

        # All tokens that are adds (not already in the cluster)
        for feature in sorted(comparison_features - feature_group):
            docs = index[feature]

            feature_hits = len(docs)

            new_hits = hits + feature_hits
            new_c = docs.union_cardinality(cluster_union)
            new_objective = new_hits / (new_c + n_features + 1)

            delta = new_objective - objective

            return_features.append(feature)
            result_delta[array_index] = delta
            array_index += 1

        already_in = array_index

        # Features that are already in the cluster, so we need to calculate a remove operator.
        # Effectively we're counting the negative of the score for removing that feature
        # as the effect of adding it to the cluster.
        for feature in sorted(feature_group & comparison_features):
            docs = index[feature]

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

            return_features.append(feature)
            result_delta[array_index] = delta
            array_index += 1

    finally:
        index.close()

    return group_key, return_features, result_delta, already_in, objective


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


def _union_bitslice(args):

    index_db_path, query_key, feature_ids = args

    try:
        index = Index(index_db_path)
        matching = BitMap()
        bitslice = [BitMap()]

        for feature_id in feature_ids:
            doc_ids = index[feature_id]

            matching |= doc_ids

            for i, bs in enumerate(bitslice):
                carry = bs & doc_ids
                bs ^= doc_ids
                doc_ids = carry
                if not carry:
                    break

            if carry:
                bitslice.append(carry)

    finally:
        index.close()

    return query_key, matching, bitslice
