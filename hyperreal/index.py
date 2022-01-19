"""
An Index is a boolean inverted index, mapping field, value tuples in documents
to document keys.

"""

import array
import collections
import concurrent.futures as cf
import heapq
import math
import multiprocessing as mp
import os
import random
import tempfile


from pyroaring import BitMap, FrozenBitMap

from hyperreal import extensions, db_utilities


CURRENT_SCHEMA_VERSION = 2


class Index:
    def __init__(self, db_path, corpus=None, pool=None, mp_context=None):
        """

        If corpus is not provided, it will be deserialised from the corpus
        representation stored with the index.

        A pool and mp_context objects may be provided to control concurrency
        across different operations. If not provided, they will be initialised
        to a spawn server

        """
        self.db_path = db_path
        self.db = db_utilities.connect_sqlite(self.db_path)

        self.mp_context = mp_context or mp.get_context("spawn")
        self.pool = pool or cf.ProcessPoolExecutor(mp_context=self.mp_context)

        for statement in """
            pragma synchronous=NORMAL;
            pragma foreign_keys=ON;
            pragma journal_mode=WAL;
            """.split(
            ";"
        ):
            self.db.execute(statement)

        db_version = list(self.db.execute("pragma user_version"))[0][0]

        if db_version < CURRENT_SCHEMA_VERSION:
            self.db.executescript(
                """
                create table if not exists corpus_data (
                    id integer primary key,
                    corpus_type text,
                    data
                );

                create table if not exists doc_key (
                    doc_id integer primary key,
                    doc_key unique
                );

                create table if not exists inverted_index (
                    field text,
                    value,
                    docs_count integer not null,
                    doc_ids roaring_bitmap not null,
                    primary key (field, value)
                );

                create table if not exists field_summary (
                    field text primary key,
                    distinct_values integer,
                    min_value,
                    max_value
                );

                create index if not exists docs_counts on inverted_index(docs_count);

                -- The summary table for clusters, including the loose hierarchy
                -- and the materialised results of the query and document counts.
                create table if not exists cluster (
                    cluster_id integer primary key,
                    feature_count integer default 0
                );

                create table if not exists feature_cluster (
                    field text,
                    value,
                    cluster_id integer references cluster(cluster_id) on delete cascade,
                    docs_count integer,
                    primary key (field, value)
                );

                create index if not exists cluster_features on feature_cluster(
                    cluster_id,
                    docs_count
                );

                -- These triggers make sure that the cluster table always demonstrates
                -- which clusters are currently active, and allows the creation of tracking
                -- metadata for new clusters on insert of the features.
                create trigger if not exists ensure_cluster before insert on feature_cluster
                    begin
                        insert or ignore into cluster(cluster_id) values (new.cluster_id);
                        update cluster set
                            feature_count = feature_count + 1
                        where cluster_id = new.cluster_id;
                    end;

                create trigger if not exists delete_feature_cluster after delete on feature_cluster
                    begin
                        update cluster set
                            feature_count = feature_count - 1
                        where cluster_id = old.cluster_id;
                    end;

                create trigger if not exists ensure_cluster_update before update on feature_cluster
                    when old.cluster_id != new.cluster_id
                    begin
                        insert or ignore into cluster(cluster_id) values (new.cluster_id);
                    end;

                create trigger if not exists update_cluster_feature_counts after update on feature_cluster
                    when old.cluster_id != new.cluster_id
                    begin
                        update cluster set
                            feature_count = feature_count + 1
                        where cluster_id = new.cluster_id;
                        update cluster set
                            feature_count = feature_count - 1
                        where cluster_id = old.cluster_id;
                    end;

                pragma user_version = 2;
                """
            )
        elif db_version > CURRENT_SCHEMA_VERSION:
            raise ValueError(
                "Index database schema version is too new for this version."
            )

        if corpus:
            self.corpus = corpus
            self.save_corpus()
        else:
            self.load_corpus()

    def close(self):
        self.corpus.close()
        self.db.close()

    def load_corpus(self):
        self.db.execute("savepoint load_corpus")

        corpus_type, data = list(
            self.db.execute("select corpus_type, data from corpus_data where id = 0")
        )[0]

        self.corpus = extensions.registry[corpus_type].deserialize(data)
        self.db.execute("release load_corpus")

    def save_corpus(self):
        self.db.execute("savepoint save_corpus")

        self.db.execute(
            "replace into corpus_data values(0, ?, ?)",
            (self.corpus.CORPUS_TYPE, self.corpus.serialize()),
        )

        self.db.execute("release save_corpus")

    def __getstate__(self):
        return self.db_path

    def __setstate__(self, args):
        self.__init__(args, pool=1)

    def __getitem__(self, key):
        return list(
            self.db.execute(
                "select doc_ids from inverted_index where (field, value) = (?, ?)", key
            )
        )[0][0]

    def index(
        self,
        raise_on_missing=True,
        n_cpus=None,
        batch_key_size=1000,
        max_batch_entries=10_000_000,
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
        - a new database is created in the background, and overwrites the original
          database file only at the end of the process.

        """
        self.db.execute("savepoint reindex")

        n_cpus = n_cpus or mp.cpu_count()

        manager = self.mp_context.Manager()

        # Note the size limit here, to provide backpressure and avoid
        # materialising keys too far in advance of the worker processes.
        process_queue = manager.Queue(n_cpus * 3)

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

                # Dispatch all of the worker processes
                futures = [
                    self.pool.submit(
                        _index_docs,
                        self.corpus,
                        process_queue,
                        temp_db_path,
                        max_batch_entries,
                    )
                    for temp_db_path in temporary_db_paths
                ]

                for doc_key in doc_keys:

                    self.db.execute("insert into doc_key values(?, ?)", doc_key)
                    batch_doc_ids.append(doc_key[0])
                    batch_doc_keys.append(doc_key[1])
                    batch_size += 1

                    if batch_size >= batch_key_size:
                        process_queue.put((batch_doc_ids, batch_doc_keys))
                        batch_doc_ids = []
                        batch_doc_keys = []
                        batch_size = 0

                else:
                    if batch_doc_ids:
                        process_queue.put((batch_doc_ids, batch_doc_keys))

                for _ in range(n_cpus):
                    process_queue.put(None)

                temp_dbs = [
                    db_utilities.connect_sqlite(f.result())
                    for f in cf.as_completed(futures)
                ]

                queries = [
                    db.execute(
                        "select * from inverted_index_segment order by field, value"
                    )
                    for db in temp_dbs
                ]

                to_merge = heapq.merge(*queries)

                self.__write_merged_segments(to_merge)

                self.save_corpus()

        except Exception:
            self.db.execute("rollback to reindex")
            raise

        finally:
            # Make sure to nicely cleanup all of the multiprocessing bits and bobs.
            self.db.execute("release reindex")
            manager.shutdown()

    def __write_merged_segments(self, to_merge):

        self.db.execute("delete from inverted_index")

        current_field, current_value, current_docs = next(to_merge)

        for field, value, doc_ids in to_merge:
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
                        current_docs,
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
                        current_docs,
                    ],
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

    def convert_query_to_keys(self, query):
        """Generate the doc_keys one by one for the given query."""
        self.db.execute("savepoint get_doc_keys")

        for doc_id in query:
            yield list(
                self.db.execute(
                    "select doc_key from doc_key where doc_id = ?", [doc_id]
                )
            )[0][0]

        self.db.execute("release get_doc_keys")

    def get_docs(self, query):
        """Retrieve the documents matching the given query set."""
        self.db.execute("savepoint get_docs")

        doc_keys = self.convert_query_to_keys(query)

        self.corpus.docs(doc_keys=doc_keys)

        self.db.execute("release get_docs")

    def initialise_clusters(self, n_clusters, min_docs=1):

        self.db.execute("savepoint initialise")

        self.db.execute("delete from feature_cluster")
        self.db.execute(
            """
            insert into feature_cluster
                select
                    field,
                    value,
                    abs(random() % ?),
                    docs_count
                from inverted_index
                where docs_count >= ?
            """,
            (n_clusters, min_docs),
        )

        self.db.execute("release initialise")

    @property
    def cluster_ids(self):
        return [r[0] for r in self.db.execute("select cluster_id from cluster")]

    def _calculate_assignments(self, group_features, group_checks, beam=1):
        """
        Determine the assignments of features to clusters given these features groups.

        This is a building block of the various iterations of refinement operations.

        Keeps a heap of assignments, so that applications can do more sophisticated beam
        searches if necessary.

        group_features: a mapping of group_keys to lists of features in that group.
        group_checks: a mapping of group keys to the features to be check against
            that group.

        """

        futures = set()
        assignments = dict()
        total_objective = 0

        # Dispatch the comparisons that involve the most features first, as they will
        # likely take the longest.
        check_order = sorted(
            (
                (
                    self,
                    group_key,
                    features,
                    group_checks[group_key],
                )
                for group_key, features in group_features.items()
            ),
            key=lambda x: len(x[2]) + len(x[3]),
            reverse=True,
        )

        futures = [
            self.pool.submit(measure_features_to_feature_group, *c) for c in check_order
        ]

        for f in cf.as_completed(futures):

            result = f.result()

            if result is None:
                continue
            else:
                group_key, check_features, delta, already_in, objective = result

            total_objective += objective

            for feature, delta in zip(check_features, delta):

                try:
                    heapq.heappushpop(
                        assignments[feature], (delta, random.random(), group_key)
                    )
                except KeyError:
                    assignments[feature] = [(-math.inf, -1)] * beam
                    heapq.heappushpop(
                        assignments[feature], (delta, random.random(), group_key)
                    )

        return assignments, total_objective

    def refine_clusters(
        self,
        iterations=10,
        sub_iterations=2,
    ):
        """
        Attempt to improve the model clustering of the features for `iterations`.

        """

        self.db.execute("savepoint refine")

        # Establish forward reverse mappings of features to clusters and vice versa.
        feature_cluster = dict()
        cluster_feature = collections.defaultdict(set)

        for field, value, cluster_id in self.db.execute(
            """
            select
                field,
                value,
                cluster_id
            from feature_cluster
            """
        ):
            feature_cluster[(field, value)] = cluster_id
            cluster_feature[cluster_id].add((field, value))

        features = list(feature_cluster)
        cluster_ids = list(cluster_feature)
        total_objective = 0

        for iteration in range(iterations):

            print(iteration)

            # We want a completely different order of feature checks on each
            # iteration. This randomised ordering is also used to break into
            # sub batches for checking.
            random.shuffle(features)

            # More subiterations --> Less perturbation of the model for each
            # set of features, at the cost of more time for each subiteration.
            for sub_iter in range(sub_iterations):

                futures = set()
                # This approach ensures that each subiteration is of
                # approximately the same size with respect to the number of
                # features.
                moving_features = set(features[sub_iter::sub_iterations])

                # Just like the features, we want random assignments of
                # clusters to batches in each subiteration.
                random.shuffle(cluster_ids)

                # We may lose clusters if everything in that cluster is moved
                # somewhere else
                n_clusters = len(cluster_ids)

                # At this point, we test against randomised groupings of batches
                # to find a comparison group to start with. This lets us scale
                # sublinearly with the number of clusters.
                # TODO: for a small number of clusters, we can probably skip
                # the batch tests and just go directly to the next step.
                n_batches = max(
                    math.ceil((n_clusters) ** 0.5), min(16, n_clusters // 2)
                )

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
                group_checks = {
                    group_key: moving_features for group_key in group_features
                }

                batch_assignments, _ = self._calculate_assignments(
                    group_features, group_checks
                )

                # Convert the group tests into individual cluster tests
                cluster_tests = collections.defaultdict(set)

                for feature, check_keys in batch_assignments.items():
                    # Test against the current cluster
                    cluster_tests[feature_cluster[feature]].add(feature)

                    # Test against each of the clusters in the best batches
                    for _, _, group_key in check_keys:
                        for cluster_id in group_key:
                            cluster_tests[cluster_id].add(feature)

                previous_objective = total_objective

                # Compute the final assignments
                assignments, total_objective = self._calculate_assignments(
                    cluster_feature, cluster_tests
                )

                # Unpack the beam search to assign to the nearest cluster
                # Note on the last iteration, the assignments will be used
                # to track the nearest neighbours for other uses.
                for feature, best_clusters in assignments.items():

                    delta, _, cluster_id = best_clusters[0]

                    cluster_feature[feature_cluster[feature]].discard(feature)
                    cluster_feature[cluster_id].add(feature)
                    feature_cluster[feature] = cluster_id

                # Prune emptied clusters
                cluster_ids = [
                    cluster_id
                    for cluster_id, values in cluster_feature.items()
                    if len(values)
                ]

        # Serialise the actual results of the clustering!
        self.db.executemany(
            """
            update feature_cluster set cluster_id = ? where (field, value) = (?, ?)
            """,
            ((cluster_id, *feature) for feature, cluster_id in feature_cluster.items()),
        )

        self.db.execute("release refine")


def _index_docs(corpus, input_queue, temp_db_path, max_batch_entries):

    local_db = db_utilities.connect_sqlite(temp_db_path)
    # Note that we create an in memory database, but use a temporary table anyway,
    # because an in-memory database can't spill to disk.
    local_db.execute("begin")
    local_db.execute(
        "create table inverted_index_segment(field, value, doc_ids roaring_bitmap)"
    )

    # This is {field: {value: doc_ids, value2: doc_ids}}
    batch = collections.defaultdict(lambda: collections.defaultdict(BitMap))
    batch_entries = 0

    def write_batch():
        for field, values in batch.items():
            local_db.executemany(
                "insert into inverted_index_segment values(?, ?, ?)",
                ((field, value, doc_ids) for value, doc_ids in values.items()),
            )
        return 0, collections.defaultdict(lambda: collections.defaultdict(BitMap))

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
                batch_entries, batch = write_batch()

    else:
        batch_entries, batch = write_batch()

    # TODO: Merge the components locally first?
    local_db.execute(
        "create index field_values on inverted_index_segment(field, value)"
    )
    local_db.execute("commit")
    local_db.close()

    return temp_db_path


def measure_features_to_feature_group(
    index, group_key, feature_group, comparison_features
):
    """
    Measure the objective for moving the given subset of features into the
    given clusters.

    First loads the given cluster features into the representative centroid, then
    measures the improvement if each of comparison features is added in turn.

    """

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
    objective = hits / c

    assignments = []

    # All tokens that are adds (not already in the cluster)
    for feature in sorted(comparison_features - feature_group):
        docs = index[feature]

        feature_hits = len(docs)

        new_hits = hits + feature_hits
        new_c = docs.union_cardinality(cluster_union)
        new_objective = new_hits / new_c

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
            old_objective = old_hits / old_c
            delta = objective - old_objective
        # If it would otherwise be a singleton cluster, just mark it as no change
        else:
            delta = 0

        return_features.append(feature)
        result_delta[array_index] = delta
        array_index += 1

    index.close()

    return group_key, return_features, result_delta, already_in, objective


if __name__ == "__main__":
    from hyperreal import corpus

    c = corpus.PlainTextSqliteCorpus("loco.db")
    i = Index("loco_index.db", c)
    print("indexing")
    i.index(n_cpus=16)

    print("initialising")
    i.initialise_clusters(256, min_docs=5)

    print("refining")
    i.refine_clusters(iterations=10)
