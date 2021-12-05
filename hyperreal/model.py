"""
The model is the core class for the actual interpretive text analytics.

The model takes a fully initialised index, and all interaction with the corpus
is directly through that.

"""
import array
import collections
import heapq
import math
import multiprocessing as mp
import random

from pyroaring import BitMap

from db_utilities import connect_sqlite


class Model:
    def __init__(self, model_db_path, index, pool=None, migrate=True):

        self.model_db_path = model_db_path
        self.db = connect_sqlite(self.model_db_path)
        self.index = index

        if not pool:
            self.mp_context = mp.get_context("spawn")
            self.pool = self.mp_context.Pool()

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

                """
            )

    def initialise(self, n_clusters, min_docs=1):

        self.db.execute("savepoint initialise")

        self.db.execute("delete from feature_cluster")
        self.db.executemany(
            "insert into feature_cluster values(?, ?, ?, ?)",
            (
                (field, value, random.randint(1, n_clusters), docs_count)
                for (field, value, docs_count) in self.index.features(min_docs=min_docs)
            ),
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
                    self.index,
                    group_key,
                    features,
                    group_checks[group_key],
                )
                for group_key, features in group_features.items()
            ),
            key=lambda x: len(x[2]) + len(x[3]),
            reverse=True,
        )

        results = [
            self.pool.apply_async(measure_features_to_feature_group, check)
            for check in check_order
        ]

        for pending_result in results:

            result = pending_result.get()

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

    def refine(
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

    return group_key, return_features, result_delta, already_in, objective


if __name__ == "__main__":

    from hyperreal import corpus, index

    c = corpus.PlainTextSqliteCorpus("test.db")
    i = index.Index("index.db", c)
    # i.index(n_cpus=6)

    model = Model("model.db", i)
    model.initialise(64, min_docs=5)
    model.refine(iterations=3)
