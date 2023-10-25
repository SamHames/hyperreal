"""
Test cases for the index functionality, including integration with some
concrete corpus objects.

"""
import collections
import concurrent.futures as cf
import csv
import heapq
import logging
import multiprocessing as mp
import pathlib
import random
import shutil
import uuid

from pyroaring import BitMap
import pytest

import hyperreal


@pytest.fixture(scope="module")
def pool():
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(4, mp_context=context) as pool:
        yield pool


@pytest.fixture
def example_index_path(tmp_path):
    random_name = tmp_path / str(uuid.uuid4())
    shutil.copy(pathlib.Path("tests", "index", "alice_index.db"), random_name)
    return random_name


@pytest.fixture
def example_index_corpora_path(tmp_path):
    random_corpus = tmp_path / str(uuid.uuid4())
    shutil.copy(pathlib.Path("tests", "corpora", "alice.db"), random_corpus)
    random_index = tmp_path / str(uuid.uuid4())
    shutil.copy(pathlib.Path("tests", "index", "alice_index.db"), random_index)

    return random_corpus, random_index


# This is a list of tuples, corresponding to the corpus class to test, and the
# concrete arguments to that class to instantiate against the test data.
def check_alice():
    with open("tests/data/alice30.txt", "r", encoding="utf-8") as f:
        docs = (line[0] for line in csv.reader(f) if line and line[0].strip())
        target_nnz = 0
        target_docs = 0
        target_positions = 0
        for d in docs:
            target_docs += 1
            target_nnz += len(set(hyperreal.utilities.tokens(d)))
            target_positions += sum(
                1 for v in hyperreal.utilities.tokens(d) if v is not None
            )

    return target_docs, target_nnz, target_positions


corpora_test_cases = [
    (
        hyperreal.corpus.PlainTextSqliteCorpus,
        [pathlib.Path("tests", "corpora", "alice.db")],
        {},
        check_alice,
    )
]


@pytest.mark.parametrize("corpus,args,kwargs,check_stats", corpora_test_cases)
def test_indexing(pool, tmp_path, corpus, args, kwargs, check_stats):
    """Test that all builtin corpora can be successfully indexed and queried."""
    c = corpus(*args, **kwargs)
    idx = hyperreal.index.Index(tmp_path / corpus.CORPUS_TYPE, c, pool=pool)

    # These are actually very bad settings, but necessary for checking
    # all code paths and concurrency.
    idx.index(
        doc_batch_size=10,
    )

    # Compare against the actual test data.
    target_docs, target_nnz, target_positions = check_stats()

    nnz = list(idx.db.execute("select sum(docs_count) from inverted_index"))[0][0]
    total_docs = list(idx.db.execute("select count(*) from doc_key"))[0][0]
    assert total_docs == target_docs
    assert nnz == target_nnz

    # Feature ids should remain the same across indexing runs
    features_field_values = {
        feature_id: (field, value)
        for feature_id, field, value in idx.db.execute(
            "select feature_id, field, value from inverted_index"
        )
    }

    idx.index(doc_batch_size=10, index_positions=True)

    for feature_id, field, value in idx.db.execute(
        "select feature_id, field, value from inverted_index"
    ):
        assert (field, value) == features_field_values[feature_id]

    idx.index(doc_batch_size=1, index_positions=True)

    positions = list(idx.db.execute("select sum(position_count) from position_index"))[
        0
    ][0]

    assert positions == target_positions

    # Make sure that there's information for every document with
    # positional information.
    assert (
        target_docs
        == list(idx.db.execute("select sum(docs_count) from position_doc_map"))[0][0]
    )

    # Test window extraction from documents.
    for (
        doc_key,
        doc_id,
        cooccurrence_windows,
    ) in idx.extract_matching_feature_windows(
        idx[("text", "hatter")], [("text", "hatter"), ("text", "mad")], 10, 5
    ):
        for match, windows in cooccurrence_windows["text"].items():
            assert match in ["mad", "hatter"]
            assert all("hatter" in window or "mad" in window for window in windows)
            assert all(len(window) <= 21 for window in windows)

    for doc_key, doc_id, concordances in idx.concordances(
        idx[("text", "hatter")], [("text", "hatter"), ("text", "mad")], 10, 5
    ):
        for concordance in concordances["text"]:
            assert "mad" in concordance or "hatter" in concordance

    # Test passage retrieval - because this is one line = one document,
    # the passage retrieval is the same as the document retrieval
    passage_query = [[("text", word)] for word in "hare hatter".split()]
    score_passages = idx.score_passages_dnf(passage_query, 50)

    assert len(score_passages) > 0
    assert len(score_passages) == len(idx[("text", "hare")] & idx[("text", "hatter")])

    # Actually render the passages as well
    rendered = list(idx.render_passages_table(score_passages))
    for doc_id, doc_key, doc in rendered:
        assert all("hare" in p for p in doc["text"])
        assert all("hatter" in p for p in doc["text"])

    assert (
        len(list(idx.render_passages_table(score_passages, random_sample_size=3))) == 3
    )


@pytest.mark.parametrize("n_clusters", [4, 16, 64])
def test_model_creation(pool, example_index_path, n_clusters):
    """Test creation of a model (the core numerical component!)."""
    idx = hyperreal.index.Index(example_index_path, pool=pool)

    idx.initialise_clusters(n_clusters)
    # The defaults will generate dense clustering for 4 clusters, hierarchical for 16, 64
    idx.refine_clusters(iterations=3)

    assert len(idx.cluster_ids) == len(idx.top_cluster_features())
    assert 1 < len(idx.cluster_ids) <= n_clusters

    idx.refine_clusters(iterations=3, group_test_batches=0)
    assert len(idx.cluster_ids) == len(idx.top_cluster_features())
    assert 1 < len(idx.cluster_ids) <= n_clusters

    # Initialising with a field that doesn't exist should create an empty model.
    idx.initialise_clusters(n_clusters, include_fields=["banana"])
    idx.refine_clusters(iterations=3)

    assert len(idx.cluster_ids) == len(idx.top_cluster_features())
    assert 0 == len(idx.cluster_ids)

    # No op cases - empty and single clusters selected.
    assert idx._refine_feature_groups({}) == (dict(), set())
    idx.refine_clusters(iterations=10, cluster_ids=[1])


def test_model_editing(example_index_path, pool):
    """Test editing functionality on an index."""
    index = hyperreal.index.Index(example_index_path, pool=pool)

    index.initialise_clusters(16)

    cluster_ids = index.cluster_ids

    assert len(cluster_ids) == 16

    all_cluster_features = {
        cluster_id: index.cluster_features(cluster_id) for cluster_id in cluster_ids
    }

    all_feature_ids = [
        feature[0] for features in all_cluster_features.values() for feature in features
    ]

    assert len(all_feature_ids) == len(set(all_feature_ids))

    # Delete single feature at random
    delete_feature_id = random.choice(all_cluster_features[0])[0]
    index.delete_features([delete_feature_id])
    assert delete_feature_id not in [
        feature[0] for feature in index.cluster_features(0)
    ]
    # Deleting the same feature shouldn't fail
    index.delete_features([delete_feature_id])

    # Delete all features in a cluster
    index.delete_features([feature[0] for feature in all_cluster_features[0]])
    assert len(index.cluster_features(0)) == 0

    assert 0 not in index.cluster_ids and len(index.cluster_ids) == 15

    index.delete_clusters([1, 2])
    assert not ({1, 2} & set(index.cluster_ids)) and len(index.cluster_ids) == 13

    # Merge clusters
    index.merge_clusters([3, 4])
    assert 4 not in index.cluster_ids and len(index.cluster_ids) == 12

    assert len(index.cluster_features(3)) == len(all_cluster_features[3]) + len(
        all_cluster_features[4]
    )

    # Create a new cluster from a set of features
    new_cluster_id = index.create_cluster_from_features(
        [feature[0] for feature in all_cluster_features[4]]
    )
    assert new_cluster_id == 16
    assert len(index.cluster_ids) == 13


def test_model_structured_sampling(example_index_corpora_path, pool, tmp_path):
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(example_index_corpora_path[1], pool=pool, corpus=corpus)

    idx.initialise_clusters(8, min_docs=5)
    idx.refine_clusters(iterations=5)

    cluster_sample, sample_clusters = idx.structured_doc_sample(docs_per_cluster=2)

    # Should only a specific number of documents sampled - note that this isn't
    # guaranteed when docs_per_cluster is larger than clusters in the dataset.
    assert (
        len(BitMap.union(*cluster_sample.values()))
        == len(BitMap.union(*sample_clusters.values()))
        == 16
    )

    assert sum(len(docs) for docs in sample_clusters.values()) >= 16

    # Test writing
    write_path = tmp_path / "test.csv"

    idx.export_document_sample(cluster_sample, sample_clusters, write_path)

    # Selective cluster exporting
    cluster_sample, sample_clusters = idx.structured_doc_sample(
        docs_per_cluster=2, cluster_ids=idx.cluster_ids[:2]
    )

    assert len(cluster_sample) == 2


def test_querying(example_index_corpora_path, pool):
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    index = hyperreal.index.Index(
        example_index_corpora_path[1], corpus=corpus, pool=pool
    )

    index.initialise_clusters(16)

    query = index[("text", "the")]
    q = len(query)
    assert q
    assert q == len(list(index.convert_query_to_keys(query)))
    assert q == len(list(index.docs(query)))
    assert q == len(index.render_docs_html(query))
    assert 5 == len(index.render_docs_html(query, random_sample_size=5))
    assert q == len(index.render_docs_table(query))
    assert 5 == len(index.render_docs_table(query, random_sample_size=5))

    for doc_id, doc_key, doc in index.docs(query):
        assert "the" in hyperreal.utilities.tokens(doc["text"])

    # This is a hacky test, as we're tokenising the representation of the text.
    for doc_key, rendered_doc in index.render_docs_html(query, random_sample_size=3):
        assert "the" in hyperreal.utilities.tokens(rendered_doc)

    # No matches, return nothing:
    assert not len(index[("nonexistent", "field")])

    # Confirm that feature_id -> feature mappings in the model are correct
    # And the cluster queries are in fact boolean combinations.
    for cluster_id in index.cluster_ids:
        cluster_matching, cluster_bs = index.cluster_query(cluster_id)

        # Used for checking the ranking with bstm
        accumulator = collections.Counter()

        for feature_id, field, value, docs_count in index.cluster_features(cluster_id):
            assert index[feature_id] == index[(field, value)]
            assert (index[feature_id] & cluster_matching) == index[feature_id]
            for doc_id in index[feature_id]:
                accumulator[doc_id] += 1

        # also test the ranking with bstm - we should retrieve the same number
        # of results by each method.
        top_k = hyperreal.utilities.bstm(cluster_matching, cluster_bs, 5)
        n_check = len(top_k)
        real_top_k = BitMap(
            heapq.nlargest(n_check, accumulator, key=lambda x: accumulator[x])
        )

        assert top_k == real_top_k

    # Confirm that feature lookup works in both directions
    feature = index.lookup_feature(1)
    assert index[1] == index[feature]

    feature_id = index.lookup_feature_id(("text", "the"))
    assert index[feature_id] == index[("text", "the")]


def test_require_corpus(example_index_corpora_path, pool):
    """Test the corpus requiring decorator works."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])

    index_wo_corpus = hyperreal.index.Index(example_index_corpora_path[1], pool=pool)
    index_wi_corpus = hyperreal.index.Index(
        example_index_corpora_path[1], corpus=corpus, pool=pool
    )

    query = index_wo_corpus[("text", "the")]

    with pytest.raises(hyperreal.index.CorpusMissingError):
        index_wo_corpus.docs(query)

    assert len(query) == len(list(index_wi_corpus.docs(query)))


def test_pivoting(example_index_path, pool):
    """Test pivoting by features and by clusters."""
    index = hyperreal.index.Index(example_index_path, pool=pool)

    index.initialise_clusters(16)

    # Test early/late truncation in each direction with large and small
    # features.
    for scoring in ("jaccard",):
        for query in [("text", "the"), ("text", "denied")]:
            pivoted = index.pivot_clusters_by_query(
                index[query], top_k=2, scoring=scoring
            )
            for cluster_id, weight, features in pivoted:
                # This feature should be first in the cluster, but the cluster
                # containing it may not always be first.
                if query == features[0][1:3]:
                    break
            else:
                assert False


@pytest.mark.parametrize("n_clusters", [4, 8, 16])
def test_fixed_seed(example_index_path, pool, n_clusters):
    """
    Test creation of a model (the core numerical component!).

    Note that byte for byte repeatability is only guaranteed with a single
    worker processing pool, as currently there is a dependency on the order
    of operations - this might be fixed in the future.

    """

    index = hyperreal.index.Index(example_index_path, random_seed=10, pool=pool)

    index.initialise_clusters(n_clusters)
    index.refine_clusters(iterations=1)

    clustering_1 = index.top_cluster_features()
    index.refine_clusters(target_clusters=n_clusters + 5, iterations=2)
    refined_clustering_1 = index.top_cluster_features()

    # Note we need to initialise a new object with the random seed, otherwise
    # as each random operation consumes items from the stream.
    index = hyperreal.index.Index(example_index_path, random_seed=10, pool=pool)

    index.initialise_clusters(n_clusters)
    index.refine_clusters(iterations=1)

    clustering_2 = index.top_cluster_features()
    index.refine_clusters(target_clusters=n_clusters + 5, iterations=2)
    refined_clustering_2 = index.top_cluster_features()

    assert clustering_1 == clustering_2
    assert refined_clustering_1 == refined_clustering_2


def test_splitting(example_index_path, pool):
    """Test splitting and saving splits works correctly"""
    index = hyperreal.index.Index(example_index_path, pool=pool)

    n_clusters = 16
    index.initialise_clusters(n_clusters)

    assert len(index.cluster_ids) == n_clusters

    for cluster_id in index.cluster_ids:
        index.refine_clusters(cluster_ids=[cluster_id], target_clusters=2, iterations=1)

    assert len(index.cluster_ids) == n_clusters * 2


def test_dissolving(example_index_path, pool):
    """Test dissolving clusters, reducing the available cluster count."""
    index = hyperreal.index.Index(example_index_path, pool=pool)

    n_clusters = 16
    index.initialise_clusters(n_clusters)
    index.refine_clusters(iterations=10)

    assert len(index.cluster_ids) == n_clusters

    index.refine_clusters(iterations=4, target_clusters=12)
    assert len(index.cluster_ids) == 12


def test_filling_empty_clusters(example_index_path, pool):
    """Test expanding the number of clusters by subdividing the largest."""
    index = hyperreal.index.Index(example_index_path, pool=pool)

    n_clusters = 8
    index.initialise_clusters(n_clusters)
    index.refine_clusters(iterations=3)
    assert len(index.cluster_ids) == n_clusters

    index.refine_clusters(iterations=3, target_clusters=12)

    assert len(index.cluster_ids) == 12


def test_termination(example_index_path, caplog, pool):
    """Test"""
    index = hyperreal.index.Index(example_index_path, pool=pool)

    n_clusters = 8
    index.initialise_clusters(n_clusters)

    with caplog.at_level(logging.INFO):
        index.refine_clusters(iterations=100)
        assert len(index.cluster_ids) == n_clusters

        for record in caplog.records:
            if "Terminating" in record.message:
                break
        else:
            assert False


def test_pinning(example_index_path, pool):
    """Test expanding the number of clusters by subdividing the largest."""
    index = hyperreal.index.Index(example_index_path, pool=pool)

    n_clusters = 8
    index.initialise_clusters(n_clusters, min_docs=5)
    index.refine_clusters(iterations=1)
    assert len(index.cluster_ids) == n_clusters

    # Get two features from the first cluster to pin.
    pinned_features = [f[0] for f in index.cluster_features(1, top_k=2)]
    index.pin_features(feature_ids=pinned_features)

    # Refine and split at the same time to confirm that splitting also doesn't move pinned features
    index.refine_clusters(iterations=3, target_clusters=12)

    whole_cluster = {f[0] for f in index.cluster_features(1)}
    for feature_id in pinned_features:
        assert feature_id in whole_cluster

    assert len(index.cluster_ids) == 12

    # Now pin a whole cluster
    index.pin_clusters(cluster_ids=[1])
    index.refine_clusters(iterations=3, target_clusters=16)

    assert whole_cluster == {f[0] for f in index.cluster_features(1)}
    assert len(index.cluster_ids) == 16


def test_graph_creation(example_index_path, pool):
    """Test that graphs can be created properly and worked with."""
    idx = hyperreal.index.Index(example_index_path, pool=pool)

    n_clusters = 32
    idx.initialise_clusters(n_clusters, min_docs=5)
    idx.refine_clusters(iterations=3)

    graph = idx.create_cluster_cooccurrence_graph(top_k=5)

    assert len(graph.nodes) == 32


def test_indexing_utility(example_index_corpora_path, tmp_path, pool):
    """Test the indexing utility function."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])

    temp_index = tmp_path / "tempindex.db"

    doc_keys = BitMap(range(1, 100))
    doc_ids = doc_keys

    hyperreal.index._index_docs(
        corpus, doc_keys, doc_ids, str(temp_index), 1, mp.Lock()
    )


def test_field_intersection(tmp_path, pool):
    data_path = pathlib.Path("tests", "data")
    target_corpora_db = tmp_path / "sx_corpus.db"
    target_index_db = tmp_path / "sx_corpus_index.db"

    sx_corpus = hyperreal.corpus.StackExchangeCorpus(str(target_corpora_db))

    sx_corpus.add_site_data(
        str(data_path / "expat_sx" / "Posts.xml"),
        str(data_path / "expat_sx" / "Comments.xml"),
        str(data_path / "expat_sx" / "Users.xml"),
        "https://expatriates.stackexchange.com",
    )

    sx_idx = hyperreal.index.Index(str(target_index_db), pool=pool, corpus=sx_corpus)
    sx_idx.index()

    queries = {
        "visa": sx_idx[("Post", "visa")],
        "June 2020": sx_idx[("created_month", "2020-06-01T00:00:00")],
    }

    values, totals, intersections = sx_idx.intersect_queries_with_field(
        queries, "created_year"
    )

    non_zero_month_intersection = 0

    assert all(c > 0 for c in intersections["visa"])

    # the 'created_month' query should only have nonzero intersection with a
    # single year.
    assert sum(1 for c in intersections["June 2020"] if c > 0) == 1
