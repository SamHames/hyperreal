"""
Test cases for the index functionality, including integration with some
concrete corpus objects.

"""
import collections
import csv
import heapq
import pathlib
import random
import shutil
import uuid

from pyroaring import BitMap
import pytest

import hyperreal


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
        for d in docs:
            target_docs += 1
            # Note, exclude the None sentinel at the end.
            target_nnz += len(set(hyperreal.utilities.tokens(d)[:-1]))

    return target_docs, target_nnz


corpora_test_cases = [
    (
        hyperreal.corpus.PlainTextSqliteCorpus,
        [pathlib.Path("tests", "corpora", "alice.db")],
        {},
        check_alice,
    )
]


@pytest.mark.parametrize("corpus,args,kwargs,check_stats", corpora_test_cases)
def test_indexing(tmp_path, corpus, args, kwargs, check_stats):
    """Test that all builtin corpora can be successfully indexed and queried."""
    c = corpus(*args, **kwargs)
    i = hyperreal.index.Index(tmp_path / corpus.CORPUS_TYPE, c)

    # These are actually very bad settings, but necessary for checking
    # all code paths and concurrency.
    i.index(doc_batch_size=10)

    # Compare against the actual test data.
    target_docs, target_nnz = check_stats()

    nnz = list(i.db.execute("select sum(docs_count) from inverted_index"))[0][0]
    total_docs = list(i.db.execute("select count(*) from doc_key"))[0][0]
    assert total_docs == target_docs
    assert nnz == target_nnz


@pytest.mark.parametrize("n_clusters", [4, 16, 64])
def test_model_creation(example_index_path, n_clusters):
    """Test creation of a model (the core numerical component!)."""
    index = hyperreal.index.Index(example_index_path)

    index.initialise_clusters(n_clusters)
    index.refine_clusters(iterations=3)

    assert len(index.cluster_ids) == len(index.top_cluster_features())
    assert 1 < len(index.cluster_ids) <= n_clusters

    # Initialising with a field that doesn't exist should create an empty model.
    index.initialise_clusters(n_clusters, include_fields=["banana"])
    index.refine_clusters(iterations=3)

    assert len(index.cluster_ids) == len(index.top_cluster_features())
    assert 0 == len(index.cluster_ids)


def test_model_editing(example_index_path):
    """Test editing functionality on an index."""
    index = hyperreal.index.Index(example_index_path)

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


def test_querying(example_index_corpora_path):
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    index = hyperreal.index.Index(example_index_corpora_path[1], corpus=corpus)

    index.initialise_clusters(16)

    query = index[("text", "the")]
    q = len(query)
    assert q
    assert q == len(list(index.convert_query_to_keys(query)))
    assert q == len(list(index.docs(query)))
    assert q == len(index.render_docs(query))
    assert 5 == len(index.render_docs(query, random_sample_size=5))

    for doc_key, doc in index.docs(query):
        assert "the" in hyperreal.utilities.tokens(doc["text"])

    # This is a hacky test, as we're tokenising the representation of the text.
    for doc_key, rendered_doc in index.render_docs(query, random_sample_size=3):
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


def test_require_corpus(example_index_corpora_path):
    """Test the corpus requiring decorator works."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])

    index_wo_corpus = hyperreal.index.Index(example_index_corpora_path[1])
    index_wi_corpus = hyperreal.index.Index(
        example_index_corpora_path[1], corpus=corpus
    )

    query = index_wo_corpus[("text", "the")]

    with pytest.raises(hyperreal.index.CorpusMissingError):
        index_wo_corpus.docs(query)

    assert len(query) == len(list(index_wi_corpus.docs(query)))


def test_pivoting(example_index_path):
    """Test pivoting by features and by clusters."""
    index = hyperreal.index.Index(example_index_path)

    index.initialise_clusters(16)

    # Test early/late truncation in each direction with large and small
    # features.
    for query in [("text", "the"), ("text", "denied")]:
        pivoted = index.pivot_clusters_by_query(index[query], top_k=2)
        for cluster_id, weight, features in pivoted:
            # This feature should be first in the cluster, but the cluster
            # containing it may not always be first.
            if query == features[0][1:3]:
                break
        else:
            assert False


def test_create_new_features(example_index_corpora_path):
    """Test creating new features - possibly from old features."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    index = hyperreal.index.Index(example_index_corpora_path[1], corpus=corpus)

    new_feature = index[("text", "the")] & index[("text", "and")]
    new_feature_key = ("custom", "arbitrary new feature")
    index[new_feature_key] = new_feature

    assert index[new_feature_key] == new_feature

    with pytest.raises(KeyError):
        index[("text", "the")] = index[("text", "the")]

    # Reindexing will remove the custom feature
    index.index()

    assert not len(index[new_feature_key])


@pytest.mark.parametrize("n_clusters", [4, 8, 16])
def test_fixed_seed(example_index_path, n_clusters):
    """Test creation of a model (the core numerical component!)."""
    index = hyperreal.index.Index(example_index_path, random_seed=10)

    index.initialise_clusters(n_clusters)
    index.refine_clusters(iterations=1)

    clustering_1 = index.top_cluster_features()
    split_1 = index.propose_cluster_split(1, iterations=1)

    # Note we need to initialise a new object with the random seed, otherwise
    # as each random operation consumes items from the stream.
    index = hyperreal.index.Index(example_index_path, random_seed=10)

    index.initialise_clusters(n_clusters)
    index.refine_clusters(iterations=1)

    clustering_2 = index.top_cluster_features()
    split_2 = index.propose_cluster_split(1, iterations=1)

    assert clustering_1 == clustering_2
    assert split_1 == split_2
