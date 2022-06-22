"""
Test cases for the index functionality, including integration with some
concrete corpus objects.

"""
import csv
import pathlib
import random
import shutil
import uuid

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
corpora_test_cases = [
    (
        hyperreal.corpus.PlainTextSqliteCorpus,
        [pathlib.Path("tests", "corpora", "alice.db")],
        {},
    )
]


@pytest.mark.parametrize("corpus,args,kwargs", corpora_test_cases)
def test_indexing(tmp_path, corpus, args, kwargs):
    """Test that all builtin corpora can be successfully indexed and queried."""
    c = corpus(*args, **kwargs)
    i = hyperreal.index.Index(tmp_path / corpus.CORPUS_TYPE, c)

    # These are actually very bad settings, but necessary for checking
    # all code paths and concurrency.
    i.index(
        batch_key_size=10,
        max_batch_entries=1000,
    )

    # Compare against the actual test data.
    with open("tests/data/alice30.txt", "r", encoding="utf-8") as f:
        docs = (line[0] for line in csv.reader(f) if line and line[0].strip())
        target_nnz = 0
        target_docs = 0
        for d in docs:
            target_docs += 1
            target_nnz += len(set(hyperreal.utilities.tokens(d)))

    nnz = list(i.db.execute("select sum(docs_count) from inverted_index"))[0][0]
    total_docs = list(i.db.execute("select count(*) from doc_key"))[0][0]
    assert total_docs == target_docs
    assert nnz == target_nnz


@pytest.mark.parametrize("n_clusters", [4, 16, 64])
def test_model_creation(example_index_path, n_clusters):
    """Test creation of a model (the core numerical component!)."""
    index = hyperreal.index.Index(example_index_path)

    index.initialise_clusters(n_clusters)
    index.refine_clusters(iterations=1)

    assert len(index.cluster_ids) == n_clusters == len(index.top_cluster_features())


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
        assert "the" in hyperreal.utilities.tokens(doc)

    # No matches, return nothing:
    assert not len(index[("nonexistent", "field")])

    # Confirm that feature_id -> feature mappings in the model are correct
    # And the cluster queries are in fact boolean combinations.
    for cluster_id in index.cluster_ids:
        cluster_query = index.cluster_query(cluster_id)

        for feature_id, field, value, docs_count in index.cluster_features(cluster_id):
            assert index[feature_id] == index[(field, value)]
            assert (index[feature_id] & cluster_query) == index[feature_id]


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
        for cluster_id, features in pivoted:
            # This feature should be first in the cluster, but the cluster
            # containing it may not always be first.
            if query == features[0][1:3]:
                break
        else:
            assert False
