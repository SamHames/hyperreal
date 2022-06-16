"""
Test cases for the index functionality, including integration with some
concrete corpus objects.

"""
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
    i.index()


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
