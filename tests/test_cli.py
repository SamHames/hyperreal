"""
Test at the CLI layer - these function closer to end to end tests and
should test many of the most common entrypoints.

Note that the Twitter tests require that some Twitter data has been collected.
See the tox environment "collect_twitter_test_data" for this.

"""
import pathlib
import sqlite3

from click.testing import CliRunner

from hyperreal import cli

data_path = pathlib.Path("tests", "data")
corpora_path = pathlib.Path("tests", "corpora")


def test_plaintext_corpus(tmp_path):

    target_corpora_db = tmp_path / "test.db"
    target_index_db = tmp_path / "test_index.db"

    runner = CliRunner()

    # Create
    result = runner.invoke(
        cli.plaintext_corpus,
        [
            "create",
            str(data_path / "alice30.txt"),
            str(target_corpora_db),
        ],
    )
    assert result.exit_code == 0

    # Index
    result = runner.invoke(
        cli.plaintext_corpus,
        [
            "index",
            str(target_corpora_db),
            str(target_index_db),
        ],
    )
    assert result.exit_code == 0

    # Model
    result = runner.invoke(
        cli.model, ["--iterations", "10", "--clusters", "10", str(target_index_db)]
    )


def test_twittersphere_corpus(tmp_path):

    target_corpora_db = corpora_path / "twitter.db"
    target_index_db = tmp_path / "twitter_index.db"

    runner = CliRunner()

    # Index
    result = runner.invoke(
        cli.twittersphere_corpus,
        [
            "index",
            str(target_corpora_db),
            str(target_index_db),
        ],
    )

    if result.exit_code != 0:
        print(result.output)

    assert result.exit_code == 0

    # Model
    result = runner.invoke(
        cli.model, ["--iterations", "10", "--clusters", "10", str(target_index_db)]
    )


def test_sx_corpus(tmp_path):

    target_corpora_db = tmp_path / "sx_corpus.db"
    target_index_db = tmp_path / "sx_corpus_index.db"

    runner = CliRunner()

    # Add site
    result = runner.invoke(
        cli.stackexchange_corpus,
        [
            "add-site",
            str(data_path / "expat_sx" / "Posts.xml"),
            str(data_path / "expat_sx" / "Comments.xml"),
            str(data_path / "expat_sx" / "Users.xml"),
            "https://expatriates.stackexchange.com",
            str(target_corpora_db),
        ],
    )
    assert result.exit_code == 0

    # Index
    result = runner.invoke(
        cli.stackexchange_corpus,
        ["index", str(target_corpora_db), str(target_index_db)],
    )
    assert result.exit_code == 0

    # Model
    result = runner.invoke(
        cli.model, ["--iterations", "1", "--clusters", "16", str(target_index_db)]
    )
