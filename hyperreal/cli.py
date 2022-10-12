"""
This module provides all of the CLI functionality for hyperreal.

"""

import csv
import concurrent.futures as cf
import json
import logging
import multiprocessing as mp
import os

import click

import hyperreal.corpus
import hyperreal.index
import hyperreal.server


logging.basicConfig(level=logging.INFO)


# The main entry command is always hyperreal
@click.group(name="hyperreal")
def cli():
    pass


@cli.group()
def plaintext_corpus():
    pass


DEFAULT_DOC_BATCH_SIZE = 5000


@plaintext_corpus.command(name="create")
@click.argument("text_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("corpus_db", type=click.Path(dir_okay=False))
def plaintext_corpus_create(text_file, corpus_db):
    """
    Create a simple corpus database from a plain text input file.

    The input file should be a plain text file, with one document per line. If
    a document begins and ends with a quote (") character, it will be treated
    as a JSON string and escapes decoded.

    If the corpus exists already, the content will be replaced with the
    contents of the text file.

    """
    click.echo(f"Replacing existing contents of {corpus_db} with {text_file}.")

    doc_corpus = hyperreal.corpus.PlainTextSqliteCorpus(corpus_db)

    with open(text_file, "r", encoding="utf-8") as infile:
        f = csv.reader(infile)
        # The only documents we drop are lines that are only whitespace.
        docs = (line[0] for line in f if line and line[0].strip())
        doc_corpus.replace_docs(docs)


@plaintext_corpus.command(name="index")
@click.argument("corpus_db", type=click.Path(exists=True, dir_okay=False))
@click.argument("index_db", type=click.Path(dir_okay=False))
@click.option(
    "--doc_batch_size",
    type=int,
    default=DEFAULT_DOC_BATCH_SIZE,
    help="The size of individual batches of documents sent for indexing. "
    "Larger sizes will require more ram, but might be more efficient for "
    "large collections.",
)
@click.option(
    "--skipgram-window-size",
    default=0,
    help="Size of window to extract skipgrams from. "
    "Default is 0, which disables this functionality",
)
@click.option(
    "--skipgram-min-docs",
    default=3,
    help="Minimum number of docs containing a skipgram for the count to be retained. "
    "This threshold limits the size of the table used to store skipgrams - set to 1 "
    "to keep all counts (not recommended).",
)
def plaintext_corpus_index(
    corpus_db, index_db, doc_batch_size, skipgram_window_size, skipgram_min_docs
):
    """
    Creates the index database representing the given plaintext corpus.

    If the index already exists it will be reindexed.

    """
    click.echo(f"Indexing {corpus_db} into {index_db}.")

    doc_corpus = hyperreal.corpus.PlainTextSqliteCorpus(corpus_db)
    doc_index = hyperreal.index.Index(index_db, corpus=doc_corpus)

    doc_index.index(
        doc_batch_size=doc_batch_size,
        skipgram_window_size=skipgram_window_size,
        skipgram_min_docs=skipgram_min_docs,
    )


@plaintext_corpus.command(name="serve")
@click.argument("corpus_db", type=click.Path(exists=True, dir_okay=False))
@click.argument("index_db", type=click.Path(exists=True, dir_okay=False))
def plaintext_corpus_serve(corpus_db, index_db):
    """
    Serve the given plaintext corpus and index via the webserver.

    """

    if not hyperreal.index.Index.is_index_db(index_db):
        raise ValueError(f"{index_db} is not a valid index file.")

    click.echo(f"Serving corpus '{corpus_db}'/ index '{index_db}'.")

    mp_context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(mp_context=mp_context) as pool:
        index_server = hyperreal.server.SingleIndexServer(
            index_db,
            corpus_class=hyperreal.corpus.PlainTextSqliteCorpus,
            corpus_args=[corpus_db],
            pool=pool,
        )
        hyperreal.server.launch_web_server(index_server)


@cli.group()
def stackexchange_corpus():
    pass


@stackexchange_corpus.command(name="add-site")
@click.argument("posts_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("comments_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("users_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("site_url", type=str)
@click.argument("corpus_db", type=click.Path(dir_okay=False))
def stackexchange_corpus_add_site(
    posts_file, comments_file, users_file, site_url, corpus_db
):
    """
    Create a simple corpus database from the stackexchange XML data dumps.

    The data dumps for all sites can be found here:
    https://archive.org/download/stackexchange

    Dumps from multiple sites can be added to the same corpus. The site_url
    value is used to differentiate the source site, and to construct URLs to
    link directly to live data - this should be the base URL of the site
    associated with the dump, such as `https://stackoverflow.com` or
    `https://travel.stackexchange.com`.

    """
    click.echo(f"Adding {site_url} data into {corpus_db}.")

    doc_corpus = hyperreal.corpus.StackExchangeCorpus(corpus_db)

    doc_corpus.add_site_data(posts_file, comments_file, users_file, site_url)


@stackexchange_corpus.command(name="index")
@click.argument("corpus_db", type=click.Path(exists=True, dir_okay=False))
@click.argument("index_db", type=click.Path(dir_okay=False))
@click.option(
    "--doc_batch_size",
    type=int,
    default=DEFAULT_DOC_BATCH_SIZE,
    help="The size of individual batches of documents sent for indexing. "
    "Larger sizes will require more ram, but might be more efficient for "
    "large collections.",
)
@click.option(
    "--skipgram-window-size",
    default=0,
    help="Size of window to extract skipgrams from. "
    "Default is 0, which disables this functionality",
)
@click.option(
    "--skipgram-min-docs",
    default=3,
    help="Minimum number of docs containing a skipgram for the count to be retained. "
    "This threshold limits the size of the table used to store skipgrams - set to 1 "
    "to keep all counts (not recommended).",
)
def stackexchange_corpus_index(
    corpus_db, index_db, doc_batch_size, skipgram_window_size, skipgram_min_docs
):
    """
    Creates the index database representing the given Stack Exchange corpus.

    If the index already exists it will be reindexed.

    """
    click.echo(f"Indexing {corpus_db} into {index_db}.")

    doc_corpus = hyperreal.corpus.StackExchangeCorpus(corpus_db)
    doc_index = hyperreal.index.Index(index_db, corpus=doc_corpus)

    doc_index.index(
        doc_batch_size=doc_batch_size,
        skipgram_window_size=skipgram_window_size,
        skipgram_min_docs=skipgram_min_docs,
    )


@stackexchange_corpus.command(name="serve")
@click.argument("corpus_db", type=click.Path(exists=True, dir_okay=False))
@click.argument("index_db", type=click.Path(exists=True, dir_okay=False))
def stackexchange_corpus_serve(corpus_db, index_db):
    """
    Serve the given StackExchange corpus and index via the webserver.

    """

    if not hyperreal.index.Index.is_index_db(index_db):
        raise ValueError(f"{index_db} is not a valid index file.")

    click.echo(f"Serving corpus '{corpus_db}'/ index '{index_db}'.")

    mp_context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(mp_context=mp_context) as pool:
        index_server = hyperreal.server.SingleIndexServer(
            index_db,
            corpus_class=hyperreal.corpus.StackExchangeCorpus,
            corpus_args=[corpus_db],
            pool=pool,
        )
        hyperreal.server.launch_web_server(index_server)


@cli.group()
def twittersphere_corpus():
    pass


@twittersphere_corpus.command(name="index")
@click.argument("corpus_db", type=click.Path(exists=True, dir_okay=False))
@click.argument("index_db", type=click.Path(dir_okay=False))
@click.option(
    "--doc_batch_size",
    type=int,
    default=100000,
    help="The size of individual batches of documents sent for indexing. "
    "Larger sizes will require more ram, but might be more efficient for "
    "large collections.",
)
@click.option(
    "--skipgram-window-size",
    default=0,
    help="Size of window to extract skipgrams from. "
    "Default is 0, which disables this functionality",
)
@click.option(
    "--skipgram-min-docs",
    default=3,
    help="Minimum number of docs containing a skipgram for the count to be retained. "
    "This threshold limits the size of the table used to store skipgrams - set to 1 "
    "to keep all counts (not recommended).",
)
def twittersphere_corpus_index(
    corpus_db, index_db, doc_batch_size, skipgram_window_size, skipgram_min_docs
):
    """
    Creates the index database representing the given Stack Exchange corpus.

    If the index already exists it will be reindexed.

    """
    click.echo(f"Indexing {corpus_db} into {index_db}.")

    doc_corpus = hyperreal.corpus.TwittersphereCorpus(corpus_db)
    doc_index = hyperreal.index.Index(index_db, corpus=doc_corpus)

    doc_index.index(
        doc_batch_size=doc_batch_size,
        skipgram_window_size=skipgram_window_size,
        skipgram_min_docs=skipgram_min_docs,
    )


@twittersphere_corpus.command(name="serve")
@click.argument("corpus_db", type=click.Path(exists=True, dir_okay=False))
@click.argument("index_db", type=click.Path(exists=True, dir_okay=False))
def twittersphere_corpus_serve(corpus_db, index_db):
    """
    Serve the given StackExchange corpus and index via the webserver.

    """

    if not hyperreal.index.Index.is_index_db(index_db):
        raise ValueError(f"{index_db} is not a valid index file.")

    click.echo(f"Serving corpus '{corpus_db}'/ index '{index_db}'.")

    mp_context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(mp_context=mp_context) as pool:
        index_server = hyperreal.server.SingleIndexServer(
            index_db,
            corpus_class=hyperreal.corpus.TwittersphereCorpus,
            corpus_args=[corpus_db],
            pool=pool,
        )
        hyperreal.server.launch_web_server(index_server)


@cli.command()
@click.argument("index_db", type=click.Path(exists=True, dir_okay=False))
@click.option("--iterations", default=10)
@click.option(
    "--clusters",
    default=64,
    help="The number of clusters to use in the model. "
    "Ignored unless this is the first run, or --restart is passed.",
)
@click.option("--min-docs", default=100)
@click.option(
    "--include-field",
    default=[],
    help="A field to include in the model initialisation. "
    "Multiple fields can be specified - if no fields are provided all available fields will be included."
    "Ignored unless this is the first run, or --restart is passed.",
    multiple=True,
)
@click.option(
    "--restart", default=False, is_flag=True, help="Restart the model from scratch."
)
@click.option(
    "--random-seed",
    default=None,
    type=int,
    help="Specify a random seed for a model run. Best used with restart",
)
def model(
    index_db, iterations, clusters, min_docs, restart, include_field, random_seed
):

    doc_index = hyperreal.index.Index(index_db, random_seed=random_seed)

    # Check if any clusters exist.
    has_clusters = bool(doc_index.cluster_ids)

    if has_clusters:
        if restart:
            click.echo(
                f"Restarting new feature cluster model with {clusters} clusters on {index_db}."
            )
            doc_index.initialise_clusters(
                n_clusters=clusters,
                min_docs=min_docs,
                include_fields=include_field or None,
            )
    else:
        click.echo(
            f"Creating new feature cluster model with {clusters} clusters on {index_db}."
        )
        doc_index.initialise_clusters(
            n_clusters=clusters,
            min_docs=min_docs,
            include_fields=include_field or None,
        )

    click.echo(f"Refining for {iterations} iterations on {index_db}.")
    doc_index.refine_clusters(iterations=iterations)


@cli.command()
@click.argument("index_db", type=click.Path(exists=True, dir_okay=False))
def serve(index_db):
    """
    Serve the given index file.

    Note that this method of serving will not provide access to the underlying
    corpus of documents - this is useful for non-consumptive types of analysis.
    If you want to view the underlying documents, see the serve subcommand for
    the specific type of corpus you're interested in.

    """

    if not hyperreal.index.Index.is_index_db(index_db):
        raise ValueError(f"{index_db} is not a valid index file.")

    mp_context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(mp_context=mp_context) as pool:
        index_server = hyperreal.server.SingleIndexServer(index_db, pool=pool)
        hyperreal.server.launch_web_server(index_server)
