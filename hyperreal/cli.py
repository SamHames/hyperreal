"""
This module provides all of the CLI functionality for hyperreal.

"""

import csv
import concurrent.futures as cf
import json
import multiprocessing as mp
import os

import click

import hyperreal.corpus
import hyperreal.index
import hyperreal.server


# The main entry command is always hyperreal
@click.group(name="hyperreal")
def cli():
    pass


@cli.group()
def plaintext_corpus():
    pass


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
def plaintext_corpus_index(corpus_db, index_db):
    """
    Creates the index database representing the given plaintext corpus.

    If the index already exists it will be reindexed.

    """
    click.echo(f"Indexing {corpus_db} into {index_db}.")

    doc_corpus = hyperreal.corpus.PlainTextSqliteCorpus(corpus_db)
    doc_index = hyperreal.index.Index(index_db, corpus=doc_corpus)

    doc_index.index()


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
            mp_context=mp_context,
        )
        hyperreal.server.launch_web_server(index_server)


@cli.command()
@click.argument("index_db", type=click.Path(exists=True, dir_okay=False))
@click.option("--iterations", default=10)
@click.option("--clusters", default=64)
@click.option("--min-docs", default=100)
def model(index_db, iterations, clusters, min_docs):

    click.echo(f"Creating feature cluster model on {index_db}.")
    doc_index = hyperreal.index.Index(index_db)
    doc_index.initialise_clusters(n_clusters=clusters, min_docs=min_docs)
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
        index_server = hyperreal.server.SingleIndexServer(
            index_db, pool=pool, mp_context=mp_context
        )
        hyperreal.server.launch_web_server(index_server)
