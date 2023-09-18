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
import networkx as nx

import hyperreal.corpus
import hyperreal.index
import hyperreal.server


logging.basicConfig(level=logging.INFO)


# Utility for making exporters that take three arguments, just with a
# different corpus type.
def make_csv_exporter(CorpusType):
    @click.argument("corpus_db", type=click.Path(exists=True, dir_okay=False))
    @click.argument("index_db", type=click.Path(exists=True, dir_okay=False))
    @click.argument("output_file", type=click.Path(dir_okay=False))
    @click.option(
        "--docs-per-cluster",
        type=click.INT,
        default=50,
        help="The maximum number of documents to sample from each cluster. "
        "If set to 0, all documents in clusters will be returned.",
    )
    def export_graph(corpus_db, index_db, output_file, docs_per_cluster):
        """
        Export a sample of documents from each cluster in the model.

        """
        if not hyperreal.index.Index.is_index_db(index_db):
            raise ValueError(f"{index_db} is not a valid index file.")

        corpus = CorpusType(corpus_db)
        idx = hyperreal.index.Index(index_db, corpus=corpus)

        cluster_samples, sample_clusters = idx.structured_doc_sample(
            docs_per_cluster=docs_per_cluster
        )
        idx.export_document_sample(cluster_samples, sample_clusters, output_file)

    return export_graph


def make_two_file_indexer(CorpusType):
    @click.argument("corpus_db", type=click.Path(exists=True, dir_okay=False))
    @click.argument("index_db", type=click.Path(dir_okay=False))
    @click.option(
        "--doc-batch-size",
        type=int,
        default=DEFAULT_DOC_BATCH_SIZE,
        help="The size of individual batches of documents sent for indexing. "
        "Larger sizes will require more ram, but might be more efficient for "
        "large collections.",
    )
    @click.option(
        "--position-window-size",
        default=0,
        type=int,
        help="""
        The window size to record approximate positional information. The
        default value of 0 disables recording this information. When > 0,
        approximate positions of values are recorded up to the given
        granularity. Smaller values require more space, but enable precise
        querying. Setting position size to 1 enables recording of exact term
        positions and precise querying for phrases, but currently only 2**32
        positions can be recorded in a single field.
        """,
    )
    def corpus_indexer(corpus_db, index_db, doc_batch_size, position_window_size):
        """
        Creates the index database representing the given corpus.

        If the index already exists it will be reindexed.

        """
        click.echo(f"Indexing {corpus_db} into {index_db}.")

        doc_corpus = CorpusType(corpus_db)

        mp_context = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(mp_context=mp_context) as pool:
            doc_index = hyperreal.index.Index(index_db, corpus=doc_corpus, pool=pool)

            doc_index.index(
                doc_batch_size=doc_batch_size, position_window_size=position_window_size
            )

    return corpus_indexer


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
        engine = hyperreal.server.launch_web_server(index_server)
        engine.block()


plaintext_corpus.command(name="index")(
    make_two_file_indexer(hyperreal.corpus.PlainTextSqliteCorpus)
)

plaintext_corpus.command(name="sample")(
    make_csv_exporter(hyperreal.corpus.PlainTextSqliteCorpus)
)


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
        engine = hyperreal.server.launch_web_server(index_server)
        engine.block()


stackexchange_corpus.command(name="sample")(
    make_csv_exporter(hyperreal.corpus.StackExchangeCorpus)
)

stackexchange_corpus.command(name="index")(
    make_two_file_indexer(hyperreal.corpus.StackExchangeCorpus)
)


@cli.group()
def twittersphere_corpus():
    pass


twittersphere_corpus.command(name="index")(
    make_two_file_indexer(hyperreal.corpus.TwittersphereCorpus)
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
        engine = hyperreal.server.launch_web_server(index_server)
        engine.block()


@cli.command()
@click.argument("index_db", type=click.Path(exists=True, dir_okay=False))
@click.option("--iterations", default=10, type=click.INT)
@click.option(
    "--clusters",
    default=None,
    type=click.INT,
    help="The number of clusters to use in the model. "
    "Ignored unless this is the first run, or --restart is passed. "
    "If not provided in those cases it will default to 64. ",
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
    "--tolerance",
    default=0.01,
    type=click.FloatRange(0, 1),
    help="Specify an early termination tolerance on the fraction of features moving. "
    "If fewer than this fraction of features moves in an iteration, the "
    "refinement will terminate early.",
)
@click.option(
    "--move-acceptance-probability",
    default=0.5,
    type=click.FloatRange(0, 1),
    help="The random chance of actually executing a possible move. "
    "Smaller values will take more iterations to converge, large "
    "values might be unstable.",
)
@click.option(
    "--random-seed",
    default=None,
    type=int,
    help="Specify a random seed for a model run. Best used with restart",
)
@click.option(
    "--minimum-cluster-features",
    default=1,
    type=click.INT,
    help="The minimum number of features in a cluster.",
)
@click.option(
    "--group-test-batches",
    default=None,
    type=click.INT,
    help="The number of grouped clusters to use to accelerate the clustering. "
    "Set to 0 to disable the group hierarchy cluster optimisation. Leaving "
    "at the default of None sets this to 0.1*clusters.",
)
@click.option(
    "--group-test-top-k",
    default=2,
    type=click.INT,
    help="The number of top groups to investigate if group-test-batches is enabled. "
    "Testing more groups will take longer but find better solutions.",
)
def model(
    index_db,
    iterations,
    clusters,
    min_docs,
    restart,
    include_field,
    random_seed,
    tolerance,
    move_acceptance_probability,
    minimum_cluster_features,
    group_test_batches,
    group_test_top_k,
):
    """
    Create or refine a new feature cluster model on the given index.

    Note that n_clusters can be changed arbitrarily, even when not initialising a new
    model with --restart.

    """
    doc_index = hyperreal.index.Index(index_db, random_seed=random_seed)

    # Check if any clusters exist.
    has_clusters = bool(doc_index.cluster_ids)

    if has_clusters:
        if restart:
            click.confirm(
                "A model already exists on this index, do you want to delete it?",
                abort=True,
            )

            # If the number of clusters isn't explicitly set.
            clusters = clusters or 64
            click.echo(
                f"Restarting new feature cluster model with {clusters} clusters on {index_db}."
            )
            doc_index.initialise_clusters(
                n_clusters=clusters,
                min_docs=min_docs,
                include_fields=include_field or None,
            )
    else:
        # If the number of clusters isn't explicitly set.
        clusters = clusters or 64
        click.echo(
            f"Creating new feature cluster model with {clusters} clusters on {index_db}."
        )
        doc_index.initialise_clusters(
            n_clusters=clusters,
            min_docs=min_docs,
            include_fields=include_field or None,
        )

    click.echo(f"Refining for {iterations} iterations on {index_db}.")
    doc_index.refine_clusters(
        iterations=iterations,
        target_clusters=clusters,
        tolerance=tolerance,
        move_acceptance_probability=move_acceptance_probability,
        minimum_cluster_features=minimum_cluster_features,
        group_test_batches=group_test_batches,
        group_test_top_k=group_test_top_k,
    )


@cli.group()
def export():
    pass


@export.command(name="graph")
@click.argument("index_db", type=click.Path(exists=True, dir_okay=False))
@click.argument("graph_file", type=click.Path(dir_okay=False))
@click.option(
    "--top-k-features",
    type=click.INT,
    default=5,
    help="The number of top features to include in the node labels.",
)
@click.option(
    "--include-field-in-label/--exclude-field-in-label",
    default=True,
    help="Include or exclude the field in node labels. This can be used "
    "to exclude showing fields when you only have a single field in the model.",
)
def export_graph(index_db, graph_file, top_k_features, include_field_in_label):
    """
    Export the clustering in the given index into the given file as graphml.
    """
    if not hyperreal.index.Index.is_index_db(index_db):
        raise ValueError(f"{index_db} is not a valid index file.")

    mp_context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(mp_context=mp_context) as pool:
        idx = hyperreal.index.Index(index_db, pool=pool)
        graph = idx.create_cluster_cooccurrence_graph(
            top_k=top_k_features, include_field_in_label=include_field_in_label
        )
        nx.write_graphml(graph, graph_file)


@export.command(name="clusters")
@click.argument("index_db", type=click.Path(exists=True, dir_okay=False))
@click.argument("cluster_file", type=click.Path(dir_okay=False))
@click.option(
    "--top-k-features",
    type=click.INT,
    default=10,
    help="The number of top features to include from each cluster."
    "Setting this to 0 will export all features in each cluster.",
)
def export_clusters(index_db, cluster_file, top_k_features):
    """
    Export all cluster features in the model to the given csv file.
    """
    if not hyperreal.index.Index.is_index_db(index_db):
        raise ValueError(f"{index_db} is not a valid index file.")

    idx = hyperreal.index.Index(index_db)

    with open(cluster_file, "w", encoding="utf-8") as output:
        writer = csv.writer(output, dialect="excel", quoting=csv.QUOTE_ALL)
        writer.writerow(("cluster_id", "feature_id", "field", "value", "docs_count"))

        top_k = top_k_features or 2**62
        features = idx.top_cluster_features(top_k=top_k)
        for cluster_id, _, cluster_features in features:
            for row in cluster_features:
                writer.writerow([cluster_id, *row])


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
        engine = hyperreal.server.launch_web_server(index_server)
        engine.block()
