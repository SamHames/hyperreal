"""
Tests for the server module - these are more integration tests, running against a
real server spawned in a subprocess.

"""

import concurrent.futures as cf
import cherrypy
import multiprocessing as mp
import pathlib
import shutil
import subprocess
import uuid

from lxml import html
import pytest
import requests

import hyperreal


servers = [
    # Index server with no associated corpus.
    (None, None, pathlib.Path("tests", "index", "alice_index.db")),
    # PlainTextCorpus
    (
        pathlib.Path("tests", "corpora", "alice.db"),
        hyperreal.corpus.PlainTextSqliteCorpus,
        pathlib.Path("tests", "index", "alice_index.db"),
    ),
]


@pytest.fixture(params=servers)
def server(tmp_path, request):
    """Server fixture that handles single file corpora."""
    source_corpus_path, source_corpus_class, source_index = request.param

    if source_corpus_path is not None:
        corpus_path = tmp_path / str(uuid.uuid4())
        shutil.copy(source_corpus_path, corpus_path)
    else:
        corpus_path = None

    index_path = tmp_path / str(uuid.uuid4())
    shutil.copy(source_index, index_path)

    context = mp.get_context("spawn")

    with cf.ProcessPoolExecutor(4, mp_context=context) as pool:
        if corpus_path is not None:
            corp = source_corpus_class(corpus_path)
        else:
            corp = None

        idx = hyperreal.index.Index(index_path, corpus=corp, pool=pool)
        idx.initialise_clusters(8, min_docs=3)
        idx.refine_clusters(iterations=5)

        index_server = hyperreal.server.SingleIndexServer(index_path, pool=pool)
        engine = hyperreal.server.launch_web_server(index_server, port=0)
        host, port = cherrypy.server.bound_addr
        yield f"http://{host}:{port}"
        engine.exit()


def test_server(server):
    """
    Start an index only server in the background using the CLI.

    Returns the local host and port of the test server.

    """

    r = requests.get(server)
    r.raise_for_status()

    # There should be an index listing in there somewhere
    doc = html.document_fromstring(r.content)
    links = [l.attrib["href"] for l in doc.findall(".//a")]

    assert "/index/0" in links

    r = requests.get(server + "/index/0")
    r.raise_for_status()

    doc = html.document_fromstring(r.content)
    links = {l.attrib["href"] for l in doc.findall(".//a")}
    must_be_present = {
        "/index/0/cluster/1",
        "/index/0/?cluster_id=1",
        "/index/0/details",
    }

    assert len(links & must_be_present) == len(must_be_present)

    for l in links:
        if l.startswith("/index/0/?feature_id="):
            must_be_present.add(l)
            break
    else:
        raise ValueError("Missing a feature_id link")

    for test_link in must_be_present:
        r = requests.get(server + test_link)
        r.raise_for_status()
