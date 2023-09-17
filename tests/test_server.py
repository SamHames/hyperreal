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

import pytest
import requests

import hyperreal


@pytest.fixture()
def server(tmp_path):
    random_name = tmp_path / str(uuid.uuid4())
    shutil.copy(pathlib.Path("tests", "index", "alice_index.db"), random_name)
    context = mp.get_context("spawn")

    with cf.ProcessPoolExecutor(4, mp_context=context) as pool:
        idx = hyperreal.index.Index(random_name, pool=pool)
        idx.initialise_clusters(8, min_docs=3)
        idx.refine_clusters(iterations=5)

        index_server = hyperreal.server.SingleIndexServer(random_name, pool=pool)
        engine = hyperreal.server.launch_web_server(index_server, port=0)
        host, port = cherrypy.server.bound_addr
        yield f"http://{host}:{port}"
        engine.exit()


def test_index_server_no_corpus(server):
    """
    Start an index only server in the background using the CLI.

    Returns the local host and port of the test server.

    """

    r = requests.get(server)
    r.raise_for_status()
