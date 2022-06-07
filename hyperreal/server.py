"""
Server components.

The server consists of the following components:

1. Starlette as the web frontend.
2. A process pool for CPU intensive work that needs to return interactively.
3. A background queue and process for long running tasks.

The whole web frontend runs as a single process: all CPU intensive tasks
need to be defered to the process pool. Long running tasks need to be
deferred to the background pool.

"""
import argparse
import os
from urllib.parse import parse_qsl

from jinja2 import PackageLoader, Environment, select_autoescape
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from starlette.staticfiles import StaticFiles
from starlette.routing import Route

import uvicorn

from hyperreal import index


templates = Environment(
    loader=PackageLoader("hyperreal"), autoescape=select_autoescape()
)


async def homepage(request):

    template = templates.get_template("index.html")

    if "feature_id" in request.query_params:
        query = request.app.state.index[int(request.query_params["feature_id"])]
        clusters = request.app.state.index.pivot_clusters_by_query(query)
    elif "cluster_id" in request.query_params:
        query = request.app.state.index.cluster_docs(
            int(request.query_params["cluster_id"])
        )
        clusters = request.app.state.index.pivot_clusters_by_query(query)
    else:
        clusters = request.app.state.index.top_cluster_features()

    return HTMLResponse(template.render(clusters=clusters))


async def edit_cluster(request):

    data = await request.form()
    method = data["method"]

    if method == "delete":
        cluster_id = int(data["cluster_id"])
        request.app.state.index.delete_clusters([cluster_id])

    return RedirectResponse(url="/", status_code=301)


async def cluster(request):

    template = templates.get_template("cluster.html")

    features = request.app.state.index.cluster_features(
        request.path_params["cluster_id"]
    )

    return HTMLResponse(template.render(features=features))


async def query(request):
    query = parse_qsl(request.url.query)

    results = request.app.state.index[query[0]]

    for feature in query[1:]:
        results |= request.app.state.index[feature]

    docs = request.app.state.index.get_docs(results[:100])

    return PlainTextResponse("\n\n".join(str(d[1]) for d in docs))


def create_app():

    routes = [
        Route("/", endpoint=homepage),
        Route("/cluster", endpoint=edit_cluster, methods=["post"]),
        Route("/cluster/{cluster_id:int}", endpoint=cluster),
        Route("/query", endpoint=query),
    ]

    app = Starlette(debug=True, routes=routes)

    if os.getenv("hyperreal_index_path"):
        app.state.index = index.Index(os.getenv("hyperreal_index_path"))

    return app


app = create_app()
