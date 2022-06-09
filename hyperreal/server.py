"""
WARNING: EXPERIMENTAL!

This is all prototype only right now, you shouldn't rely on any part of this
to be stable or consistent.

The server consists of the following components:

1. Starlette as the web frontend, and a single loaded index on the app
state. This is the index specified on the command line to run the
webserver on top.

The whole web frontend runs as a single process: all CPU intensive tasks
need to be defered to the process pool.

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


async def delete_clusters(request):

    data = await request.form()

    cluster_ids = [int(cluster_id) for cluster_id in data.getlist("cluster_id")]
    request.app.state.index.delete_clusters(cluster_ids)

    return RedirectResponse(url="/", status_code=303)


async def merge_clusters(request):

    data = await request.form()

    cluster_ids = [int(cluster_id) for cluster_id in data.getlist("cluster_id")]
    new_cluster_id = request.app.state.index.merge_clusters(cluster_ids)

    return RedirectResponse(url=f"/?cluster_id={new_cluster_id}", status_code=303)


async def delete_features(request):

    data = await request.form()
    return_url = f"/cluster/{data['cluster_id']}" if "cluster_id" in data else "/"

    feature_ids = [int(feature_id) for feature_id in data.getlist("feature_id")]
    request.app.state.index.delete_features(feature_ids)

    return RedirectResponse(url=return_url, status_code=303)


async def create_cluster(request):

    data = await request.form()

    feature_ids = [int(feature_id) for feature_id in data.getlist("feature_id")]
    new_cluster_id = request.app.state.index.create_cluster_from_features(feature_ids)

    return RedirectResponse(url=f"/cluster/{new_cluster_id}", status_code=303)


async def cluster(request):

    template = templates.get_template("cluster.html")

    cluster_id = int(request.path_params["cluster_id"])

    features = request.app.state.index.cluster_features(cluster_id)
    n_features = len(features)

    if "feature_id" in request.query_params:
        query = request.app.state.index[int(request.query_params["feature_id"])]
        features = request.app.state.index.pivot_clusters_by_query(
            query, cluster_ids=[cluster_id], top_k=n_features
        )[0][-1]
    elif "cluster_id" in request.query_params:
        query = request.app.state.index.cluster_docs(
            int(request.query_params["cluster_id"])
        )
        features = request.app.state.index.pivot_clusters_by_query(
            query, cluster_ids=[cluster_id], top_k=n_features
        )[0][-1]

    return HTMLResponse(template.render(cluster_id=cluster_id, features=features))


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
        Route("/cluster/create", endpoint=create_cluster, methods=["post"]),
        Route("/cluster/delete", endpoint=delete_clusters, methods=["post"]),
        Route("/cluster/merge", endpoint=merge_clusters, methods=["post"]),
        Route("/cluster/{cluster_id:int}", endpoint=cluster),
        Route("/feature/delete", endpoint=delete_features, methods=["post"]),
        Route("/query", endpoint=query),
    ]

    app = Starlette(debug=True, routes=routes)

    if os.getenv("hyperreal_index_path"):
        app.state.index = index.Index(os.getenv("hyperreal_index_path"))

    return app


app = create_app()
