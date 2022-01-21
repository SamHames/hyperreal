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

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, PlainTextResponse
from starlette.staticfiles import StaticFiles
from starlette.routing import Route
from starlette.templating import Jinja2Templates
import uvicorn

from hyperreal import index


async def homepage(request):
    return PlainTextResponse("hello")


async def cluster(request):

    features = request.app.state.index.cluster_features(
        request.path_params["cluster_id"]
    )

    return PlainTextResponse("\n".join(", ".join(str(i) for i in f) for f in features))


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
        Route("/cluster/{cluster_id:int}", endpoint=cluster),
        Route("/query", endpoint=query),
    ]

    app = Starlette(debug=True, routes=routes)

    if os.getenv("hyperreal_index_path"):
        app.state.index = index.Index(os.getenv("hyperreal_index_path"))

    return app


app = create_app()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("index_path", help="Which index to serve.")

    args = parser.parse_args()

    if not index.Index.is_index_db(args.index_path):
        raise ValueError(f"{args.index_path} is not a valid index file.")

    os.environ["hyperreal_index_path"] = args.index_path

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
