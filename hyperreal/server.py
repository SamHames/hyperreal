"""
Server components.

The server consists of the following components:

1. Starlette frontend for handling requests.
2. An interactive process pool for handling significant computation by the front end
3. A background queue and process for long running tasks.

"""
import argparse

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

    print(request.query_params)
    print(request.query_params["value"])
    print(await request.form())

    return PlainTextResponse("\n".join(", ".join(str(i) for i in f) for f in features))


def create_app(index_path):

    routes = [
        Route("/", endpoint=homepage),
        Route("/cluster/{cluster_id:int}", endpoint=cluster),
    ]

    app = Starlette(debug=True, routes=routes)

    app.state.index = index.Index(index_path)

    return app


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("index_path", help="Which index to serve.")

    args = parser.parse_args()

    if not index.Index.is_index_db(args.index_path):
        raise ValueError(f"{args.index_path} is not a valid index file.")

    app = create_app(args.index_path)

    uvicorn.run(app, host="0.0.0.0", port=8000)
