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

import hyperreal.index


templates = Environment(
    loader=PackageLoader("hyperreal"), autoescape=select_autoescape()
)


import cherrypy
from jinja2 import PackageLoader, Environment, select_autoescape

from hyperreal import index


templates = Environment(
    loader=PackageLoader("hyperreal"), autoescape=select_autoescape()
)


@cherrypy.tools.register("before_handler")
def lookup_index():
    """This tool looks up the provided index using the configured webserver."""
    index_id = int(cherrypy.request.params["index_id"])
    cherrypy.request.index = cherrypy.request.config["index_server"].index(index_id)


@cherrypy.tools.register("on_end_request")
def cleanup_index():
    """If an index has been setup for this request, close it."""

    if hasattr(cherrypy.request, "index"):
        cherrypy.request.index.close()


class Cluster:
    @cherrypy.expose
    def index(self, index_id, cluster_id):
        return f"cluster {cluster_id}"


@cherrypy.popargs("cluster_id", handler=Cluster())
class ClusterOverview:
    @cherrypy.expose
    def index(self, index_id):
        return "cluster"

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    def delete(self, index_id, cluster_id=None):
        return f"{cluster_id} deleted"
        #     cluster_ids = [int(cluster_id) for cluster_id in data.getlist("cluster_id")]
        #     request.app.state.index.delete_clusters(cluster_ids)

        #     return RedirectResponse(url="/", status_code=303)

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    def create(self, index_id, feature_id=None):
        return f"{feature_id} created"

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    def merge(self, index_id, cluster_id=None):
        return f"{cluster_id} merged"


class FeatureOverview:
    @cherrypy.expose
    def index(self, index_id, feature_id=None):
        return "feature"

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    def remove_from_model(self, index_id, feature_id=None):
        return f"{feature_id} deleted"


@cherrypy.popargs("index_id")
@cherrypy.tools.cleanup_index()
@cherrypy.tools.lookup_index()
class Index:

    cluster = ClusterOverview()
    feature = FeatureOverview()

    def __init__(self):
        # self.index = index.Index("test_data/expat_index.db")
        pass

    @cherrypy.expose
    def index(self, index_id, feature_id=None, cluster_id=None):

        template = templates.get_template("index.html")

        rendered_docs = []

        if feature_id is not None:
            query = cherrypy.request.index[int(feature_id)]
            clusters = cherrypy.request.index.pivot_clusters_by_query(query)

            if cherrypy.request.index.corpus is not None:
                rendered_docs = cherrypy.request.index.get_rendered_docs(
                    query, random_sample_size=5
                )

            total_docs = len(query)

        elif cluster_id is not None:
            query = cherrypy.request.index.cluster_docs(int(cluster_id))
            clusters = cherrypy.request.index.pivot_clusters_by_query(query)

            if cherrypy.request.index.corpus is not None:
                rendered_docs = cherrypy.request.index.get_rendered_docs(
                    query, random_sample_size=5
                )

            total_docs = len(query)

        else:
            clusters = cherrypy.request.index.top_cluster_features()
            total_docs = 0

        return template.render(
            clusters=clusters,
            total_docs=total_docs,
            rendered_docs=rendered_docs,
            # Design note: might be worth letting templates grab the request
            # context, and avoid passing this around for everything that
            # needs it?
            index_id=index_id,
        )


@cherrypy.popargs("index_id", handler=Index())
class IndexOverview:
    @cherrypy.expose
    def index(self):
        return f"index overview method {cherrypy.request.config}"


class Root:
    """
    There will be more things at the base layer in the future.

    But for now we will only worry about the /index layer and
    associated operations.
    """

    @cherrypy.expose
    def index(self):
        raise cherrypy.HTTPRedirect("/index/")


class SingleIndexServer:
    def __init__(
        self,
        index_path,
        corpus_class=None,
        corpus_args=None,
        corpus_kwargs=None,
        pool=None,
        mp_context=None,
    ):
        """
        Helper class for serving a single index via the webserver.

        An index will be created on demand when a request requires.

        This will create a single multiprocessing pool to be shared across
        indexes.

        """
        self.corpus_class = corpus_class
        self.corpus_args = corpus_args
        self.corpus_kwargs = corpus_kwargs
        self.index_path = index_path

        self.mp_context = mp_context
        self.pool = pool

    def index(self, index_id):
        if index_id != 0:
            raise cherrypy.HTTPError(404)

        if self.corpus_class:
            corpus = self.corpus_class(
                *(self.corpus_args or []), **(self.corpus_kwargs or {})
            )
        else:
            corpus = None

        return hyperreal.index.Index(
            self.index_path, corpus=corpus, pool=self.pool, mp_context=self.mp_context
        )

    def list_indices(self):
        return {
            0: (
                self.index_path,
                self.corpus_class,
                self.corpus_args,
                self.corpus_kwargs,
            )
        }


def launch_web_server(index_server, auto_reload=False):
    """Launch the web server using the given instance of an index server."""

    if not auto_reload:
        cherrypy.config.update({"global": {"engine.autoreload.on": False}})

    cherrypy.tree.mount(
        Root(),
        "/",
        {
            "/": {
                "tools.response_headers.on": True,
                "tools.response_headers.headers": [
                    ("Connection", "close"),
                ],
            }
        },
    )
    cherrypy.tree.mount(
        IndexOverview(),
        "/index",
        {
            "/": {
                "index_server": index_server,
                "tools.response_headers.on": True,
                "tools.response_headers.headers": [
                    ("Connection", "close"),
                ],
            }
        },
    )

    cherrypy.engine.signals.subscribe()
    cherrypy.engine.start()
    cherrypy.engine.block()


# async def homepage(request):

#     template = templates.get_template("index.html")

#     if "feature_id" in request.query_params:
#         query = request.app.state.index[int(request.query_params["feature_id"])]
#         clusters = request.app.state.index.pivot_clusters_by_query(query)
#         rendered_docs = request.app.state.index.get_rendered_docs(
#             query, random_sample_size=5
#         )
#         total_docs = len(query)
#     elif "cluster_id" in request.query_params:
#         query = request.app.state.index.cluster_docs(
#             int(request.query_params["cluster_id"])
#         )
#         clusters = request.app.state.index.pivot_clusters_by_query(query)
#         rendered_docs = request.app.state.index.get_rendered_docs(
#             query, random_sample_size=5
#         )
#         total_docs = len(query)
#     else:
#         clusters = request.app.state.index.top_cluster_features()
#         rendered_docs = []
#         total_docs = 0

#     return HTMLResponse(
#         template.render(
#             clusters=clusters, total_docs=total_docs, rendered_docs=rendered_docs
#         )
#     )


# async def delete_clusters(request):

#     data = await request.form()

#     cluster_ids = [int(cluster_id) for cluster_id in data.getlist("cluster_id")]
#     request.app.state.index.delete_clusters(cluster_ids)

#     return RedirectResponse(url="/", status_code=303)


# async def merge_clusters(request):

#     data = await request.form()

#     cluster_ids = [int(cluster_id) for cluster_id in data.getlist("cluster_id")]
#     new_cluster_id = request.app.state.index.merge_clusters(cluster_ids)

#     return RedirectResponse(url=f"/?cluster_id={new_cluster_id}", status_code=303)


# async def delete_features(request):

#     data = await request.form()
#     return_url = f"/cluster/{data['cluster_id']}" if "cluster_id" in data else "/"

#     feature_ids = [int(feature_id) for feature_id in data.getlist("feature_id")]
#     request.app.state.index.delete_features(feature_ids)

#     return RedirectResponse(url=return_url, status_code=303)


# async def create_cluster(request):

#     data = await request.form()

#     feature_ids = [int(feature_id) for feature_id in data.getlist("feature_id")]
#     new_cluster_id = request.app.state.index.create_cluster_from_features(feature_ids)

#     return RedirectResponse(url=f"/cluster/{new_cluster_id}", status_code=303)


# async def cluster(request):

#     template = templates.get_template("cluster.html")

#     cluster_id = int(request.path_params["cluster_id"])

#     features = request.app.state.index.cluster_features(cluster_id)
#     n_features = len(features)

#     if "feature_id" in request.query_params:
#         query = request.app.state.index[int(request.query_params["feature_id"])]
#         features = request.app.state.index.pivot_clusters_by_query(
#             query, cluster_ids=[cluster_id], top_k=n_features
#         )[0][-1]
#     elif "cluster_id" in request.query_params:
#         query = request.app.state.index.cluster_docs(
#             int(request.query_params["cluster_id"])
#         )
#         features = request.app.state.index.pivot_clusters_by_query(
#             query, cluster_ids=[cluster_id], top_k=n_features
#         )[0][-1]

#     return HTMLResponse(template.render(cluster_id=cluster_id, features=features))


# async def query(request):
#     query = parse_qsl(request.url.query)

#     results = request.app.state.index[query[0]]

#     for feature in query[1:]:
#         results |= request.app.state.index[feature]

#     docs = request.app.state.index.get_docs(results[:100])

#     return PlainTextResponse("\n\n".join(str(d[1]) for d in docs))


# def create_app():

#     routes = [
#         Route("/", endpoint=homepage),
#         Route("/cluster/create", endpoint=create_cluster, methods=["post"]),
#         Route("/cluster/delete", endpoint=delete_clusters, methods=["post"]),
#         Route("/cluster/merge", endpoint=merge_clusters, methods=["post"]),
#         Route("/cluster/{cluster_id:int}", endpoint=cluster),
#         Route("/feature/delete", endpoint=delete_features, methods=["post"]),
#         Route("/query", endpoint=query),
#     ]

#     app = Starlette(debug=True, routes=routes)

#     if os.getenv("hyperreal_index_path"):
#         app.state.index = index.Index(os.getenv("hyperreal_index_path"))

#     return app


# app = create_app()
