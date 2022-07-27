"""
Cherrypy based webserver for serving an index (or in future) a set of indexes.

"""
import argparse
import os
from urllib.parse import parse_qsl

from jinja2 import PackageLoader, Environment, select_autoescape

import hyperreal.index
import hyperreal.utilities


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


@cherrypy.tools.register("before_handler")
def ensure_list(**kwargs):
    """Ensure that the given variables are always a list of the given type."""

    for key, converter in kwargs.items():
        value = cherrypy.request.params.get(key)
        if value is None:
            cherrypy.request.params[key] = []
        elif isinstance(value, list):
            cherrypy.request.params[key] = [converter(item) for item in value]
        else:
            cherrypy.request.params[key] = [converter(value)]


class Cluster:
    @cherrypy.expose
    def index(self, index_id, cluster_id, feature_id=None):
        template = templates.get_template("cluster.html")

        cluster_id = int(cluster_id)

        features = cherrypy.request.index.cluster_features(cluster_id)
        n_features = len(features)

        if feature_id is not None:
            query = cherrypy.request.index[int(feature_id)]
            features = cherrypy.request.index.pivot_clusters_by_query(
                query, cluster_ids=[cluster_id], top_k=n_features
            )[0][-1]

        return template.render(
            cluster_id=cluster_id, features=features, index_id=index_id
        )


@cherrypy.popargs("cluster_id", handler=Cluster())
class ClusterOverview:
    @cherrypy.expose
    def index(self, index_id, cluster_id=None, feature_id=None):
        pass

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    @cherrypy.tools.ensure_list(cluster_id=int)
    def delete(self, index_id, cluster_id=None):

        cherrypy.request.index.delete_clusters(cherrypy.request.params["cluster_id"])

        raise cherrypy.HTTPRedirect(f"/index/{index_id}")

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    @cherrypy.tools.ensure_list(feature_id=int)
    def create(self, index_id, feature_id=None, cluster_id=None):

        new_cluster_id = cherrypy.request.index.create_cluster_from_features(
            cherrypy.request.params["feature_id"]
        )
        raise cherrypy.HTTPRedirect(f"/index/{index_id}/cluster/{new_cluster_id}")

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    @cherrypy.tools.ensure_list(cluster_id=int)
    def merge(self, index_id, cluster_id=None):
        merge_cluster_id = cherrypy.request.index.merge_clusters(
            cherrypy.request.params["cluster_id"]
        )
        raise cherrypy.HTTPRedirect(f"/index/{index_id}/cluster/{merge_cluster_id}")


class FeatureOverview:
    @cherrypy.expose
    def index(self, index_id, feature_id=None):
        return "feature"

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    @cherrypy.tools.ensure_list(feature_id=int)
    def remove_from_model(self, index_id, feature_id=None, cluster_id=None):
        cherrypy.request.index.delete_features(cherrypy.request.params["feature_id"])
        if cluster_id is not None:
            raise cherrypy.HTTPRedirect(f"/index/{index_id}/cluster/{cluster_id}")
        else:
            raise cherrypy.HTTPRedirect(f"/index/{index_id}/")


@cherrypy.popargs("index_id")
@cherrypy.tools.cleanup_index()
@cherrypy.tools.lookup_index()
class Index:

    cluster = ClusterOverview()
    feature = FeatureOverview()

    @cherrypy.expose
    def index(self, index_id, feature_id=None, cluster_id=None, top_k="5"):

        template = templates.get_template("index.html")

        rendered_docs = []
        highlight_cluster_id = None
        highlight_feature_id = None

        if feature_id is not None:
            query = cherrypy.request.index[int(feature_id)]
            clusters = cherrypy.request.index.pivot_clusters_by_query(query)

            if cherrypy.request.index.corpus is not None:
                rendered_docs = cherrypy.request.index.render_docs(
                    query, random_sample_size=int(top_k)
                )

            total_docs = len(query)
            highlight_feature_id = int(feature_id)

        elif cluster_id is not None:
            query, bitslice = cherrypy.request.index.cluster_query(int(cluster_id))
            clusters = cherrypy.request.index.pivot_clusters_by_query(query)

            if cherrypy.request.index.corpus is not None:
                ranked = hyperreal.utilities.bstm(query, bitslice, int(top_k))
                rendered_docs = cherrypy.request.index.render_docs(ranked)

            total_docs = len(query)
            highlight_cluster_id = int(cluster_id)

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
            highlight_feature_id=highlight_feature_id,
            highlight_cluster_id=highlight_cluster_id,
        )


@cherrypy.popargs("index_id", handler=Index())
class IndexOverview:
    @cherrypy.expose
    def index(self):

        template = templates.get_template("index_listing.html")
        indices = cherrypy.request.config["index_server"].list_indices()
        return template.render(indices=indices)


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
