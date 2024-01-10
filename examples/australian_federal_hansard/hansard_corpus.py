"""

[1] Sherratt, Tim. (2019). GLAM-Workbench/australian-commonwealth-hansard
(v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.3544706

"""

from collections import defaultdict, namedtuple
import concurrent.futures as cf
from datetime import date
import logging
import multiprocessing as mp
import os
import sys
import traceback
import zipfile

from lxml import html
from markupsafe import Markup

from hyperreal.index import Index
from hyperreal.corpus import SqliteBackedCorpus
from hyperreal.utilities import tokens
import hyperreal.server


class HansardCorpus(SqliteBackedCorpus):
    CORPUS_TYPE = "Australian Federal Hansard"

    def __init__(self, db_path):
        """
        A corpus to wrap around the data model found in https://github.com/SamHames/hansard-tidy

        """
        super().__init__(db_path)

    def docs(self, doc_keys=None):
        self.db.execute("savepoint docs")
        try:
            # Note that it's valid to pass an empty sequence of doc_keys,
            # so we need to check sentinel explicitly.
            if doc_keys is None:
                doc_keys = self.keys()

            for key in doc_keys:
                doc = list(
                    self.db.execute(
                        """
                        select
                            page_id,
                            url,
                            access_time,
                            debate_id,
                            pp.date,
                            pp.house,
                            pp.parl_no,
                            title,
                            page_html
                        from proceedings_page pp
                        inner join debate using(debate_id)
                        where page_id = ?
                        """,
                        [key],
                    )
                )[0]

                yield key, doc

        finally:
            self.db.execute("release docs")

    def index(self, doc):
        root = html.fromstring(doc["page_html"])

        page_tokens = tokens(" ".join(root.itertext()))

        page_date = date.fromisoformat(doc["date"])
        month = page_date.replace(day=1)
        year = month.replace(month=1)

        return {
            "page": page_tokens,
            "house": set([doc["house"]]),
            "debate_id": set([doc["debate_id"]]),
            "date": set([page_date.isoformat()]),
            "month": set([month.isoformat()]),
            "year": set([year.isoformat()]),
            "parl_no": set([doc["parl_no"]]),
            # Index the formatting elements so we can systematically investigate them
            "html_classes": {
                str(clss) for elem in root.iter() for clss in elem.classes
            },
            "html_tags": {str(elem.tag) for elem in root.iter()},
        }

    def render_docs_html(self, doc_keys):
        """Return the given documents as HTML."""
        docs = []

        for key, doc in self.docs(doc_keys=doc_keys):
            summary = ", ".join(
                doc[item] for item in ("date", "house", "title") if doc[item]
            )

            doc_html = """
                <details>
                    <summary>{}</summary>
                    <div>
                        <a href="{}">See in context</a>
                    </div>
                    <div>{}</div>
                </details>
                """.format(
                summary, doc["url"], doc["page_html"]
            )

            docs.append((key, Markup(doc_html)))

        return docs

    def keys(self):
        """The pages are the central document."""
        return (
            r["page_id"]
            for r in self.db.execute("select page_id from proceedings_page")
        )


if __name__ == "__main__":
    try:
        os.remove("process_hansard.log")
    except FileNotFoundError:
        pass

    db = "tidy_hansard.db"
    db_index = "tidy_hansard_index.db"

    logging.basicConfig(filename="process_hansard.log", level=logging.INFO)
    corpus = HansardCorpus(db)

    args = sys.argv[1:]

    mp_context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(mp_context=mp_context) as pool:
        idx = Index(db_index, corpus=corpus, pool=pool)

        if "index" in args:
            idx.index(doc_batch_size=50000, index_positions=True)

        if "cluster" in args:
            idx.initialise_clusters(
                n_clusters=256, min_docs=100, include_fields=["page"]
            )
            idx.refine_clusters(
                iterations=50, group_test_batches=32, group_test_top_k=2
            )

        index_server = hyperreal.server.SingleIndexServer(
            db_index,
            corpus_class=HansardCorpus,
            corpus_args=[db],
            pool=pool,
        )
        engine = hyperreal.server.launch_web_server(index_server)
        engine.block()
