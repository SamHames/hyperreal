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

from lxml import etree
from markupsafe import Markup

from hyperreal.index import Index
from hyperreal.corpus import SqliteBackedCorpus
from hyperreal.utilities import tokens, presentable_tokens
import hyperreal.server


class HansardCorpus(SqliteBackedCorpus):
    CORPUS_TYPE = "Australian Federal Hansard"

    table_fields = ["date", "house", "debate_title", "subdebate_title", "speech"]

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
                            speech.speech_id,
                            speech.date,
                            speech.house,
                            speech_number,
                            debate.debate_id,
                            debate.debate,
                            debate.subdebate_1,
                            debate.subdebate_2,
                            xml_url,
                            html_url,
                            speech_xml
                        from speech
                        inner join debate using(debate_id)
                        inner join transcript using(transcript_id)
                        where speech.speech_id = ?
                        """,
                        [key],
                    )
                )[0]

                yield key, doc

        finally:
            self.db.execute("release docs")

    def index(self, doc):
        root = etree.fromstring(doc["speech_xml"])

        # TODO: This will need additional work to work with all the different
        # hansard schemas.
        talkers = root.xpath("//talker")
        for elem in talkers:
            elem.clear()

        speech_tokens = tokens(" ".join(root.itertext()))

        speech_date = date.fromisoformat(doc["date"])
        speech_month = speech_date.replace(day=1)
        speech_year = speech_month.replace(month=1)

        return {
            "speech": speech_tokens,
            "house": set([doc["house"]]),
            "debate": set([doc["debate"]]),
            "subdebate_1": set([doc["subdebate_1"]]),
            "subdebate_2": set([doc["subdebate_2"]]),
            "debate_id": set([doc["debate_id"]]),
            "speech_date": set([speech_date.isoformat()]),
            "speech_month": set([speech_month.isoformat()]),
            "speech_year": set([speech_year.isoformat()]),
        }

    def pretty_index(self, doc):
        root = etree.fromstring(doc["speech_xml"])

        # TODO: This will need additional work to work with all the different
        # hansard schemas.
        talkers = root.xpath("//talker")
        for elem in talkers:
            elem.clear()

        speech_tokens = presentable_tokens(" ".join(root.itertext()))

        speech_date = date.fromisoformat(doc["date"])

        return {
            "speech": speech_tokens,
            "house": set([doc["house"]]),
            "debate": set([doc["debate"]]),
            "subdebate_1": set([doc["subdebate_1"]]),
            "subdebate_2": set([doc["subdebate_2"]]),
            "speech_date": set([speech_date.isoformat()]),
            "transcript_url": set([doc["html_url"]]),
        }

    def render_docs_html(self, doc_keys):
        """Return the given documents as HTML."""
        docs = []

        for key, doc in self.docs(doc_keys=doc_keys):
            tree = etree.fromstring(doc["speech_xml"])

            text = " ".join(s for s in tree.itertext())
            text = "<br />".join(s for s in text.splitlines() if s.strip())

            summary = ", ".join(
                doc[item]
                for item in ("date", "house", "debate", "subdebate_1", "subdebate_2")
                if doc[item]
            )

            doc_html = """
                <details>
                    <summary>{}</summary>
                    <div>
                        <a href="{}">See full day</a>
                    </div>
                    <div>{}</div>
                </details>
                """.format(
                summary, doc["html_url"], text
            )

            docs.append((key, Markup(doc_html)))

        return docs

    def keys(self):
        """The speeches are the central document."""
        return (r["speech_id"] for r in self.db.execute("select speech_id from speech"))


if __name__ == "__main__":
    try:
        os.remove("process_hansard.log")
    except FileNotFoundError:
        pass

    logging.basicConfig(filename="process_hansard.log", level=logging.INFO)
    corpus = HansardCorpus("hansard.db")

    args = sys.argv[1:]

    mp_context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(mp_context=mp_context) as pool:
        idx = Index("hansard_index.db", corpus=corpus, pool=pool)

        if "index" in args:
            idx.index(doc_batch_size=100000, position_window_size=1)

        if "cluster" in args:
            idx.initialise_clusters(
                n_clusters=512, min_docs=10, include_fields=["speech"]
            )
            idx.refine_clusters(iterations=50, group_test_batches=100)

        index_server = hyperreal.server.SingleIndexServer(
            "hansard_index.db",
            corpus_class=HansardCorpus,
            corpus_args=["hansard.db"],
            pool=pool,
        )
        engine = hyperreal.server.launch_web_server(index_server)
        engine.block()
