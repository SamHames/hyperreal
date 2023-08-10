"""
This module creates a corpus of hansard speeches - the data model is based on (and
uses code derived from) Tim Sherratt's GLAM Workbench [1].

Note that there are differences in structure between the historical
(-2005) Hansard data and data since that point - they can broadly be brought
together into the same schema but there might be edge cases I've missed.

Note also: some of the source data is not properly structured XML - this
approach uses BeautifulSoup in XML mode both for convenience and also because
it does better with malformed XML data in general.

[1] Sherratt, Tim. (2019). GLAM-Workbench/australian-commonwealth-hansard
(v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.3544706

"""

from collections import defaultdict, namedtuple
import concurrent.futures as cf
from datetime import date
import logging
import multiprocessing as mp
import os
import traceback
from xml.etree import ElementTree
import zipfile

from markupsafe import Markup

from hyperreal.index import Index
from hyperreal.corpus import SqliteBackedCorpus
from hyperreal.utilities import tokens
import hyperreal.server


class HansardCorpus(SqliteBackedCorpus):
    CORPUS_TYPE = "Australian Federal Hansard"

    table_fields = ["date", "house", "debate_title", "subdebate_title", "speech"]

    def __init__(self, db_path):
        """
        A relational model of the proceedings of the Australian Federal
        parliament.

        This models the basic elements of the data as recorded in the Hansard XML.

        The fundamental unit for this data processing is the speech, which is
        part of a named debate and optional subdebate.

        """
        super().__init__(db_path)

    def _insert_rows(self, rows):
        self.db.execute(
            "insert into session values (?, ?, ?, ?, ?, ?)", rows["session"]
        )
        self.db.executemany("insert into debate values (?, ?, ?, ?, ?)", rows["debate"])
        self.db.executemany(
            "insert into subdebate values (?, ?, ?, ?, ?, ?)", rows["subdebate"]
        )
        self.db.executemany(
            "insert into speech values (null, ?, ?, ?, ?, ?, ?, ?)", rows["speech"]
        )

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
                            date,
                            house,
                            debate.title as debate_title,
                            subdebate.title as subdebate_title,
                            speech
                        from speech
                        left outer join session using (date, house)
                        left outer join debate using (date, house, debate_no)
                        left outer join subdebate using (date, house, debate_no, subdebate_no)
                        where speech.speech_id = ?
                        """,
                        [key],
                    )
                )[0]

                yield key, doc

        finally:
            self.db.execute("release docs")

    def index(self, doc):
        try:
            root = ElementTree.fromstring(doc["speech"])
        except ElementTree.ParseError:
            # Special case 1 - stray closing >
            root = ElementTree.fromstring(doc["speech"].replace(b"&gt;\n", b""))

        talkers = root.findall(".//talker")
        # Make sure older style hansard talker tags aren't indexed as text.
        for elem in talkers:
            elem.clear()

        speech_tokens = tokens(" ".join(root.itertext()))

        speech_date = date.fromisoformat(doc["date"])
        speech_month = speech_date.replace(day=1)
        speech_year = speech_month.replace(month=1)

        return {
            "speech": speech_tokens,
            "speech_date": set([speech_date.isoformat()]),
            "speech_month": set([speech_month.isoformat()]),
            "speech_year": set([speech_year.isoformat()]),
        }

    def render_docs_html(self, doc_keys):
        """Return the given documents as HTML."""
        docs = []

        for key, doc in self.docs(doc_keys=doc_keys):
            tree = ElementTree.ElementTree(ElementTree.fromstring(doc["speech"]))

            # TODO: render these speeches better...
            if doc["date"] < "2006-02-07":
                paras = [
                    f"<p>{node.text}</p>"
                    for node in tree.iter()
                    if node.tag in {"para", "quote", "list"}
                ]
            else:
                paras = [doc["speech"].decode("utf8")]

            summary = ", ".join(
                doc[item]
                for item in ("date", "house", "debate_title", "subdebate_title")
                if doc[item]
            )

            doc_html = "<details><summary>{}</summary><div>{}<div></details>".format(
                summary, "\n".join(paras)
            )

            docs.append((key, Markup(doc_html)))

        return docs

    def render_docs_table(self, doc_keys):
        """Return the given documents as HTML."""

        docs = self.docs(doc_keys=doc_keys)

        out_docs = []
        for key, doc in docs:
            speech = ElementTree.fromstring(doc["speech"])
            ElementTree.indent(speech)
            doc["speech"] = "".join(speech.itertext())
            out_docs.append((key, doc))

        return out_docs

    def keys(self):
        """The speeches are the central document."""
        return (r["speech_id"] for r in self.db.execute("select speech_id from speech"))

    def replace_speeches(self, historic_zip, current_zip):
        """
        Replace the existing corpus with newly processed files.

        Pass paths to the zipped historic and current files as downloaded
        with the download_data.py script.

        """
        self.db.execute("pragma journal_mode=WAL")
        self.db.execute("savepoint add_texts")

        try:
            self.db.execute("drop table if exists session")
            self.db.execute(
                """
                create table session (
                    date,
                    parliament_no,
                    house,
                    session_no,
                    period_no,
                    source_file,
                    primary key (date, house)
                )
                """
            )
            self.db.execute("drop table if exists debate")
            self.db.execute(
                """
                create table debate (
                    date,
                    house,
                    debate_no,
                    title,
                    type,
                    primary key (date, house, debate_no)
                )
                """,
            )
            self.db.execute("drop table if exists subdebate")
            self.db.execute(
                """
                create table subdebate (
                    date,
                    house,
                    debate_no,
                    subdebate_no,
                    title,
                    type,
                    primary key (date, house, debate_no, subdebate_no),
                    foreign key (date, house, debate_no) references debate
                )
                """,
            )
            self.db.execute("drop table if exists speech")
            self.db.execute(
                """
                create table speech(
                    speech_id integer primary key,
                    date,
                    house,
                    debate_no,
                    subdebate_no,
                    speech_no,
                    speech_type,
                    speech,
                    unique (
                        date,
                        house,
                        debate_no,
                        subdebate_no,
                        speech_no
                    ),
                    foreign key (date, house, debate_no) references debate,
                    foreign key (date, house, debate_no, subdebate_no) references subdebate
                )
                """
            )

            self.db.execute("drop table if exists interjection")

            files = (historic_zip, current_zip)
            functions = (process_xml_historic, process_xml_current)

            for file_path, process_function in zip(files, functions):
                with zipfile.ZipFile(file_path, "r") as zip_data:
                    to_process = [
                        fileinfo
                        for fileinfo in zip_data.infolist()
                        if (
                            not fileinfo.is_dir() and fileinfo.filename.endswith(".xml")
                        )
                    ]
                    n_process = len(to_process)

                    for progress, fileinfo in enumerate(to_process):
                        raw_data = zip_data.open(fileinfo).read()
                        rows = process_function(raw_data, fileinfo.filename)

                        if rows:
                            try:
                                self._insert_rows(rows)
                            except Exception as e:
                                print(f"Error inserting file {fileinfo.filename}")
                                traceback.print_exc()

                        print(f"{progress + 1} / {n_process}", end="\r", flush=True)
                print(f"{file_path} complete.")

        except Exception:
            self.db.execute("rollback to add_texts")
            raise

        finally:
            self.db.execute("release add_texts")


Session = namedtuple(
    "Session",
    ["date", "parliament_no", "house", "session_no", "period_no", "source_file"],
)

Debate = namedtuple(
    "Debate",
    [
        "date",
        "house",
        "debate_no",
        "title",
        "type",
    ],
)

SubDebate = namedtuple(
    "SubDebate",
    [
        "date",
        "house",
        "debate_no",
        "subdebate_no",
        "title",
        "type",
    ],
)


Speech = namedtuple(
    "Speech",
    [
        "date",
        "house",
        "debate_no",
        "subdebate_no",
        "speech_no",
        "type",
        "speech",
    ],
)


Interjection = namedtuple(
    "Interjection",
    ["date", "house", "debate_no", "subdebate_no", "speech_no", "interjection"],
)


def process_xml_historic(xml_data, source_file):
    """
    Process an XML transcript from the historical data.

    This returns a dictionary mapping a table name to rows for that table.

    Note that currently the segments returned are just raw XML blobs,
    no additional filtering.

    """

    rows = defaultdict(list)

    if xml_data.startswith(b"\n<!DOCTYPE html"):
        # There's a few dodgy not really XML files.
        return rows

    root = ElementTree.fromstring(xml_data)

    header = root.find("session.header")

    session = Session(
        header.find("date").text,
        int(header.find("parliament.no").text),
        header.find("chamber").text,
        int(header.find("session.no").text),
        int(header.find("period.no").text),
        source_file,
    )

    rows["session"] = session

    transcript = root.find("chamber.xscript")

    for debate_seq, debate in enumerate(transcript.findall("debate")):
        # Note this can be missing - sometimes it's just a series of subdebates instead.
        info = debate.find("debateinfo")

        if info is not None:
            title = (
                info.find("title").text
                if info.find("title") is not None
                else "<missing title>"
            )
            debate_type = (
                info.find("type").text
                if info.find("type") is not None
                else "<missing type>"
            )
        # If missing,
        else:
            title = "<missing debateinfo>"
            debate_type = "<missing debateinfo>"

        debate_info = Debate(
            session.date, session.house, debate_seq, title, debate_type
        )

        rows["debate"].append(debate_info)

        # This is not present at the debate level, only when we recurse.
        subdebate_seq = None

        # Note that questions and answers aren't explicitly delimited until
        # later in the corpus.
        speeches = (
            debate.findall("speech")
            + debate.findall("question")
            + debate.findall("answer")
        )

        for speech_seq, speech in enumerate(speeches):
            speech_info = Speech(
                session.date,
                session.house,
                debate_seq,
                subdebate_seq,
                speech_seq,
                speech.tag,
                ElementTree.tostring(speech),
            )
            rows["speech"].append(speech_info)

        for subdebate_seq, subdebate in enumerate(debate.findall("subdebate.1")):
            info = subdebate.find("subdebateinfo")

            if info is not None:
                title = (
                    info.find("title").text
                    if info.find("title") is not None
                    else "<missing title>"
                )
                subdebate_type = (
                    info.find("type").text
                    if info.find("type") is not None
                    else "<missing type>"
                )
            # If missing,
            else:
                title = "<missing debateinfo>"
                subdebate_type = "<missing debateinfo>"

            subdebate_info = SubDebate(
                session.date,
                session.house,
                debate_seq,
                subdebate_seq,
                title,
                subdebate_type,
            )

            rows["subdebate"].append(subdebate_info)

            # Note that questions and answers aren't explicitly delimited until
            # later in the corpus.
            speeches = (
                subdebate.findall("speech")
                + subdebate.findall("question")
                + subdebate.findall("answer")
            )

            for speech_seq, speech_data in enumerate(speeches):
                speech = Speech(
                    session.date,
                    session.house,
                    debate_seq,
                    subdebate_seq,
                    speech_seq,
                    speech_data.tag,
                    ElementTree.tostring(
                        speech_data,
                    ),
                )
                rows["speech"].append(speech)

    return rows


def process_xml_current(xml_data, source_file):
    """
    Process an XML transcript from the current data.

    This returns a dictionary mapping a table name to rows for that table.

    Note that currently the segments returned are just raw XML blobs,
    no additional filtering.

    There is a lot of additional metadata in the current files that isn't
    present in the historical data. TODO: work out whether it makes sense to
    base the schema on the current and fill in as much as possible from the
    historical instead of current?

    Modelling notes:

    For consistency with the historical data that wraps up longer segments
    into a single speech, <speech> segments are merged together with their
    interjections. So a sequence like the following is merged into a single
    speech element:

        <speech type="speech">Content</speech>
        <speech type="interjection">Content</speech>
        <speech type="continuing">Content</speech>

    Historical data uses debate/subdebate - for the modern data it's more
    major heading/minor heading, and these are flat items in the stream
    rather than headers for containers.

    """

    rows = defaultdict(list)

    if "represent" in source_file:
        house = "REPS"
    else:
        house = "SENATE"

    date = source_file.split("/")[1].split(".")[0]

    session = Session(
        date,
        # TODO: Map dates back to parliament numbers as it is no longer in
        # this file.
        None,
        house,
        # The session and other numbers can probably be derived from sequence
        # or somewhere else?
        None,
        None,
        source_file,
    )
    rows["session"] = session

    root = ElementTree.fromstring(xml_data)

    debate_seq = 0
    subdebate_seq = 0
    speech_seq = 0
    current_speeches = []

    def process_speech_tags(current_speeches):
        """
        For consistency with the historical data and to provide complete
        context, we're going to glue together speech tags that represent a
        contiguous unit including interjections and continuations. Note that
        interjections by individuals appear to be labelled as interjections,
        but general interjections or affirmations(hear, hear! or "Members of
        the opposition interjecting") are labelled as speech elements
        followed by a continuation. Speech elements that are followed by a
        continuation are considered like interjections as part of the same
        contiguous speech by a speaker. TODO: investigate the continuation
        attribute further and see if that can be used to disambiguate.

        """
        talktypes = [speech_elem.attrib["talktype"] for speech_elem in current_speeches]
        # Add sentinel so we can always check the next element.
        talktypes.append(None)
        contiguous_starts = [
            i
            for i, talktype in enumerate(talktypes)
            if talktype == "speech" and talktypes[i + 1] != "continuation"
        ]

        # Make sure that we include the first segment if it does not start
        # with a talktype of speech:
        if not contiguous_starts or contiguous_starts[0] != 0:
            contiguous_starts.insert(0, 0)

        # Handle the final segment
        contiguous_starts.append(len(talktypes) + 1)
        contiguous_speeches = []

        # Now generate the actuals speeches
        for start, end in zip(contiguous_starts, contiguous_starts[1:]):
            contiguous_speech = ElementTree.Element("div")
            for speech_elem in current_speeches[start:end]:
                contiguous_speech.append(speech_elem)

            contiguous_speeches.append(ElementTree.tostring(contiguous_speech))

        return contiguous_speeches

    for elem in root:
        # Process the batch of speeches from the last debate/subdebate
        # sequence before we reset the debate/subdebate
        if current_speeches and elem.tag in ("major-heading", "minor-heading"):
            for speech in process_speech_tags(current_speeches):
                speech_info = Speech(
                    session.date,
                    session.house,
                    debate_seq,
                    subdebate_seq,
                    speech_seq,
                    None,
                    speech,
                )

                speech_seq += 1
                rows["speech"].append(speech_info)

            current_speeches = []

        if elem.tag == "major-heading":
            debate_info = Debate(
                session.date, session.house, debate_seq, elem.text.strip(), None
            )
            rows["debate"].append(debate_info)

            debate_seq += 1

            # This is a sentinel to test the assumption that every major
            # heading also has a following minor heading - this lets
            # us simplify the processing if it remains true.
            subdebate_info = None

        elif elem.tag == "minor-heading":
            subdebate_info = SubDebate(
                session.date,
                session.house,
                debate_seq,
                subdebate_seq,
                elem.text.strip(),
                None,
            )
            rows["subdebate"].append(subdebate_info)

            subdebate_seq += 1

        elif elem.tag == "speech":
            current_speeches.append(elem)

    # Make sure to process the last batch in the file!
    return rows


if __name__ == "__main__":
    try:
        os.remove("process_hansard.log")
    except FileNotFoundError:
        pass

    logging.basicConfig(filename="process_hansard.log", level=logging.INFO)
    corpus = HansardCorpus("test.db")

    corpus.replace_speeches(
        os.path.join("data", "hansard_1901-1980_1998-2005.zip"),
        os.path.join("data", "hansard_2005-present.zip"),
    )

    mp_context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(mp_context=mp_context) as pool:
        idx = Index("test_index.db", corpus=corpus, pool=pool)
        idx.index(doc_batch_size=10000, position_window_size=1)
        idx.initialise_clusters(n_clusters=512, min_docs=10, include_fields=["speech"])
        idx.refine_clusters(iterations=50)

        index_server = hyperreal.server.SingleIndexServer(
            "test_index.db",
            corpus_class=HansardCorpus,
            corpus_args=["test.db"],
            pool=pool,
        )
        hyperreal.server.launch_web_server(index_server)
