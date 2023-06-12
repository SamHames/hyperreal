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
    def __init__(self, db_path):
        """
        A relational model of the proceedings of the Australian Federal
        parliament.

        This models the basic elements of the data as recorded in the Hansard XML.

        The fundamental unit for this data processing is the speech, which is
        part of a named debate and optional subdebate.

        """
        super().__init__(db_path)

    def process_historical_transcript(self, xmldata):
        pass

    def process_current_transcript(self, xmldata):
        pass

    def _insert_rows(self, rows):
        self.db.execute(
            "insert into session values (?, ?, ?, ?, ?, ?)", rows["session"]
        )
        self.db.executemany("insert into debate values (?, ?, ?, ?, ?)", rows["debate"])
        self.db.executemany(
            "insert into subdebate values (?, ?, ?, ?, ?, ?)", rows["subdebate"]
        )
        self.db.executemany(
            "insert into speech values (?, ?, ?, ?, ?, ?, ?)", rows["speech"]
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
                            speech.rowid,
                            speech,
                            date,
                            house,
                            debate.title as debate_title,
                            subdebate.title as subdebate_title
                        from speech
                        left outer join session using (date, house)
                        left outer join debate using (date, house, debate_no)
                        left outer join subdebate using (date, house, debate_no, subdebate_no)
                        where speech.rowid = ?
                        """,
                        [key],
                    )
                )[0]

                yield key, doc

        finally:
            self.db.execute("release docs")

    def index(self, doc):
        root = ElementTree.fromstring(doc["speech"])

        speech_tokens = []

        for tag in ["para", "quote", "list"]:
            for element in root.findall(f".//{tag}"):
                speech_tokens.extend(tokens(element.text or ""))
                speech_tokens.append(None)

        return {"speech": speech_tokens}

    def render_docs_html(self, doc_keys):
        """Return the given documents as HTML."""
        docs = []

        for key, doc in self.docs(doc_keys=doc_keys):
            tree = ElementTree.ElementTree(ElementTree.fromstring(doc["speech"]))

            paras = [
                f"<p>{node.text}</p>"
                for node in tree.iter()
                if node.tag in {"para", "quote", "list"}
            ]

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

    def keys(self):
        """The keys are the rowids on the speech table."""
        return (r["rowid"] for r in self.db.execute("select rowid from speech"))

    def replace_speeches(self, *zipfiles):
        """
        Replace the existing corpus with newly processed files.

        Pass a sequence of zipfiles, and all xml files in each will
        be processed.

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
                    date,
                    house,
                    debate_no,
                    subdebate_no,
                    speech_no,
                    speech_type,
                    speech,
                    primary key (
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

            for filepath in zipfiles:
                with zipfile.ZipFile(filepath, "r") as zip_data:
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
                        rows = process_xml_historic(raw_data, fileinfo.filename)

                        if rows:
                            try:
                                self._insert_rows(rows)
                            except Exception as e:
                                print(f"Error inserting file {fileinfo.filename}")
                                traceback.print_exc()

                        print(f"{progress + 1} / {n_process}", end="\r", flush=True)

        except Exception:
            # self.db.execute("rollback to add_texts")
            raise

        finally:
            self.db.execute("release add_texts")


def count_words(para):
    """
    Count the number of words in an element.
    """
    words = 0
    for string in para.stripped_strings:
        words += len(string.split())
    return words


def get_paras(section):
    """
    Find all the para type containers in an element and count the total number of words.
    """
    words = 0
    for para in section.find_all(["para", "quote", "list"], recursive=False):
        words += count_words(para)
    return words


def get_words_in_speech(start, speech):
    """
    Get the top-level containers in a speech and find the total number of words across them all.
    """
    words = 0
    words += get_paras(start)
    words += get_paras(speech)
    for cont in speech.find_all("continue", recursive=False):
        cont_start = cont.find("talk.start", recursive=False)
        words += get_paras(cont_start)
        words += get_paras(cont)
    return words


def get_interjections(speech):
    """
    Get details of any interjections within a speech.
    """
    speeches = []
    for index, intj in enumerate(speech.find_all("interjection", recursive=False)):
        start = intj.find("talk.start", recursive=False)
        speaker = start.find("talker")
        name = speaker.find("name", role="metadata").string
        id = speaker.find("name.id").string
        words = get_words_in_speech(start, intj)
        speeches.append(
            {
                "interjection_idx": index,
                "speaker": name,
                "id": id,
                "type": intj.name,
                "words": words,
            }
        )
    return speeches


def get_subdebates(debate):
    """
    Get details of any subdebates within a debate.
    """
    speeches = []
    for index, sub in enumerate(debate.find_all("subdebate.1", recursive=False)):
        subdebate_info = {
            "subdebate_title": sub.subdebateinfo.title.string,
            "subdebate_idx": index,
        }
        new_speeches = get_speeches(sub)
        # Add the subdebate info to the speech
        for sp in new_speeches:
            sp.update(subdebate_info)
        speeches += new_speeches
    return speeches


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
                debate.findall("speech")
                + debate.findall("question")
                + debate.findall("answer")
            )

            for speech_seq, speech_data in enumerate(speeches):
                speech = Speech(
                    session.date,
                    session.house,
                    debate_seq,
                    subdebate_seq,
                    speech_seq,
                    speech_data.tag,
                    ElementTree.tostring(speech_data),
                )
                rows["speech"].append(speech)

    return rows


if __name__ == "__main__":
    __spec__ = None

    corpus = HansardCorpus("test.db")

    # corpus.replace_speeches(os.path.join("data", "hansard_1901-1980_1998-2005.zip"))

    idx = Index("test_index.db", corpus=corpus)
    # idx.index()

    mp_context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(mp_context=mp_context) as pool:
        index_server = hyperreal.server.SingleIndexServer(
            "test_index.db",
            corpus_class=HansardCorpus,
            corpus_args=["test.db"],
            pool=pool,
        )
        hyperreal.server.launch_web_server(index_server)
