from html.parser import HTMLParser
from io import StringIO

from pyroaring import BitMap
import regex


word_tokenizer = regex.compile(
    # Note this handles 'quoted' words a little weirdly: 'orange' is tokenised
    # as ["orange", "'"] I'd prefer to tokenise this as p"'", "orange", "'"]
    # but the regex library behaves weirdly. So for now phrase search for
    # quoted strings won't work.
    r"\b\p{Word_Break=WSegSpace}*'?",
    flags=regex.WORD | regex.UNICODE | regex.V1,
)

social_media_cleaner = regex.compile(
    # Find strings that start with either:
    #       - URLS like http://t.co/ or https://t.co/
    #       - @mentions
    #       - #hashtags
    # And grab everything from the trailing slash up to but not including the next
    # whitespace character or end of text
    r"(?:https?://|#|@).*?(?=(?:\s|$))"
)

plain_text_tokenizer = regex.compile(
    # Note this handles 'quoted' words a little weirdly: 'orange' is tokenised
    # as ["orange", "'"] I'd prefer to tokenise this as p"'", "orange", "'"]
    # but the regex library behaves weirdly. So for now phrase search for
    # quoted strings won't work.
    r"\b\p{Word_Break=WSegSpace}*'?",
    flags=regex.WORD | regex.UNICODE | regex.V1,
)


def social_media_tokens(entry):
    cleaned = social_media_cleaner.sub("", entry.lower())
    return [token for token in word_tokenizer.split(cleaned) if token.strip()]


def tokens(entry):
    cleaned = entry.lower()
    return [token for token in word_tokenizer.split(cleaned) if token.strip()]


class HTMLTextLines(HTMLParser):
    """
    Parser for extracting the text from the data elements of given HTML.

    """

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.lines = []

    def handle_data(self, d):
        self.lines.append(d)

    def get_lines(self):
        return self.lines


def text_from_html(html):
    """
    Returns a list of the text contained in the data elements of the given HTML string.

    """
    s = HTMLTextLines()
    s.feed(html)
    lines = s.get_lines()
    s.close()
    return lines


def bstm(matching, bitslice, top_k):
    """
    Applies the bit sliced term matching procedure.

    """

    if len(matching) <= top_k:
        return matching

    e = matching.copy()
    g = BitMap()

    for i in reversed(range(len(bitslice))):
        x = g | (e & bitslice[i])
        n = len(x)

        if n > top_k:
            e &= bitslice[i]
        elif n < top_k:
            g = x
            e -= bitslice[i]
        else:
            e &= bitslice[i]
            break

    return g | e
