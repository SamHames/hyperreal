from html.parser import HTMLParser
from io import StringIO
import math

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
    tokens = [token for token in word_tokenizer.split(cleaned) if token.strip()]
    tokens.append(None)
    return tokens


def tokens(entry):
    cleaned = entry.lower()
    tokens = [token for token in word_tokenizer.split(cleaned) if token.strip()]
    tokens.append(None)
    return tokens


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


def long_distance_bigrams(items, max_window_size, include_items=None):
    """
    Return the set of pairs of items from the stream that occur within
    `max_window_size` of each other.

    The returned value is a set of (item_a, item_b, offset) triples,
    representing the items and the ordinal distance between them in the
    stream.

    The stream can be a sequence of any type of items, except None, which is
    used as a sentinel to indicate boundaries that the window should not
    cross. For example:

    >>> sorted(
    ...     long_distance_bigrams(
    ...         ['a', 'b', 'c', None, 'd', 'e'],
    ...         2
    ...     )
    ... )
    [('a', 'b', 1), ('a', 'c', 2), ('b', 'c', 1), ('d', 'e', 1)]

    `include_items` is a set of items to consider - if provided values
    not in this set will not be generated as pairs. This is primarily
    useful to prefilter items and limit the size of the set returned.

    >>> sorted(
    ...     long_distance_bigrams(
    ...         ['a', 'b', 'c', None, 'd', 'e'],
    ...         2,
    ...         include_items={'a', 'b', 'd', 'e'}
    ...     )
    ... )
    [('a', 'b', 1), ('d', 'e', 1)]

    """

    bigrams = set()

    if include_items is not None:
        position_stream = [
            (i, item)
            for i, item in enumerate(items)
            if (item is None or item in include_items)
        ]
    else:
        position_stream = list(enumerate(items))

    for i, (position_a, item_a) in enumerate(position_stream):

        if item_a is None:
            continue

        for position_b, item_b in position_stream[i + 1 : i + 1 + max_window_size]:

            gap = position_b - position_a

            if item_b is None or (gap > max_window_size):
                break

            bigrams.add((item_a, item_b, gap))

    return bigrams


def round_datetime(datetime):

    minute = datetime.replace(second=0, microsecond=0)
    hour = minute.replace(minute=0)
    day = hour.replace(hour=0)
    month = day.replace(day=1)
    year = month.replace(month=1)

    return {"minute": minute, "hour": hour, "day": day, "month": month, "year": year}
