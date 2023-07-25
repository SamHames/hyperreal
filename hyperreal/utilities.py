import heapq
from html.parser import HTMLParser
from io import StringIO
import itertools
import math
import operator

from pyroaring import BitMap
import regex


# Used to transform left/right sided curly quotes into their straight quote
# equivalents. This is particularly important on social media as the IOS
# keyboard defaults to curly quotes, unlike Android and Desktops.
curly_quote_translator = str.maketrans("‘’“”", "''\"\"")

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
    cleaned = social_media_cleaner.sub(
        "", entry.translate(curly_quote_translator).lower()
    )
    tokens = [token for token in word_tokenizer.split(cleaned) if token.strip()]
    # This is a terminator token, to make sure that collocations aren't
    # identified across textual boundaries.
    tokens.append(None)
    return tokens


def tokens(entry):
    cleaned = entry.lower()
    tokens = [token for token in word_tokenizer.split(cleaned) if token.strip()]
    # This is a terminator token, to make sure that collocations aren't
    # identified across textual boundaries.
    tokens.append(None)
    return tokens


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


def compute_bitslice(bitmaps):
    """Compute the bitslice of the given bitmaps."""
    matching = BitMap()
    bitslice = [BitMap()]

    for bitmap in bitmaps:
        doc_ids = bitmap

        matching |= doc_ids

        for i, bs in enumerate(bitslice):
            carry = bs & doc_ids
            bs ^= doc_ids
            doc_ids = carry
            if not carry:
                break

        if carry:
            bitslice.append(carry)

    return matching, bitslice


def weight_bitslice(bitslice):
    """
    Return an iterator of weights for each document in the bitslice.

    """

    # Can't be a lambda as we need to capture the layer eagerly
    def slice_gen(bm, layer):
        weight = 2**layer
        for doc_id in bm:
            yield (doc_id, weight)

    # Returns them in merged order
    ordered = heapq.merge(*(slice_gen(bm, i) for i, bm in enumerate(bitslice)))

    grouped = itertools.groupby(ordered, key=operator.itemgetter(0))

    for key, group in grouped:
        yield key, sum(g[1] for g in group)


def approximate_positions_with_sentinels(values, position_window_size):
    """
    Turn a Sequence of values into a generator of (position_bucket, value).

    This function handles None as sentinel values to enforce position breaks
    and the rounding of position values.

    Examples:

    This is position_window_size = 1, which is the exact position case.

    >>> list(approximate_positions_with_sentinels(['the', 'cat', 'sat'], 1))
    [(0, 'the'), (1, 'cat'), (2, 'sat')]

    Approximate positions (position_window_size > 1) accumulate multiple
    values into the same position as in the following examples for 2, 3:

    >>> list(approximate_positions_with_sentinels(['the', 'cat', 'sat'], 2))
    [(0, 'the'), (0, 'cat'), (1, 'sat')]

    >>> values = "the cat sat on the mat".split()
    >>> list(approximate_positions_with_sentinels(values, 3))
    [(0, 'the'), (0, 'cat'), (0, 'sat'), (1, 'on'), (1, 'the'), (1, 'mat')]

    'None' values are sentinel markers to terminate the position bucket
    earlier. This can be used to control breaking more finely in a stream
    of tokens than would otherwise be possible.

    >>> values.insert(3, None)
    >>> list(approximate_positions_with_sentinels(values, 2))
    [(0, 'the'), (0, 'cat'), (1, 'sat'), (2, 'on'), (2, 'the'), (3, 'mat')]

    Lastly a negative position value only changes position at sentinel None's.
    This allows you to control everything about the procedure.

    >>> list(approximate_positions_with_sentinels(values, -2))
    [(0, 'the'), (0, 'cat'), (0, 'sat'), (1, 'on'), (1, 'the'), (1, 'mat')]

    """

    current_position = 0
    current_size = 0

    for value in values:
        if value is None:
            # If we've placed anything in the current bucket, we
            # need to advance the position counter - otherwise
            # we can just keep advancing. This turns consecutive
            # sentinels into a single sentinel.
            if current_size:
                current_position += 1
                current_size = 0

            continue

        current_size += 1

        yield (current_position, value)

        if current_size == position_window_size:
            current_position += 1
            current_size = 0
