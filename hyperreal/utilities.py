import regex


def dict_factory(cursor, row):
    # Based on the Python standard library docs:
    # https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.row_factory
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


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


def social_media_tokens(entry):
    cleaned = social_media_cleaner.sub("", entry.lower())
    return [token for token in word_tokenizer.split(cleaned) if token]
