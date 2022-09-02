import sqlite3

import pyroaring


def dict_factory(cursor, row):
    # Based on the Python standard library docs:
    # https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.row_factory
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def connect_sqlite(db_path, row_factory=None):
    """
    A standardised initialisation approach for SQLite.

    This connect function:

    - sets isolation_level to None, so DBAPI does not manage transactions
    - connects to the database so column type declaration parsing is active

    Note that this function setups up global adapters on the sqlite module,
    so use with care.

    """

    conn = sqlite3.connect(
        db_path, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None
    )

    conn.create_aggregate("roaring_union", 1, RoaringUnion)

    if row_factory:
        conn.row_factory = row_factory

    return conn


def save_bitmap(bm):
    return bm.serialize()


def load_bitmap(bm_bytes):
    return pyroaring.BitMap.deserialize(bm_bytes)


sqlite3.register_adapter(pyroaring.BitMap, save_bitmap)
sqlite3.register_adapter(pyroaring.FrozenBitMap, save_bitmap)
sqlite3.register_converter("roaring_bitmap", load_bitmap)


class RoaringUnion:
    def __init__(self):
        self.bitmap = pyroaring.BitMap()

    def step(self, bitmap):
        self.bitmap |= pyroaring.BitMap.deserialize(bitmap)

    def finalize(self):
        return self.bitmap.serialize()
