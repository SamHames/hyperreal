"""
_schema.py: used for managing the schema of the index database.

Migrations are handled as follows:

1. The CURRENT_SCHEMA script always represents the current schema of the
database, associated with CURRENT_SCHEMA_VERSION.
2. The migrations dict maps earlier application versions to:
    - a sequence of statements to be run before applying CURRENT_SCHEMA
    - a sequence of statements to be run after applying CURRENT_SCHEMA

"""

# The application ID uses SQLite's pragma application_id to quickly identify index
# databases from everything else.
MAGIC_APPLICATION_ID = 715973853
CURRENT_SCHEMA_VERSION = 8

CURRENT_SCHEMA = f"""
    create table if not exists doc_key (
        doc_id integer primary key,
        doc_key unique
    );

    --------
    create table if not exists inverted_index (
        feature_id integer primary key,
        field text not null,
        value not null,
        docs_count integer not null,
        doc_ids roaring_bitmap not null,
        unique (field, value)
    );

    --------
    create table if not exists field_summary (
        field text primary key,
        distinct_values integer,
        min_value,
        max_value
    );

    --------
    create table if not exists skipgram_count (
        feature_id_a integer references inverted_index(feature_id) on delete cascade,
        feature_id_b integer references inverted_index(feature_id) on delete cascade,
        distance integer,
        docs_count integer,
        primary key (feature_id_a, feature_id_b, distance)
    ) without rowid;

    --------
    create index if not exists docs_counts on inverted_index(docs_count);
    --------
    create index if not exists field_docs_counts on inverted_index(field, docs_count);

    --------
    -- The summary table for clusters, including the loose hierarchy
    -- and the materialised results of the query and document counts.
    create table if not exists cluster (
        cluster_id integer primary key,
        feature_count integer default 0,
        -- Length of doc_ids/number of docs retrieved by the union
        docs_count integer default 0,
        -- Sum of the length of the individual feature queries that form the union
        weight integer default 0,
        doc_ids roaring_bitmap,
        -- Whether the cluster is pinned, and should be excluded from automatic clustering.
        pinned bool default 0
    );

    --------
    create table if not exists feature_cluster (
        feature_id integer primary key references inverted_index(feature_id) on delete cascade,
        cluster_id integer references cluster(cluster_id) on delete cascade,
        docs_count integer,
        -- Whether the feature is pinned, and shouldn't be considered for moving.
        pinned bool default 0
    );

    --------
    create index if not exists cluster_features on feature_cluster(
        cluster_id,
        docs_count
    );

    --------
    -- Used to track when clusters have changed, to mark that housekeeping
    -- functions need to run. Previously a more complex set of triggers was used,
    -- but that leads to performance issues on models with large numbers of
    -- features as triggers are only executed per row in sqlite.
    create table if not exists changed_cluster (
        cluster_id integer primary key references cluster on delete cascade
    );

    --------
    -- Cleanup old triggers if they're still around - these are now updated in
    -- the application layer
    drop trigger if exists mark_cluster_changed;
    --------
    drop trigger if exists ensure_cluster;
    --------
    drop trigger if exists delete_feature_cluster;
    --------
    drop trigger if exists ensure_cluster_update;
    --------
    drop trigger if exists update_cluster_feature_counts;

    --------
    create trigger if not exists insert_feature_checks before insert on feature_cluster
        begin
            -- Make sure the cluster exists in the tracking table for foreign key relationships
            insert or ignore into cluster(cluster_id) values (new.cluster_id);
            -- Make sure that the new cluster is marked as changed so it can be summarised
            insert or ignore into changed_cluster(cluster_id) values (new.cluster_id);
        end;

    --------
    create trigger if not exists update_feature_checks before update on feature_cluster
        when old.cluster_id != new.cluster_id
        begin
            -- Make sure the new cluster exists in the tracking table for foreign
            -- key relationships
            insert or ignore into cluster(cluster_id) values (new.cluster_id);

            -- Make sure that the new and old clusters are marked as changed
            -- so it can be summarised
            insert or ignore into changed_cluster(cluster_id) 
                values (new.cluster_id), (old.cluster_id);
        end;

    --------
    create trigger if not exists delete_feature_checks before delete on feature_cluster
        begin
            -- Make sure that the new and old clusters are marked as changed
            -- so it can be summarised
            insert or ignore into changed_cluster(cluster_id) 
                values (old.cluster_id);
        end;

    --------
    pragma user_version = { CURRENT_SCHEMA_VERSION };
    --------
    pragma application_id = { MAGIC_APPLICATION_ID };
"""

migrations = {
    4: (
        [
            "alter table cluster add column docs_count integer default 0",
            "alter table cluster add column weight integer default 0",
            "alter table cluster add column doc_ids roaring_bitmap",
        ],
        [
            "insert into changed_cluster select cluster_id from cluster",
        ],
    ),
    5: ([], []),
    6: ([], []),
    7: (
        [
            "alter table cluster add column pinned bool default 0",
            "alter table feature_cluster add column pinned bool default 0",
        ],
        [],
    ),
}


class MigrationError(ValueError):
    pass


class IndexVersionMismatch(ValueError):
    pass


def migrate(db):
    """
    Migrate the database to the current version of the index schema.

    Returns True if a migration operation ran, False otherwise.

    """

    db_version = list(db.execute("pragma user_version"))[0][0]

    if db_version == 0:
        # Check that this is a database with no tables, and error if not - don't
        # want to create these tables on top of an unrelated database.
        table_count = list(db.execute("select count(*) from sqlite_master"))[0][0]

        if table_count > 0:
            raise MigrationError(
                f"Database at '{db_path}' is not empty, and cannot be used as an index."
            )
        else:
            db.execute("begin")

            for statement in CURRENT_SCHEMA.split("--------"):
                db.execute(statement)

            db.execute("commit")

        # Note that an initialisation is treated as if the index already existed
        # as in this case there won't be any further action to take.
        return False

    elif db_version == CURRENT_SCHEMA_VERSION:
        return False

    elif db_version in migrations:
        db.execute("begin")

        m = sorted(migrations.items())

        # Run premigration steps
        for version, migration in m:
            if version >= db_version:
                for statement in migration[0]:
                    db.execute(statement)

        # Run the schema script
        for statement in CURRENT_SCHEMA.split("--------"):
            db.execute(statement)

        # Run postmigration steps
        for version, migration in m:
            if version >= db_version:
                for statement in migration[1]:
                    db.execute(statement)

        db.execute("commit")

        return True

    else:
        raise MigrationError(f"Unknown database schema version '{db_version}'")
