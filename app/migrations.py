"""Schema migrations, versioned by `PRAGMA user_version`.

No migration framework. This is one SQLite file used by one office, and Alembic
would be more machinery than the problem deserves. What it does need is to be
**safe to run on every startup** and safe to run against a v1 database that
already holds real jobs and real spend.

Two properties make that true:

  * Each step is guarded by `user_version`, so it runs exactly once.
  * Each step is also written to be individually re-runnable — `IF NOT EXISTS`
    everywhere, and column additions that check first. If a migration is
    interrupted halfway (power cut on an office PC is not hypothetical), the
    next start completes it rather than failing on "duplicate column name".

Migrations run inside a transaction, and `user_version` is bumped inside the
same transaction, so a crash leaves the database on the previous version with
none of the step applied.

**Never edit a migration that has shipped.** Add a new one. The version number
is the only record of what a given database has already had done to it.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable

log = logging.getLogger(__name__)

Migration = Callable[[sqlite3.Connection], None]


# --- helpers ----------------------------------------------------------------


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _add_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """ALTER TABLE ... ADD COLUMN, but only if it is not already there.

    SQLite has no `ADD COLUMN IF NOT EXISTS`, and a re-run would otherwise die
    with "duplicate column name". Checking first is what makes a half-applied
    migration recoverable.

    A column added this way cannot carry a non-NULL default, which is why every
    column below is nullable. That is fine: NULL here means "belongs to a v1 row
    that predates accounts", and the queries are written to expect it.
    """
    if column not in _columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _exec(conn: sqlite3.Connection, *statements: str) -> None:
    for sql in statements:
        conn.execute(sql)


# --- 1: the v1 baseline -----------------------------------------------------


def _m001_baseline(conn: sqlite3.Connection) -> None:
    """The jobs table as v1 shipped it.

    Present so a fresh database and an existing one converge on the same path.
    An existing v1 database already has this; `IF NOT EXISTS` makes that a no-op.
    """
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            idempotency_key   TEXT NOT NULL UNIQUE,
            username          TEXT NOT NULL,
            task              TEXT NOT NULL,
            model_id          TEXT NOT NULL,
            request_id        TEXT,
            status            TEXT NOT NULL,
            est_usd           REAL NOT NULL DEFAULT 0,
            actual_usd        REAL,
            params_json       TEXT,
            input_path        TEXT,
            output_path       TEXT,
            error             TEXT,
            created_at        TEXT NOT NULL,
            updated_at        TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON jobs(username, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)",
    )


# --- 2: accounts ------------------------------------------------------------


def _m002_accounts(conn: sqlite3.Connection) -> None:
    """Users and sessions.

    `sessions.token_hash` is the primary key and the raw token is never stored.
    The cookie holds the token; the database holds its SHA-256. Someone who
    walks off with `render_studio.db` gets no usable sessions out of it.

    `budget_monthly_usd` is nullable on purpose: NULL means "use the office
    default from .env", so raising the default raises it for everyone who has
    not been given a specific figure.
    """
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS users (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            email              TEXT NOT NULL UNIQUE,
            name               TEXT NOT NULL,
            password_hash      TEXT NOT NULL,
            status             TEXT NOT NULL DEFAULT 'pending',
            is_admin           INTEGER NOT NULL DEFAULT 0,
            budget_monthly_usd REAL,
            created_at         TEXT NOT NULL,
            approved_at        TEXT,
            approved_by        INTEGER REFERENCES users(id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash  TEXT PRIMARY KEY,
            user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at  TEXT NOT NULL,
            expires_at  TEXT NOT NULL,
            revoked_at  TEXT,
            user_agent  TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_users_status ON users(status)",
    )


# --- 3: projects, assets, and the many-to-many that makes multi-reference work


def _m003_projects_and_assets(conn: sqlite3.Connection) -> None:
    """Architects work in projects, and a render has several inputs.

    `assets.sha256` is the content hash. Uploading the same massing view for the
    sixth render should not upload it to fal.ai a sixth time, so before any
    upload we look for an existing row with the same hash that already has a
    `fal_url` and reuse it. `fal_url_at` records when that URL was obtained —
    fal's CDN retention is documented ambiguously, so anything relying on an old
    URL needs to know how old it is.

    `job_inputs` is keyed on (job_id, position) rather than (job_id, asset_id):
    the same image can legitimately appear twice in one request in two different
    roles, and position is what the prompt text refers to as "image 2".
    """
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS projects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id    INTEGER REFERENCES users(id),
            name        TEXT NOT NULL,
            client      TEXT,
            archived    INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS assets (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id   INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            uploaded_by  INTEGER REFERENCES users(id),
            sha256       TEXT NOT NULL,
            local_path   TEXT,
            fal_url      TEXT,
            fal_url_at   TEXT,
            width        INTEGER,
            height       INTEGER,
            kind         TEXT NOT NULL DEFAULT 'upload',
            created_at   TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_assets_sha ON assets(sha256)",
        "CREATE INDEX IF NOT EXISTS idx_assets_project ON assets(project_id, created_at)",
        """
        CREATE TABLE IF NOT EXISTS job_inputs (
            job_id    INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            asset_id  INTEGER NOT NULL REFERENCES assets(id),
            role      TEXT NOT NULL,
            position  INTEGER NOT NULL,
            PRIMARY KEY (job_id, position)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_job_inputs_asset ON job_inputs(asset_id)",
    )


# --- 4: lineage on jobs -----------------------------------------------------


def _m004_jobs_lineage(conn: sqlite3.Connection) -> None:
    """The edit tree, plus the columns that make editing cheap.

    `output_url` is the one that earns its keep. Continuing from a finished
    render means handing fal.ai an image it is already hosting — with the URL
    cached here, an edit costs no download, no re-upload and no wait.

    `root_job_id` is denormalised deliberately. Walking a parent chain in SQLite
    means either a recursive CTE per render or N queries; storing the root makes
    a whole lineage one indexed lookup, and a render's root never changes.

    `username` is kept. v1 rows have no user_id and their spend still has to
    count, so the ledger reads both — see db.spend_this_month().
    """
    _add_column(conn, "jobs", "user_id", "INTEGER REFERENCES users(id)")
    _add_column(conn, "jobs", "project_id", "INTEGER REFERENCES projects(id)")
    _add_column(conn, "jobs", "parent_job_id", "INTEGER REFERENCES jobs(id)")
    _add_column(conn, "jobs", "root_job_id", "INTEGER REFERENCES jobs(id)")
    _add_column(conn, "jobs", "output_url", "TEXT")
    _add_column(conn, "jobs", "output_seed", "INTEGER")
    _add_column(conn, "jobs", "engine", "TEXT")
    _add_column(conn, "jobs", "mask_asset_id", "INTEGER REFERENCES assets(id)")
    _exec(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_jobs_owner ON jobs(user_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_root ON jobs(root_job_id)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_parent ON jobs(parent_job_id)",
    )
    # Every pre-existing job is the root of its own (single-node) lineage.
    conn.execute("UPDATE jobs SET root_job_id = id WHERE root_job_id IS NULL")


# --- 5: prompt rewriting ----------------------------------------------------


def _m005_prompt_revisions(conn: sqlite3.Connection) -> None:
    """Every rewrite, what it cost, and whether the architect took it.

    `accepted` is the only way to find out in three months whether the rewriter
    is earning its keep. A tool that quietly rewrites prompts and never records
    whether anyone kept the result cannot be evaluated, only believed in.

    `job_id` is nullable: a rewrite happens before submission and may never
    become a job at all.
    """
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS prompt_revisions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id           INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
            user_id          INTEGER REFERENCES users(id),
            task             TEXT NOT NULL,
            engine           TEXT,
            raw_prompt       TEXT NOT NULL,
            enhanced_prompt  TEXT,
            warnings_json    TEXT,
            provider         TEXT,
            model            TEXT,
            accepted         INTEGER NOT NULL DEFAULT 0,
            cost_usd         REAL NOT NULL DEFAULT 0,
            created_at       TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_prompt_rev_job ON prompt_revisions(job_id)",
        "CREATE INDEX IF NOT EXISTS idx_prompt_rev_created ON prompt_revisions(created_at)",
    )


# --- 6: runtime controls ----------------------------------------------------


def _m006_app_settings(conn: sqlite3.Connection) -> None:
    """The handful of settings an administrator can change without a restart.

    Everything else stays in `.env`, where a value that decides how the process
    boots belongs. These two are different: the kill switch and the office cap
    are what you reach for *during* an incident, and "restart the app" is not an
    answer when the reason you want the kill switch is that money is leaving.

    Key/value with the stored string parsed at read time, rather than a column
    per setting — a new runtime control should not need a migration.

    A row is absent until an admin sets one, and absent means "use the .env
    value". So the file stays the default and the database only records
    deliberate overrides.
    """
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key         TEXT PRIMARY KEY,
            value       TEXT,
            updated_at  TEXT NOT NULL,
            updated_by  INTEGER REFERENCES users(id)
        )
        """,
    )


def _m007_depth_source(conn: sqlite3.Connection) -> None:
    """Where a job's depth control image came from: cad, estimated or none.

    NULL means the job predates the column (and, in practice, was conditioned on
    the colour image or a CAD pass - it cannot be told apart after the fact).
    """
    _add_column(conn, "jobs", "depth_source", "TEXT")



def _m008_job_outputs(conn: sqlite3.Connection) -> None:
    """Every image a job produced, not just the first.

    A job asked for four images and kept one. `jobs.output_path/output_url`
    stay as they were (index 0), so every reader written before this table
    still works. The table is the full set, one row per saved image, unique on
    (job_id, idx).

    Backfill: each job that already has an output_path gets its index-0 row, so
    the table is complete from the moment it exists and no reader needs a
    fallback to the jobs columns.

    `retry_of_job_id` links a retry to the failed job it re-sends. `timed_out_at`
    records a job the poller gave up on, so the failure can be told apart from
    one fal.ai reported.
    """
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS job_outputs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id      INTEGER NOT NULL REFERENCES jobs(id),
            idx         INTEGER NOT NULL,
            path        TEXT,
            url         TEXT,
            seed        INTEGER,
            created_at  TEXT NOT NULL,
            UNIQUE (job_id, idx)
        )
        """,
    )
    _add_column(conn, "jobs", "retry_of_job_id", "INTEGER REFERENCES jobs(id)")
    _add_column(conn, "jobs", "timed_out_at", "TEXT")
    _exec(
        conn,
        """
        INSERT OR IGNORE INTO job_outputs (job_id, idx, path, url, seed, created_at)
        SELECT id, 0, output_path, output_url, output_seed, updated_at
          FROM jobs
         WHERE output_path IS NOT NULL
        """,
    )


def _m009_retry_index(conn: sqlite3.Connection) -> None:
    """Index the lookup behind "which retries came from this job".

    `retries_of` filters jobs on retry_of_job_id on every Retry and every job
    listing, and nothing indexed the column. A new index only; no data changes.
    """
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_retry_of ON jobs(retry_of_job_id)")


# Order is the contract. Append only.
MIGRATIONS: list[tuple[int, str, Migration]] = [
    (1, "baseline_jobs", _m001_baseline),
    (2, "accounts", _m002_accounts),
    (3, "projects_and_assets", _m003_projects_and_assets),
    (4, "jobs_lineage", _m004_jobs_lineage),
    (5, "prompt_revisions", _m005_prompt_revisions),
    (6, "app_settings", _m006_app_settings),
    (7, "depth_source", _m007_depth_source),
    (8, "job_outputs", _m008_job_outputs),
    (9, "retry_index", _m009_retry_index),
]

SCHEMA_VERSION = MIGRATIONS[-1][0]


def current_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: sqlite3.Connection) -> int:
    """Bring the database up to SCHEMA_VERSION. Returns the version reached.

    Refuses to run against a database from a *newer* build rather than guessing
    what to do with it — that means someone downgraded the app while the data
    moved on, and silently operating on a schema this code does not understand
    is how data gets mangled.
    """
    version = current_version(conn)
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database is at schema version {version}, but this build only knows "
            f"up to {SCHEMA_VERSION}. It was written by a newer version of the "
            "app. Update the app rather than running this one against it."
        )

    for number, name, step in MIGRATIONS:
        if number <= version:
            continue
        log.info("Applying migration %s: %s", number, name)
        # IMMEDIATE, not a plain deferred BEGIN. A deferred transaction takes
        # the write lock at its first write, and if another connection already
        # holds it SQLite returns SQLITE_BUSY *immediately* rather than
        # consulting the busy handler, because waiting there could deadlock.
        # Two processes opening a fresh database at once — the API and a second
        # process on the same machine — therefore had one of them fail on
        # ALTER TABLE instead of queueing behind the other. IMMEDIATE takes the
        # lock up front, where the 30s busy timeout does apply.
        conn.execute("BEGIN IMMEDIATE")
        try:
            step(conn)
            # Not parameterisable — PRAGMA takes a literal. `number` comes from
            # the table above and is an int, so there is nothing to inject.
            conn.execute(f"PRAGMA user_version = {int(number)}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            log.exception("Migration %s (%s) failed and was rolled back", number, name)
            raise
        version = number

    return version
