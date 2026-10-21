"""SQLite storage. Single writer, WAL mode — legitimate for an internal tool.

The `jobs` table doubles as the spend ledger. One source of truth: every row that
was submitted carries its estimated cost, so summing over a month is the spend.
There is no separate ledger to drift out of sync with reality.

Schema lives in `app/migrations.py` and is applied on connect. This module is
queries only — it does not define tables, so there is exactly one place that
describes what the database looks like.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import secrets
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app import migrations
from app.models import AssetKind, ImageRole, UserStatus

log = logging.getLogger(__name__)

# Terminal states never re-poll. Anything else is still in flight.
TERMINAL = {"completed", "failed", "refused", "cancelled"}

SESSION_DAYS = 30


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def month_prefix(when: datetime | None = None) -> str:
    """'2026-08' — used to scope spend queries to the current calendar month."""
    when = when or datetime.now(UTC)
    return when.strftime("%Y-%m")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: the API hands this connection between threads.
    # FastAPI serves requests from a thread pool and the poller runs on its own. SQLite itself is built
    # serialized (sqlite3.threadsafety == 3), and WAL plus the 30s busy timeout
    # cover concurrent access at office scale.
    conn = sqlite3.connect(
        db_path, timeout=30.0, isolation_level=None, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    # Before anything that can block. `timeout=` above already sets this, but
    # only for statements that consult the busy handler at all.
    conn.execute("PRAGMA busy_timeout=30000;")

    # Switching journal modes needs a lock that the busy handler does NOT wait
    # for, so this raised "database is locked" outright when a second process
    # opened the database while the API was inside a write (reproduced
    # 2026-08-27 by connecting during a BEGIN IMMEDIATE). WAL lives in the file
    # header and persists, so once anyone has set it, it is set for everyone —
    # which makes reading first both cheaper and safe.
    if str(conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower() != "wal":
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError:
            log.warning(
                "Could not switch %s to WAL; another connection holds it. "
                "Continuing in the journal mode already set.", db_path
            )

    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    migrations.migrate(conn)
    return conn


# --- Transactions -----------------------------------------------------------

# The connection is autocommit and shared process-wide (api/deps.py), so a
# multi-statement transaction has two problems at once. Across processes, two
# writers can interleave read-then-write; within this process, a second thread
# issuing a write while one thread sits inside BEGIN would silently join that
# thread's transaction and be committed or rolled back with it.
#
# BEGIN IMMEDIATE solves the first — it takes the database's write lock up
# front rather than at first write, so a concurrent writer waits (up to the 30s
# busy timeout) instead of reading stale numbers. This lock solves the second:
# one transaction at a time on the shared handle. Reads outside a transaction
# are unaffected; WAL gives them a snapshot.
_write_lock = threading.RLock()


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Run a block as one all-or-nothing write.

    Re-entrant: a nested call joins the outer transaction rather than opening a
    second one, which SQLite does not have, so a function that needs atomicity
    stays composable with a caller that already opened a transaction.
    """
    with _write_lock:
        if conn.in_transaction:
            yield
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")


class ForeignKeyViolation(Exception):
    """A row pointed at something that does not exist.

    Separate from the duplicate-key case because they mean opposite things to a
    caller: a duplicate idempotency key is success (the work is already paid
    for), a dangling project_id is a bad request. Collapsing both into "return
    None" is what turned `project_id: 424242` into a 500.
    """


def _is_foreign_key_error(exc: sqlite3.IntegrityError) -> bool:
    """Tell a FOREIGN KEY failure from a UNIQUE one.

    `sqlite_errorname` is exact and has been on IntegrityError since 3.11; the
    message check is the fallback for anything that does not carry it.
    """
    name = getattr(exc, "sqlite_errorname", "") or ""
    if name:
        return name == "SQLITE_CONSTRAINT_FOREIGNKEY"
    return "FOREIGN KEY constraint failed" in str(exc)


def _name_missing_reference(
    conn: sqlite3.Connection,
    *,
    project_id: int | None = None,
    user_id: int | None = None,
    parent_job_id: int | None = None,
    asset_id: int | None = None,
) -> str:
    """Which supplied id does not exist, in words the caller can act on.

    SQLite says only "FOREIGN KEY constraint failed" — it does not name the
    column. Re-checking the handful of ids the caller passed is cheap and turns
    an opaque failure into a sentence naming the row that is missing.
    """
    if project_id is not None and get_project(conn, project_id) is None:
        return f"No project {project_id}. It may have been deleted."
    if asset_id is not None and get_asset(conn, asset_id) is None:
        return f"No asset {asset_id}."
    if parent_job_id is not None and get_job(conn, parent_job_id) is None:
        return f"No render {parent_job_id} to continue from."
    if user_id is not None and get_user(conn, user_id) is None:
        return f"No account {user_id}."
    return "That request referred to a record that no longer exists."


# --- Jobs -------------------------------------------------------------------


def reserve_job(
    conn: sqlite3.Connection,
    *,
    idempotency_key: str,
    username: str,
    task: str,
    model_id: str,
    est_usd: float,
    params_json: str,
    input_path: str | None,
    user_id: int | None = None,
    project_id: int | None = None,
    parent_job_id: int | None = None,
    engine: str | None = None,
    mask_asset_id: int | None = None,
    retry_of_job_id: int | None = None,
) -> int | None:
    """Insert a 'reserved' row BEFORE calling fal.ai.

    Reserving first is what makes the budget guard honest: the estimated cost is
    counted against the ceiling from the moment we commit to spending it, not
    after the call returns. If the submit then fails, the row is marked failed
    and its cost is released.

    Returns the job id, or None if this idempotency_key was already used —
    which means a duplicate submission, and we must not spend twice.

    Raises ForeignKeyViolation when the row points at a project, account or
    parent that does not exist. That case used to be swallowed as "duplicate",
    which returned None to a caller that then dereferenced a job it had never
    inserted — a bad project_id arrived at the browser as a 500.
    """
    now = utcnow()
    # A child inherits its parent's root, so a whole lineage is one indexed
    # query rather than a walk up the chain.
    root_job_id: int | None = None
    if parent_job_id is not None:
        parent = get_job(conn, parent_job_id)
        if parent is None:
            raise ValueError(f"Cannot continue from job {parent_job_id}: no such job.")
        root_job_id = int(parent["root_job_id"] or parent["id"])

    # The INSERT and the root_job_id UPDATE are one write: a crash between them
    # would leave a reserved job with no lineage, which `lineage()` reads as an
    # empty tree and the version view as a render that never happened.
    with transaction(conn):
        try:
            cur = conn.execute(
                """
                INSERT INTO jobs (idempotency_key, username, task, model_id, status,
                                  est_usd, params_json, input_path, created_at, updated_at,
                                  user_id, project_id, parent_job_id, root_job_id,
                                  engine, mask_asset_id, retry_of_job_id)
                VALUES (?, ?, ?, ?, 'reserved', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (idempotency_key, username, task, model_id, est_usd,
                 params_json, input_path, now, now,
                 user_id, project_id, parent_job_id, root_job_id,
                 engine, mask_asset_id, retry_of_job_id),
            )
        except sqlite3.IntegrityError as exc:
            if _is_foreign_key_error(exc):
                raise ForeignKeyViolation(
                    _name_missing_reference(
                        conn, project_id=project_id, user_id=user_id,
                        parent_job_id=parent_job_id, asset_id=mask_asset_id,
                    )
                ) from exc
            return None

        job_id = int(cur.lastrowid)
        if root_job_id is None:
            # An original render is the root of its own lineage.
            conn.execute(
                "UPDATE jobs SET root_job_id = ? WHERE id = ?", (job_id, job_id)
            )
    return job_id


@dataclass(frozen=True)
class Reservation:
    """The outcome of an atomic spend-check-and-reserve.

    Deliberately data rather than an exception: `app/budget.py` owns the wording
    of a refusal and this module owns the write, and reversing that would mean
    db importing budget importing db.
    """

    job_id: int | None = None
    duplicate_of: int | None = None
    # None when the job was reserved; "per_user" or "monthly" when a cap fired.
    refused: str | None = None
    month_total_usd: float = 0.0
    month_user_usd: float = 0.0

    @property
    def duplicate(self) -> bool:
        return self.duplicate_of is not None


def reserve_job_within_budget(
    conn: sqlite3.Connection,
    *,
    monthly_cap_usd: float,
    per_user_cap_usd: float,
    est_usd: float,
    username: str,
    user_id: int | None = None,
    **reserve_kwargs: Any,
) -> Reservation:
    """Check the month's spend and write the reservation as one transaction.

    This is the fix for the overshoot: reading a SUM() and inserting the row
    that changes it were two separate statements with nothing between them, so
    N simultaneous submissions all read the same pre-spend total and all passed.
    The overshoot was the per-job cap times the concurrency — five architects
    clicking together could commit five jobs against the last dollar.

    BEGIN IMMEDIATE takes the write lock before the SUM is read, so a second
    submission blocks until the first has either inserted its row or rolled
    back. The number the second one reads is therefore the number after the
    first one spent.

    Callers pass the remaining `reserve_job` arguments through; the cheap gates
    that need no consistency — the kill switch, the per-job cap — stay in
    `budget.check()` where they can refuse before any lock is taken.
    """
    with transaction(conn):
        month_total = committed_spend_this_month(conn)
        month_user = spend_this_month(conn, username=username, user_id=user_id)

        if month_user + est_usd > per_user_cap_usd:
            return Reservation(
                refused="per_user",
                month_total_usd=month_total,
                month_user_usd=month_user,
            )
        if month_total + est_usd > monthly_cap_usd:
            return Reservation(
                refused="monthly",
                month_total_usd=month_total,
                month_user_usd=month_user,
            )

        job_id = reserve_job(
            conn,
            est_usd=est_usd,
            username=username,
            user_id=user_id,
            **reserve_kwargs,
        )
        if job_id is None:
            existing = find_by_key(conn, str(reserve_kwargs["idempotency_key"]))
            return Reservation(
                duplicate_of=int(existing["id"]) if existing else None,
                month_total_usd=month_total,
                month_user_usd=month_user,
            )
        return Reservation(
            job_id=job_id,
            month_total_usd=month_total,
            month_user_usd=month_user,
        )


def update_job(conn: sqlite3.Connection, job_id: int, **fields: Any) -> None:
    if not fields:
        return
    allowed = {
        "request_id", "status", "actual_usd", "output_path", "error", "est_usd",
        "output_url", "output_seed", "timed_out_at",
    }
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"update_job: refusing to write unknown columns {bad}")
    fields["updated_at"] = utcnow()
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", (*fields.values(), job_id))


def get_job(conn: sqlite3.Connection, job_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def set_job_depth_source(conn: sqlite3.Connection, job_id: int, depth_source: str) -> None:
    """Record whether the depth control was `cad`, `estimated` or `none`."""
    with transaction(conn):
        conn.execute("UPDATE jobs SET depth_source = ? WHERE id = ?", (depth_source, job_id))


def add_job_output(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    idx: int,
    path: str | None,
    url: str | None,
    seed: int | None,
) -> None:
    """Record one image a job produced. Idempotent on (job_id, idx).

    A re-poll that lands on the same index replaces the row rather than failing
    on the unique key, so a crash between two writes cannot wedge a job.
    """
    conn.execute(
        """
        INSERT INTO job_outputs (job_id, idx, path, url, seed, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (job_id, idx) DO UPDATE SET
            path = excluded.path, url = excluded.url, seed = excluded.seed
        """,
        (job_id, idx, path, url, seed, utcnow()),
    )


def list_job_outputs(conn: sqlite3.Connection, job_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM job_outputs WHERE job_id = ? ORDER BY idx ASC", (job_id,)
    ).fetchall()


def get_job_output(conn: sqlite3.Connection, job_id: int, idx: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM job_outputs WHERE job_id = ? AND idx = ?", (job_id, idx)
    ).fetchone()


def job_outputs_for(
    conn: sqlite3.Connection, job_ids: Sequence[int]
) -> dict[int, list[sqlite3.Row]]:
    """Outputs for many jobs in one query, keyed by job id.

    The jobs list calls this once per page. Asking per job would be one query
    for each of up to 200 rows on every poll of the screen.
    """
    if not job_ids:
        return {}
    placeholders = ", ".join("?" for _ in job_ids)
    rows = conn.execute(
        f"SELECT * FROM job_outputs WHERE job_id IN ({placeholders}) ORDER BY job_id, idx",
        tuple(int(j) for j in job_ids),
    ).fetchall()
    grouped: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(int(row["job_id"]), []).append(row)
    return grouped


def find_by_key(conn: sqlite3.Connection, idempotency_key: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
    ).fetchone()


def recent_jobs(
    conn: sqlite3.Connection,
    limit: int = 50,
    *,
    project_id: int | None = None,
    user_id: int | None = None,
) -> list[sqlite3.Row]:
    sql = "SELECT * FROM jobs WHERE 1=1"
    params: list[Any] = []
    if project_id is not None:
        sql += " AND project_id = ?"
        params.append(project_id)
    if user_id is not None:
        sql += " AND user_id = ?"
        params.append(user_id)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return conn.execute(sql, params).fetchall()


def activity_jobs(
    conn: sqlite3.Connection,
    limit: int,
    *,
    before_id: int | None = None,
    user_id: int | None = None,
    status: str | None = None,
) -> list[sqlite3.Row]:
    """Jobs from every user, newest first, with the name of whoever made each.

    Pages by id rather than by offset: new renders arrive while the administrator
    is reading, and an offset would show the same row twice after each arrival.
    The users join is a LEFT one because rows written before accounts existed
    have no user_id and must still appear.
    """
    sql = (
        "SELECT jobs.*, users.name AS user_name "
        "FROM jobs LEFT JOIN users ON users.id = jobs.user_id WHERE 1=1"
    )
    # A person's pre-account jobs (user_id NULL, their email as username) belong
    # to them, the same rule spend_this_month applies.
    params: list[Any] = []
    if before_id is not None:
        sql += " AND jobs.id < ?"
        params.append(before_id)
    if user_id is not None:
        sql += (
            " AND (jobs.user_id = ? OR (jobs.user_id IS NULL AND jobs.username ="
            " (SELECT email FROM users WHERE id = ?)))"
        )
        params += [user_id, user_id]
    if status is not None:
        sql += " AND jobs.status = ?"
        params.append(status)
    sql += " ORDER BY jobs.id DESC LIMIT ?"
    params.append(limit)
    return conn.execute(sql, params).fetchall()


def activity_totals(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """One line per person: how much they have rendered and what it cost this month.

    Everyone with an account is listed, even with no jobs, so the administrator
    can see who has never used the tool. Jobs from before accounts existed have
    no user_id; they are grouped by their free-text username instead, so that
    history is not dropped from the totals. The month boundary is the same
    calendar-month prefix spend_this_month uses, and spend is taken from it
    directly so the two numbers can never disagree.
    """
    prefix = month_prefix() + "%"
    out: list[dict[str, Any]] = []

    for u in conn.execute(
        """
        SELECT users.id, users.email, users.name, users.status,
               COUNT(jobs.id) AS jobs_total,
               COALESCE(SUM(CASE WHEN jobs.created_at LIKE ? THEN 1 ELSE 0 END), 0)
                   AS jobs_month,
               MAX(jobs.created_at) AS last_active
        FROM users LEFT JOIN jobs
          ON (jobs.user_id = users.id
              OR (jobs.user_id IS NULL AND jobs.username = users.email))
        GROUP BY users.id
        ORDER BY jobs_month DESC, users.name
        """,
        (prefix,),
    ).fetchall():
        out.append({
            "user_id": int(u["id"]),
            "email": str(u["email"]),
            "name": str(u["name"] or ""),
            "status": str(u["status"]),
            "jobs_total": int(u["jobs_total"]),
            "jobs_month": int(u["jobs_month"]),
            "spent_month_usd": spend_this_month(
                conn, username=str(u["email"]), user_id=int(u["id"])
            ),
            "last_active": u["last_active"],
        })

    for o in conn.execute(
        """
        SELECT username, COUNT(*) AS jobs_total,
               COALESCE(SUM(CASE WHEN created_at LIKE ? THEN 1 ELSE 0 END), 0)
                   AS jobs_month,
               MAX(created_at) AS last_active
        FROM jobs WHERE user_id IS NULL
          AND username NOT IN (SELECT email FROM users)
        GROUP BY username
        """,
        (prefix,),
    ).fetchall():
        out.append({
            "user_id": None,
            "email": str(o["username"]),
            "name": "",
            "status": "legacy",
            "jobs_total": int(o["jobs_total"]),
            "jobs_month": int(o["jobs_month"]),
            "spent_month_usd": spend_this_month(conn, username=str(o["username"])),
            "last_active": o["last_active"],
        })
    return out


def pending_jobs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    placeholders = ", ".join("?" for _ in TERMINAL)
    return conn.execute(
        f"SELECT * FROM jobs WHERE status NOT IN ({placeholders}) AND request_id IS NOT NULL",
        tuple(TERMINAL),
    ).fetchall()


def retries_of(conn: sqlite3.Connection, job_id: int) -> list[sqlite3.Row]:
    """Every job sent again from this one, oldest first."""
    return conn.execute(
        "SELECT * FROM jobs WHERE retry_of_job_id = ? ORDER BY id ASC", (job_id,)
    ).fetchall()


def lineage(conn: sqlite3.Connection, job_id: int) -> list[sqlite3.Row]:
    """Every render descended from the same original, oldest first.

    The UI draws this as a tree using parent_job_id; the query returns it flat
    because assembling a tree from a flat list is trivial and a recursive CTE
    per render is not.
    """
    row = get_job(conn, job_id)
    if row is None:
        return []
    root = int(row["root_job_id"] or row["id"])
    return conn.execute(
        "SELECT * FROM jobs WHERE root_job_id = ? ORDER BY id ASC", (root,)
    ).fetchall()


# --- Spend ------------------------------------------------------------------


def spend_this_month(
    conn: sqlite3.Connection,
    username: str | None = None,
    *,
    user_id: int | None = None,
) -> float:
    """Committed spend for the current month.

    Uses actual_usd where fal.ai reported one, else the estimate. Refused and
    failed jobs are excluded — they never reached the API, so they cost nothing.

    When a user_id is given, rows written before accounts existed are matched by
    their free-text username as well. Without that, activating accounts would
    silently reset everyone's monthly spend to zero and hand the office a second
    full budget in the same month.
    """
    prefix = month_prefix() + "%"
    sql = """
        SELECT COALESCE(SUM(COALESCE(actual_usd, est_usd)), 0.0) AS total
        FROM jobs
        WHERE created_at LIKE ?
          AND status NOT IN ('refused', 'failed')
    """
    params: list[Any] = [prefix]

    if user_id is not None:
        if username is not None:
            sql += " AND (user_id = ? OR (user_id IS NULL AND username = ?))"
            params += [user_id, username]
        else:
            sql += " AND user_id = ?"
            params.append(user_id)
    elif username is not None:
        sql += " AND username = ?"
        params.append(username)

    row = conn.execute(sql, params).fetchone()
    return float(row["total"] or 0.0)


def committed_spend_this_month(
    conn: sqlite3.Connection,
    username: str | None = None,
    *,
    user_id: int | None = None,
) -> float:
    """What the office caps are measured against this month: the renders.

    This is the total the caps are measured against.
    """
    return spend_this_month(conn, username, user_id=user_id)


# --- Users ------------------------------------------------------------------


def create_user(
    conn: sqlite3.Connection,
    *,
    email: str,
    name: str,
    password_hash: str,
    status: UserStatus = UserStatus.PENDING,
    is_admin: bool = False,
) -> int | None:
    """Returns the new user id, or None if that email is already registered.

    None rather than an exception because the caller must respond identically
    either way — telling an anonymous visitor which addresses already have
    accounts is a disclosure nobody needs to make.
    """
    try:
        cur = conn.execute(
            """
            INSERT INTO users (email, name, password_hash, status, is_admin,
                               created_at, approved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (email.strip().lower(), name.strip(), password_hash, str(status),
             int(is_admin), utcnow(),
             utcnow() if status == UserStatus.ACTIVE else None),
        )
        return int(cur.lastrowid)
    except sqlite3.IntegrityError:
        return None


def get_user(conn: sqlite3.Connection, user_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def get_user_by_email(conn: sqlite3.Connection, email: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
    ).fetchone()


def list_users(
    conn: sqlite3.Connection, status: UserStatus | None = None
) -> list[sqlite3.Row]:
    if status is None:
        return conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    return conn.execute(
        "SELECT * FROM users WHERE status = ? ORDER BY created_at DESC", (str(status),)
    ).fetchall()


def set_user_status(
    conn: sqlite3.Connection,
    user_id: int,
    status: UserStatus,
    *,
    approved_by: int | None = None,
) -> None:
    conn.execute(
        """
        UPDATE users
           SET status = ?,
               approved_at = CASE WHEN ? = 'active' AND approved_at IS NULL
                                  THEN ? ELSE approved_at END,
               approved_by = COALESCE(?, approved_by)
         WHERE id = ?
        """,
        (str(status), str(status), utcnow(), approved_by, user_id),
    )


def set_user_budget(
    conn: sqlite3.Connection, user_id: int, budget_usd: float | None
) -> None:
    """NULL means 'use the office default', not 'zero'."""
    conn.execute(
        "UPDATE users SET budget_monthly_usd = ? WHERE id = ?", (budget_usd, user_id)
    )


def count_users(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"])


# --- Sessions ---------------------------------------------------------------


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(
    conn: sqlite3.Connection,
    user_id: int,
    *,
    user_agent: str | None = None,
    days: int = SESSION_DAYS,
) -> str:
    """Mint a session and return the RAW token — the only time it exists.

    Only its hash is stored. A copy of the database therefore yields no usable
    sessions, which matters more than usual here: the database file sits on an
    office PC that several people can reach.

    `days` must come from settings.session_days at the call site — the cookie's
    max_age is set from the same value, and if the two drift the browser holds
    a cookie the database has already expired (or the reverse).
    """
    token = secrets.token_urlsafe(32)
    expires = datetime.now(UTC) + timedelta(days=days)
    conn.execute(
        """
        INSERT INTO sessions (token_hash, user_id, created_at, expires_at, user_agent)
        VALUES (?, ?, ?, ?, ?)
        """,
        (_hash_token(token), user_id, utcnow(), expires.isoformat(), user_agent),
    )
    return token


def session_user(conn: sqlite3.Connection, token: str) -> sqlite3.Row | None:
    """Resolve a raw session token to an ACTIVE user, or None.

    Status is checked here rather than at login only, so suspending someone
    takes effect on their next request instead of whenever their cookie happens
    to expire.
    """
    row = conn.execute(
        """
        SELECT u.* FROM sessions s
          JOIN users u ON u.id = s.user_id
         WHERE s.token_hash = ?
           AND s.revoked_at IS NULL
           AND s.expires_at > ?
           AND u.status = 'active'
        """,
        (_hash_token(token), utcnow()),
    ).fetchone()
    return row


def revoke_session(conn: sqlite3.Connection, token: str) -> None:
    conn.execute(
        "UPDATE sessions SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
        (utcnow(), _hash_token(token)),
    )


def revoke_all_sessions(conn: sqlite3.Connection, user_id: int) -> None:
    """Used when an account is suspended. Signing someone out is not optional."""
    conn.execute(
        "UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
        (utcnow(), user_id),
    )


def purge_expired_sessions(conn: sqlite3.Connection) -> int:
    cur = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (utcnow(),))
    return cur.rowcount or 0


# --- Projects ---------------------------------------------------------------


def create_project(
    conn: sqlite3.Connection, *, owner_id: int | None, name: str, client: str | None = None
) -> int:
    now = utcnow()
    cur = conn.execute(
        """
        INSERT INTO projects (owner_id, name, client, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (owner_id, name.strip(), (client or "").strip() or None, now, now),
    )
    return int(cur.lastrowid)


def get_project(conn: sqlite3.Connection, project_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()


def list_projects(
    conn: sqlite3.Connection, *, include_archived: bool = False
) -> list[sqlite3.Row]:
    sql = "SELECT * FROM projects"
    if not include_archived:
        sql += " WHERE archived = 0"
    sql += " ORDER BY updated_at DESC"
    return conn.execute(sql).fetchall()


def touch_project(conn: sqlite3.Connection, project_id: int) -> None:
    conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (utcnow(), project_id))


def set_project_archived(conn: sqlite3.Connection, project_id: int, archived: bool) -> None:
    conn.execute(
        "UPDATE projects SET archived = ?, updated_at = ? WHERE id = ?",
        (int(archived), utcnow(), project_id),
    )


# --- Assets -----------------------------------------------------------------


def create_asset(
    conn: sqlite3.Connection,
    *,
    sha256: str,
    project_id: int | None = None,
    uploaded_by: int | None = None,
    local_path: str | None = None,
    fal_url: str | None = None,
    width: int | None = None,
    height: int | None = None,
    kind: AssetKind = AssetKind.UPLOAD,
) -> int:
    """Insert an asset row.

    Raises ForeignKeyViolation for a project or uploader that does not exist.
    There was no handling here at all, so uploading against `project_id=424242`
    escaped as a bare IntegrityError and reached the browser as a 500.
    """
    try:
        cur = conn.execute(
            """
            INSERT INTO assets (project_id, uploaded_by, sha256, local_path, fal_url,
                                fal_url_at, width, height, kind, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (project_id, uploaded_by, sha256, local_path, fal_url,
             utcnow() if fal_url else None, width, height, str(kind), utcnow()),
        )
    except sqlite3.IntegrityError as exc:
        if _is_foreign_key_error(exc):
            raise ForeignKeyViolation(
                _name_missing_reference(conn, project_id=project_id, user_id=uploaded_by)
            ) from exc
        raise
    return int(cur.lastrowid)


def get_asset(conn: sqlite3.Connection, asset_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()


def find_asset_by_sha(
    conn: sqlite3.Connection, sha256: str, *, project_id: int | None = None
) -> sqlite3.Row | None:
    """The same image in the same project is the same asset."""
    if project_id is None:
        return conn.execute(
            "SELECT * FROM assets WHERE sha256 = ? AND project_id IS NULL "
            "ORDER BY id ASC LIMIT 1",
            (sha256,),
        ).fetchone()
    return conn.execute(
        "SELECT * FROM assets WHERE sha256 = ? AND project_id = ? ORDER BY id ASC LIMIT 1",
        (sha256, project_id),
    ).fetchone()


def find_fal_url_by_sha(conn: sqlite3.Connection, sha256: str) -> str | None:
    """Any project's copy of this exact image will do for the upload cache.

    Assets are scoped to a project so the galleries stay separate, but fal.ai's
    storage is not — identical bytes uploaded twice produce two identical files
    and one wasted round trip. This is the lookup that avoids it.
    """
    row = conn.execute(
        "SELECT fal_url FROM assets WHERE sha256 = ? AND fal_url IS NOT NULL "
        "ORDER BY fal_url_at DESC LIMIT 1",
        (sha256,),
    ).fetchone()
    return str(row["fal_url"]) if row else None


def set_asset_fal_url(conn: sqlite3.Connection, asset_id: int, fal_url: str) -> None:
    conn.execute(
        "UPDATE assets SET fal_url = ?, fal_url_at = ? WHERE id = ?",
        (fal_url, utcnow(), asset_id),
    )


def list_assets(
    conn: sqlite3.Connection, project_id: int, *, kind: AssetKind | None = None
) -> list[sqlite3.Row]:
    sql = "SELECT * FROM assets WHERE project_id = ?"
    params: list[Any] = [project_id]
    if kind is not None:
        sql += " AND kind = ?"
        params.append(str(kind))
    sql += " ORDER BY id DESC"
    return conn.execute(sql, params).fetchall()


# --- Job inputs -------------------------------------------------------------


def add_job_input(
    conn: sqlite3.Connection, *, job_id: int, asset_id: int, role: ImageRole, position: int
) -> None:
    conn.execute(
        "INSERT INTO job_inputs (job_id, asset_id, role, position) VALUES (?, ?, ?, ?)",
        (job_id, asset_id, str(role), position),
    )


def job_inputs(conn: sqlite3.Connection, job_id: int) -> list[sqlite3.Row]:
    """The reference images of one job, in the order they were sent.

    Position order is not cosmetic: the prompt refers to images by position
    ("image 1 is the geometry"), so reading them back in a different order would
    mean the recorded prompt describes a different arrangement than was sent.
    """
    return conn.execute(
        """
        SELECT ji.role, ji.position, a.*
          FROM job_inputs ji
          JOIN assets a ON a.id = ji.asset_id
         WHERE ji.job_id = ?
         ORDER BY ji.position ASC
        """,
        (job_id,),
    ).fetchall()


# --- Runtime controls -------------------------------------------------------


def get_app_setting(conn: sqlite3.Connection, key: str) -> str | None:
    """The stored override for one runtime control, or None for 'use .env'."""
    row = conn.execute(
        "SELECT value FROM app_settings WHERE key = ?", (key,)
    ).fetchone()
    return None if row is None else (str(row["value"]) if row["value"] is not None else None)


def set_app_setting(
    conn: sqlite3.Connection, key: str, value: str | None, *, updated_by: int | None
) -> None:
    """Store an override. `value=None` records an explicit null, not a clear."""
    conn.execute(
        """
        INSERT INTO app_settings (key, value, updated_at, updated_by)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at,
            updated_by = excluded.updated_by
        """,
        (key, value, utcnow(), updated_by),
    )


def clear_app_setting(conn: sqlite3.Connection, key: str) -> None:
    """Drop the override so the `.env` value applies again."""
    conn.execute("DELETE FROM app_settings WHERE key = ?", (key,))


def seed_from_db(value: object) -> int | None:
    """Decode a stored seed: an int, or the "0x..." hex text used for seeds past
    SQLite's signed 64-bit range (see fal_gateway._output_seed)."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.lower().startswith("0x"):
        return int(text, 16)
    try:
        return int(text)
    except ValueError:
        return None
