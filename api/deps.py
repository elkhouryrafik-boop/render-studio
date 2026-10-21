"""Shared dependencies: the connection, the settings, and who is asking.

Connections are per request. FastAPI resolves a sync dependency on one
threadpool thread and runs the endpoint on another, so a connection cached per
thread is still handed between threads and still collides: one
`sqlite3.Connection` used concurrently raises `sqlite3.InterfaceError: bad
parameter or other API misuse` and corrupts in-flight cursors (the Studio saw
spurious 401s, 404s and 500s whenever four image tiles loaded at once, before
and after the per-thread attempt). `get_conn` therefore opens a connection when
a request starts and closes it when the response is done; `db.connect` sets WAL,
the 30 s busy timeout and foreign keys on each one, and opening costs about a
millisecond. The poller and the lifespan use `connection()` the same way.

The authentication dependency is the only thing standing between the office LAN
and someone else's renders, so it is deliberately small enough to read in one
go and has no branch that returns a user without a valid session.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, Request, status

from app import db
from app.config import Settings, get_settings
from app.models import UserStatus
from app.routing import RoutingTable
from app.services.render import Actor

SESSION_COOKIE = "rs_session"


@lru_cache(maxsize=1)
def _routing_table() -> RoutingTable:
    return RoutingTable.load(get_settings().routing_path)


@contextlib.contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    """A connection for one unit of work, closed when that work ends."""
    conn = db.connect(get_settings().db_path)
    try:
        yield conn
    finally:
        with contextlib.suppress(sqlite3.Error):
            conn.close()


def get_conn() -> Iterator[sqlite3.Connection]:
    """One connection per request (FastAPI runs the teardown after the response).

    Tests override this dependency with a lambda returning their temporary
    connection; FastAPI accepts a plain callable in place of a generator.
    """
    with connection() as conn:
        yield conn


def get_config() -> Settings:
    return get_settings()


def get_table() -> RoutingTable:
    return _routing_table()


ConnDep = Annotated[sqlite3.Connection, Depends(get_conn)]
SettingsDep = Annotated[Settings, Depends(get_config)]
TableDep = Annotated[RoutingTable, Depends(get_table)]


def client_ip(request: Request) -> str:
    """The key rate limits are counted against.

    No X-Forwarded-For handling. On a LAN there is no proxy in front, and
    trusting that header without one lets any client pick its own rate-limit
    bucket by sending a different value each attempt. If Caddy is put in front
    later, this is the function to revisit — and only then.
    """
    return request.client.host if request.client else "unknown"


ClientDep = Annotated[str, Depends(client_ip)]


def current_user(
    conn: ConnDep,
    session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> sqlite3.Row:
    """The signed-in, active user, or 401.

    Status is re-checked on every request rather than trusted from login time,
    so suspending an account takes effect immediately instead of whenever that
    person's cookie happens to expire.
    """
    row = db.session_user(conn, session) if session else None
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not signed in.",
        )
    return row


UserDep = Annotated[sqlite3.Row, Depends(current_user)]


def current_admin(user: UserDep) -> sqlite3.Row:
    if not user["is_admin"] or str(user["status"]) != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="That needs an administrator.",
        )
    return user


AdminDep = Annotated[sqlite3.Row, Depends(current_admin)]


def current_actor(user: UserDep) -> Actor:
    """The signed-in user, in the form the render service takes."""
    return Actor.from_user(user)


ActorDep = Annotated[Actor, Depends(current_actor)]
