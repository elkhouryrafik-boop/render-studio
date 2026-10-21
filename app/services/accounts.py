"""Accounts: the access code, signing up, signing in, and approval.

The shape, in order:

    office access code  ->  sign up  ->  pending  ->  admin approves  ->  active

The access code decides who may *ask* for an account. Approval decides who may
spend money. Those are deliberately two different gates, because a shared code
leaks — someone reads it off a whiteboard, or forwards the welcome email — and a
leaked code should cost you an approval you decline, not a month's budget.

Email addresses are **not verified**. Nothing is sent. `admin@example.test` can be
typed by anyone holding the code, so the address is a label for the administrator
to recognise, not proof of identity. Approval is the only real control, and it is
stated here so no later code assumes the address means more than it does.
"""

from __future__ import annotations

import logging
import re
import secrets
import sqlite3
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from app import db, ratelimit
from app.config import Settings
from app.models import UserStatus

log = logging.getLogger(__name__)

# argon2id at the library's defaults, which track the RFC 9106 recommendations
# and are memory-hard — the property that makes GPU cracking expensive rather
# than merely slow.
_hasher = PasswordHasher()

# A dummy hash of a value nobody knows, verified when an email does not exist so
# that a missing account and a wrong password take the same time to reject.
# Without it, response timing tells an attacker which addresses are registered.
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(32))

# Deliberately permissive. Address validation is not identity verification here,
# and rejecting someone's genuinely odd corporate address helps nobody.
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Authentication or registration failed. Message is safe to show a user."""

    def __init__(self, message: str, reason: str = "auth") -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class SignupResult:
    user_id: int
    status: UserStatus
    is_admin: bool

    @property
    def needs_approval(self) -> bool:
        return self.status == UserStatus.PENDING


# --- The access code --------------------------------------------------------


def check_access_code(settings: Settings, supplied: str, *, client: str = "local") -> None:
    """Verify the office access code, in constant time, under a rate limit.

    `secrets.compare_digest` rather than `==`: a plain comparison returns as soon
    as two bytes differ, so the time it takes leaks how much of the code was
    correct, and a guess can be built one character at a time.
    """
    expected = settings.require_access_code()
    ratelimit.limiter.check("access_code", client, ratelimit.ACCESS_CODE)

    if not secrets.compare_digest(str(supplied or ""), expected):
        ratelimit.limiter.hit("access_code", client, ratelimit.ACCESS_CODE)
        raise AuthError("That access code is not right.", reason="bad_code")

    ratelimit.limiter.reset("access_code", client)


# --- Signing up -------------------------------------------------------------


def _validate_password(settings: Settings, password: str) -> None:
    if len(password or "") < settings.password_min_length:
        raise AuthError(
            f"Passwords need at least {settings.password_min_length} characters.",
            reason="weak_password",
        )


def signup(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    email: str,
    name: str,
    password: str,
    access_code: str,
    client: str = "local",
) -> SignupResult:
    """Create an account. It can do nothing until an administrator approves it.

    The address in `ADMIN_EMAIL` is the exception: the first account claiming it
    is created active and admin, because otherwise there is nobody to approve
    anybody and the office is locked out of its own app.
    """
    check_access_code(settings, access_code, client=client)
    ratelimit.limiter.check("signup", client, ratelimit.SIGNUP)

    email = (email or "").strip().lower()
    name = (name or "").strip()

    if not _EMAIL.match(email):
        raise AuthError("That does not look like an email address.", reason="bad_email")
    if not name:
        raise AuthError("A name is needed so renders can be attributed.", reason="no_name")
    _validate_password(settings, password)

    # The bootstrap admin, and the only accounts that skip approval: the office
    # domains listed in AUTO_APPROVE_DOMAINS, and everyone when approval is off.
    is_bootstrap_admin = bool(settings.admin_email) and email == settings.admin_email
    domain = email.rsplit("@", 1)[-1]
    trusted_domain = domain in settings.auto_approve_domain_set()
    status = (
        UserStatus.ACTIVE
        if is_bootstrap_admin or trusted_domain or not settings.signup_requires_approval
        else UserStatus.PENDING
    )

    user_id = db.create_user(
        conn,
        email=email,
        name=name,
        password_hash=_hasher.hash(password),
        status=status,
        is_admin=is_bootstrap_admin,
    )
    if user_id is None:
        # This address already has an account. Saying so would tell an anonymous
        # visitor which of their colleagues are registered, so the wording covers
        # both cases and the administrator sorts out the genuine collisions.
        ratelimit.limiter.hit("signup", client, ratelimit.SIGNUP)
        raise AuthError(
            "That address cannot be registered. If you already have an account, "
            "sign in instead; if you think this is wrong, ask the administrator.",
            reason="taken",
        )

    log.info("Account created: %s (status=%s, admin=%s)", email, status, is_bootstrap_admin)
    return SignupResult(user_id=user_id, status=status, is_admin=is_bootstrap_admin)


# --- Telling a newcomer they have been approved ------------------------------


def approval_status(conn: sqlite3.Connection, *, email: str, client: str = "local") -> str:
    """"pending", "active", or "unknown". Nothing else about the account leaks.

    Suspended and unknown addresses both read "unknown", so the answer cannot be
    used to find out whether a suspension happened. Every lookup counts against
    `ratelimit.APPROVAL_LOOKUP_ANY`; only unknown answers count against
    `ratelimit.APPROVAL_LOOKUP`.
    """
    ratelimit.limiter.check("approval_any", client, ratelimit.APPROVAL_LOOKUP_ANY)
    ratelimit.limiter.check("approval", client, ratelimit.APPROVAL_LOOKUP)
    ratelimit.limiter.hit("approval_any", client, ratelimit.APPROVAL_LOOKUP_ANY)
    row = db.get_user_by_email(conn, (email or "").strip().lower())
    state = str(row["status"]) if row is not None else ""
    if state in {UserStatus.PENDING, UserStatus.ACTIVE}:
        return state
    ratelimit.limiter.hit("approval", client, ratelimit.APPROVAL_LOOKUP)
    return "unknown"


# --- Signing in -------------------------------------------------------------


def login(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    email: str,
    password: str,
    client: str = "local",
    user_agent: str | None = None,
) -> tuple[sqlite3.Row, str]:
    """Verify credentials and mint a session. Returns (user row, raw token).

    Wrong password, unknown address and suspended account all produce the same
    message. Distinguishing them would turn the login form into a directory of
    who works here and whose account is disabled.

    The one exception is a *pending* account, which gets told it is waiting for
    approval — that person already proved they hold the access code and typed
    their own password correctly, and leaving them staring at "wrong password"
    while they wait would be cruel and would generate support requests.
    """
    ratelimit.limiter.check("login", client, ratelimit.LOGIN)
    email = (email or "").strip().lower()
    row = db.get_user_by_email(conn, email)

    stored = str(row["password_hash"]) if row is not None else _DUMMY_HASH
    try:
        _hasher.verify(stored, password or "")
        ok = True
    except (VerifyMismatchError, InvalidHashError):
        ok = False

    if row is None or not ok:
        ratelimit.limiter.hit("login", client, ratelimit.LOGIN)
        raise AuthError("Email or password is not right.", reason="bad_credentials")

    status = UserStatus(str(row["status"]))
    if status == UserStatus.PENDING:
        ratelimit.limiter.hit("login", client, ratelimit.LOGIN)
        raise AuthError(
            "This account is waiting for an administrator to approve it.",
            reason="pending",
        )
    if status != UserStatus.ACTIVE:
        ratelimit.limiter.hit("login", client, ratelimit.LOGIN)
        raise AuthError("Email or password is not right.", reason="bad_credentials")

    # Argon2's parameters get stronger between releases. Rehashing on a correct
    # password is the only moment the plaintext is available to do it with.
    if _hasher.check_needs_rehash(stored):
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (_hasher.hash(password), int(row["id"])),
        )

    ratelimit.limiter.reset("login", client)
    token = db.create_session(
        conn, int(row["id"]), user_agent=user_agent, days=settings.session_days
    )
    return row, token


def logout(conn: sqlite3.Connection, token: str) -> None:
    db.revoke_session(conn, token)


def current_user(conn: sqlite3.Connection, token: str | None) -> sqlite3.Row | None:
    """Resolve a session cookie to an active user, or None."""
    if not token:
        return None
    return db.session_user(conn, token)


def change_password(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    user_id: int,
    current_password: str,
    new_password: str,
) -> None:
    """Changing a password signs every other device out. That is the point of it.

    Rate limited on the account, because this route accepts `current_password`
    from inside a live session: a borrowed browser is otherwise an offline-speed
    guessing oracle for the password it already has access to. Only failures
    count, so somebody who mistypes once is not locked out of their own account.
    """
    scope_key = str(user_id)
    ratelimit.limiter.check("password_change", scope_key, ratelimit.PASSWORD_CHANGE)

    row = db.get_user(conn, user_id)
    if row is None:
        raise AuthError("No such account.", reason="no_user")
    try:
        _hasher.verify(str(row["password_hash"]), current_password or "")
    except (VerifyMismatchError, InvalidHashError) as exc:
        ratelimit.limiter.hit("password_change", scope_key, ratelimit.PASSWORD_CHANGE)
        raise AuthError("Current password is not right.", reason="bad_credentials") from exc

    _validate_password(settings, new_password)
    # One write. A new password that is live while the old sessions are still
    # valid is the window this whole operation exists to close.
    with db.transaction(conn):
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (_hasher.hash(new_password), user_id),
        )
        db.revoke_all_sessions(conn, user_id)
    ratelimit.limiter.reset("password_change", scope_key)


# --- Administration ---------------------------------------------------------


def _require_admin(conn: sqlite3.Connection, admin_id: int) -> sqlite3.Row:
    row = db.get_user(conn, admin_id)
    if row is None or not row["is_admin"] or str(row["status"]) != UserStatus.ACTIVE:
        raise AuthError("That action needs an administrator.", reason="not_admin")
    return row


def approve(conn: sqlite3.Connection, *, admin_id: int, user_id: int) -> None:
    _require_admin(conn, admin_id)
    db.set_user_status(conn, user_id, UserStatus.ACTIVE, approved_by=admin_id)
    log.info("User %s approved by %s", user_id, admin_id)


def suspend(conn: sqlite3.Connection, *, admin_id: int, user_id: int) -> None:
    """Suspend an account and sign it out everywhere.

    Leaving live sessions running would make suspension take effect whenever
    that person's cookie happened to expire, which is up to thirty days of
    someone spending a budget you have already decided they should not.
    """
    _require_admin(conn, admin_id)
    if admin_id == user_id:
        raise AuthError("Suspending your own account would lock you out.", reason="self")
    db.set_user_status(conn, user_id, UserStatus.SUSPENDED)
    db.revoke_all_sessions(conn, user_id)
    log.info("User %s suspended by %s", user_id, admin_id)


def set_budget(
    conn: sqlite3.Connection, *, admin_id: int, user_id: int, budget_usd: float | None
) -> None:
    """None restores the office default. Zero means this person cannot render."""
    _require_admin(conn, admin_id)
    if budget_usd is not None and budget_usd < 0:
        raise AuthError("A budget cannot be negative.", reason="bad_budget")
    db.set_user_budget(conn, user_id, budget_usd)


def grant_admin(conn: sqlite3.Connection, *, admin_id: int, user_id: int) -> None:
    _require_admin(conn, admin_id)
    conn.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user_id,))


def revoke_admin(conn: sqlite3.Connection, *, admin_id: int, user_id: int) -> None:
    """Refuses to remove the last administrator.

    An office with no admin cannot approve anyone, cannot suspend anyone, and
    cannot restore its own access without editing the database by hand.
    """
    _require_admin(conn, admin_id)
    remaining = conn.execute(
        "SELECT COUNT(*) AS n FROM users WHERE is_admin = 1 AND status = 'active' AND id != ?",
        (user_id,),
    ).fetchone()["n"]
    if not remaining:
        raise AuthError(
            "That is the only administrator. Promote someone else first.",
            reason="last_admin",
        )
    conn.execute("UPDATE users SET is_admin = 0 WHERE id = ?", (user_id,))


def pending_accounts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return db.list_users(conn, UserStatus.PENDING)
