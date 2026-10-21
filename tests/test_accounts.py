"""Authentication tests.

The security-relevant properties here are the ones that are easy to regress
without noticing, because nothing visibly breaks when they do: a constant-time
comparison replaced by `==`, a rate limit that counts successes, an error
message that quietly starts distinguishing "no such user" from "wrong password".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import db, ratelimit
from app.config import Settings
from app.models import UserStatus
from app.services import accounts

CODE = "office-code-123"
GOOD_PASSWORD = "correct-horse-battery"


@pytest.fixture(autouse=True)
def clean_limiter():
    """The limiter is process-global on purpose, so tests must not inherit state."""
    ratelimit.limiter.clear()
    yield
    ratelimit.limiter.clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        fal_key="test-key-not-real",
        office_access_code=CODE,
        admin_email="boss@example.test",
        data_dir=tmp_path,
        db_path=tmp_path / "t.db",
    )


@pytest.fixture
def conn(tmp_path: Path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def _signup(conn, settings, email="a@example.test", name="A", password=GOOD_PASSWORD, **kw):
    return accounts.signup(
        conn, settings, email=email, name=name, password=password,
        access_code=CODE, **kw,
    )


# --- The access code --------------------------------------------------------


def test_a_wrong_access_code_is_refused(settings: Settings):
    with pytest.raises(accounts.AuthError, match="access code"):
        accounts.check_access_code(settings, "nope")


def test_an_unset_access_code_disables_signup_rather_than_allowing_everyone(
    conn, tmp_path: Path
):
    """An empty code must not compare equal to an empty submission."""
    blank = Settings(fal_key="k", office_access_code="", data_dir=tmp_path,
                     db_path=tmp_path / "t.db")
    with pytest.raises(RuntimeError, match="OFFICE_ACCESS_CODE is not set"):
        accounts.check_access_code(blank, "")


def test_the_access_code_locks_out_after_repeated_guesses(settings: Settings):
    """Guessing a shared code over a LAN is a script. It has to hit a wall."""
    for _ in range(ratelimit.ACCESS_CODE.attempts):
        with pytest.raises(accounts.AuthError):
            accounts.check_access_code(settings, "wrong", client="10.0.0.9")

    with pytest.raises(ratelimit.RateLimited):
        accounts.check_access_code(settings, "wrong", client="10.0.0.9")

    # Even the correct code is refused while the lockout stands.
    with pytest.raises(ratelimit.RateLimited):
        accounts.check_access_code(settings, CODE, client="10.0.0.9")


def test_one_persons_lockout_does_not_lock_out_the_office(settings: Settings):
    for _ in range(ratelimit.ACCESS_CODE.attempts):
        with pytest.raises(accounts.AuthError):
            accounts.check_access_code(settings, "wrong", client="10.0.0.9")
    accounts.check_access_code(settings, CODE, client="10.0.0.10")


def test_a_correct_code_clears_the_attempt_history(settings: Settings):
    for _ in range(3):
        with pytest.raises(accounts.AuthError):
            accounts.check_access_code(settings, "wrong", client="10.0.0.11")
    accounts.check_access_code(settings, CODE, client="10.0.0.11")
    for _ in range(ratelimit.ACCESS_CODE.attempts - 1):
        with pytest.raises(accounts.AuthError):
            accounts.check_access_code(settings, "wrong", client="10.0.0.11")
    # Still under the limit — the counter genuinely reset rather than decayed.
    accounts.check_access_code(settings, CODE, client="10.0.0.11")


# --- Signing up -------------------------------------------------------------


def test_signup_without_the_code_creates_nothing(conn, settings: Settings):
    with pytest.raises(accounts.AuthError, match="access code"):
        accounts.signup(
            conn, settings, email="a@example.test", name="A",
            password=GOOD_PASSWORD, access_code="wrong",
        )
    assert db.count_users(conn) == 0


def test_a_new_account_waits_for_approval(conn, settings: Settings):
    result = _signup(conn, settings)
    assert result.needs_approval
    assert db.get_user(conn, result.user_id)["status"] == UserStatus.PENDING


def test_the_bootstrap_admin_skips_approval(conn, settings: Settings):
    """Otherwise there is nobody to approve anybody and the office is locked out."""
    result = _signup(conn, settings, email="BOSS@example.test", name="Boss")
    assert result.is_admin
    assert result.status == UserStatus.ACTIVE
    assert not result.needs_approval


def _with_auto_approve(tmp_path: Path, domains: str) -> Settings:
    return Settings(
        fal_key="k", office_access_code=CODE, admin_email="boss@example.test",
        auto_approve_domains=domains, data_dir=tmp_path, db_path=tmp_path / "t.db",
    )


def test_an_address_on_an_auto_approve_domain_is_active_at_once(conn, tmp_path: Path):
    """Domain match is case-insensitive and tolerates spaces in the list."""
    s = _with_auto_approve(tmp_path, " example.test, partner.example ")
    result = _signup(conn, s, email="Ana@example.test")
    assert result.status == UserStatus.ACTIVE
    assert not result.needs_approval
    assert not result.is_admin
    assert _signup(conn, s, email="bo@partner.example").status == UserStatus.ACTIVE


def test_an_address_on_any_other_domain_still_waits(conn, tmp_path: Path):
    """Only the exact domain counts: a suffix or a lookalike must not slip through."""
    s = _with_auto_approve(tmp_path, "example.test")
    assert _signup(conn, s, email="c@notexample.test").status == UserStatus.PENDING
    assert _signup(conn, s, email="d@example.test.evil.net").status == UserStatus.PENDING
    assert _signup(conn, s, email="e@sub.example.test").status == UserStatus.PENDING


def test_an_empty_auto_approve_list_approves_nobody_by_domain(conn, tmp_path: Path):
    s = _with_auto_approve(tmp_path, "")
    assert _signup(conn, s, email="f@example.test").status == UserStatus.PENDING


def test_a_short_password_is_refused(conn, settings: Settings):
    with pytest.raises(accounts.AuthError, match="at least"):
        _signup(conn, settings, password="short")
    assert db.count_users(conn) == 0


def test_a_taken_address_does_not_say_it_is_taken(conn, settings: Settings):
    """Telling an anonymous visitor which colleagues have accounts is a disclosure."""
    _signup(conn, settings)
    with pytest.raises(accounts.AuthError) as exc:
        _signup(conn, settings, email="A@example.test", name="Impostor")
    message = str(exc.value).lower()
    assert "already" not in message or "if you already have an account" in message
    assert "exists" not in message
    assert db.count_users(conn) == 1


def test_signup_can_be_set_to_not_require_approval(conn, tmp_path: Path):
    settings = Settings(
        fal_key="k", office_access_code=CODE, signup_requires_approval=False,
        data_dir=tmp_path, db_path=tmp_path / "t.db",
    )
    result = _signup(conn, settings)
    assert result.status == UserStatus.ACTIVE
    assert not result.is_admin


# --- Signing in -------------------------------------------------------------


def _active(conn, settings, email="a@example.test"):
    result = _signup(conn, settings, email=email)
    db.set_user_status(conn, result.user_id, UserStatus.ACTIVE)
    return result.user_id


def test_a_good_password_returns_a_working_session(conn, settings: Settings):
    uid = _active(conn, settings)
    row, token = accounts.login(
        conn, settings, email="a@example.test", password=GOOD_PASSWORD
    )
    assert row["id"] == uid
    assert accounts.current_user(conn, token)["id"] == uid


def test_a_wrong_password_and_an_unknown_address_read_identically(
    conn, settings: Settings
):
    """The login form must not double as a directory of who works here."""
    _active(conn, settings)
    with pytest.raises(accounts.AuthError) as wrong:
        accounts.login(conn, settings, email="a@example.test", password="not-it-at-all")
    with pytest.raises(accounts.AuthError) as unknown:
        accounts.login(conn, settings, email="ghost@example.test", password=GOOD_PASSWORD)
    assert str(wrong.value) == str(unknown.value)
    assert wrong.value.reason == unknown.value.reason


def test_an_unknown_address_still_verifies_a_hash(conn, settings: Settings, monkeypatch):
    """The dummy verify is what stops response timing revealing who is registered.

    Asserted by counting calls rather than by timing, because a timing assertion
    on a busy machine is a flaky test that gets deleted.
    """
    calls: list[str] = []
    real = accounts._hasher

    class CountingHasher:
        """PasswordHasher uses __slots__, so the instance itself is replaced."""

        def verify(self, stored, plain):
            calls.append(stored)
            return real.verify(stored, plain)

        def hash(self, plain):
            return real.hash(plain)

        def check_needs_rehash(self, stored):
            return real.check_needs_rehash(stored)

    monkeypatch.setattr(accounts, "_hasher", CountingHasher())
    with pytest.raises(accounts.AuthError):
        accounts.login(conn, settings, email="ghost@example.test", password="whatever")
    assert calls == [accounts._DUMMY_HASH], (
        "an unknown address must still cost one hash verification, or response "
        "timing reveals which addresses are registered"
    )


def test_a_pending_account_is_told_it_is_waiting(conn, settings: Settings):
    _signup(conn, settings)
    with pytest.raises(accounts.AuthError) as exc:
        accounts.login(conn, settings, email="a@example.test", password=GOOD_PASSWORD)
    assert exc.value.reason == "pending"


def test_a_suspended_account_is_not_told_it_is_suspended(conn, settings: Settings):
    uid = _active(conn, settings)
    db.set_user_status(conn, uid, UserStatus.SUSPENDED)
    with pytest.raises(accounts.AuthError) as exc:
        accounts.login(conn, settings, email="a@example.test", password=GOOD_PASSWORD)
    assert exc.value.reason == "bad_credentials"


def test_successful_logins_do_not_consume_the_rate_limit(conn, settings: Settings):
    """Someone signing in forty times a day must not lock themselves out."""
    _active(conn, settings)
    for _ in range(ratelimit.LOGIN.attempts + 5):
        accounts.login(
            conn, settings, email="a@example.test", password=GOOD_PASSWORD,
            client="10.0.0.20",
        )


def test_failed_logins_do_lock_out(conn, settings: Settings):
    _active(conn, settings)
    for _ in range(ratelimit.LOGIN.attempts):
        with pytest.raises(accounts.AuthError):
            accounts.login(
                conn, settings, email="a@example.test", password="wrong",
                client="10.0.0.21",
            )
    with pytest.raises(ratelimit.RateLimited):
        accounts.login(
            conn, settings, email="a@example.test", password=GOOD_PASSWORD,
            client="10.0.0.21",
        )


def test_logging_out_kills_the_session(conn, settings: Settings):
    _active(conn, settings)
    _row, token = accounts.login(
        conn, settings, email="a@example.test", password=GOOD_PASSWORD
    )
    accounts.logout(conn, token)
    assert accounts.current_user(conn, token) is None


def test_the_password_is_not_stored(conn, settings: Settings):
    _signup(conn, settings)
    stored = db.get_user_by_email(conn, "a@example.test")["password_hash"]
    assert GOOD_PASSWORD not in stored
    assert stored.startswith("$argon2id$")


def test_changing_a_password_signs_out_every_device(conn, settings: Settings):
    uid = _active(conn, settings)
    _row, phone = accounts.login(
        conn, settings, email="a@example.test", password=GOOD_PASSWORD
    )
    _row, laptop = accounts.login(
        conn, settings, email="a@example.test", password=GOOD_PASSWORD
    )
    accounts.change_password(
        conn, settings, user_id=uid,
        current_password=GOOD_PASSWORD, new_password="a-brand-new-passphrase",
    )
    assert accounts.current_user(conn, phone) is None
    assert accounts.current_user(conn, laptop) is None


# --- Administration ---------------------------------------------------------


def test_only_an_admin_can_approve(conn, settings: Settings):
    ordinary = _active(conn, settings, email="a@example.test")
    waiting = _signup(conn, settings, email="b@example.test", name="B").user_id
    with pytest.raises(accounts.AuthError, match="administrator"):
        accounts.approve(conn, admin_id=ordinary, user_id=waiting)
    assert db.get_user(conn, waiting)["status"] == UserStatus.PENDING


def test_an_admin_approves_and_the_account_can_sign_in(conn, settings: Settings):
    boss = _signup(conn, settings, email="boss@example.test", name="Boss").user_id
    waiting = _signup(conn, settings, email="b@example.test", name="B").user_id
    accounts.approve(conn, admin_id=boss, user_id=waiting)

    row = db.get_user(conn, waiting)
    assert row["status"] == UserStatus.ACTIVE
    assert row["approved_by"] == boss
    assert row["approved_at"]
    accounts.login(conn, settings, email="b@example.test", password=GOOD_PASSWORD)


def test_suspending_signs_that_person_out_immediately(conn, settings: Settings):
    """Otherwise suspension takes effect whenever their cookie happens to expire."""
    boss = _signup(conn, settings, email="boss@example.test", name="Boss").user_id
    victim = _active(conn, settings, email="b@example.test")
    _row, token = accounts.login(
        conn, settings, email="b@example.test", password=GOOD_PASSWORD
    )
    assert accounts.current_user(conn, token) is not None

    accounts.suspend(conn, admin_id=boss, user_id=victim)
    assert accounts.current_user(conn, token) is None


def test_an_admin_cannot_suspend_themselves(conn, settings: Settings):
    boss = _signup(conn, settings, email="boss@example.test", name="Boss").user_id
    with pytest.raises(accounts.AuthError, match="lock you out"):
        accounts.suspend(conn, admin_id=boss, user_id=boss)


def test_the_last_admin_cannot_be_demoted(conn, settings: Settings):
    boss = _signup(conn, settings, email="boss@example.test", name="Boss").user_id
    with pytest.raises(accounts.AuthError, match="only administrator"):
        accounts.revoke_admin(conn, admin_id=boss, user_id=boss)

    second = _active(conn, settings, email="b@example.test")
    accounts.grant_admin(conn, admin_id=boss, user_id=second)
    accounts.revoke_admin(conn, admin_id=boss, user_id=boss)
    assert db.get_user(conn, boss)["is_admin"] == 0


def test_an_admin_sets_a_personal_budget(conn, settings: Settings):
    boss = _signup(conn, settings, email="boss@example.test", name="Boss").user_id
    other = _active(conn, settings, email="b@example.test")

    accounts.set_budget(conn, admin_id=boss, user_id=other, budget_usd=0.0)
    assert db.get_user(conn, other)["budget_monthly_usd"] == 0.0

    accounts.set_budget(conn, admin_id=boss, user_id=other, budget_usd=None)
    assert db.get_user(conn, other)["budget_monthly_usd"] is None

    with pytest.raises(accounts.AuthError, match="negative"):
        accounts.set_budget(conn, admin_id=boss, user_id=other, budget_usd=-5)


def test_pending_accounts_are_listed_for_the_admin(conn, settings: Settings):
    _signup(conn, settings, email="a@example.test", name="A")
    _signup(conn, settings, email="b@example.test", name="B")
    _signup(conn, settings, email="boss@example.test", name="Boss")
    waiting = {r["email"] for r in accounts.pending_accounts(conn)}
    assert waiting == {"a@example.test", "b@example.test"}


def test_the_access_code_is_not_in_a_config_dump(settings: Settings):
    dumped = settings.safe_dump()
    assert dumped["office_access_code"] == "SET"
    assert CODE not in str(dumped)
    assert CODE not in repr(settings)


def test_the_new_password_and_the_revocations_land_together(conn, settings: Settings):
    """Two statements, one meaning. A new password that is live while the old
    sessions are still valid is the window this operation exists to close, so
    it is one transaction rather than two writes that usually both succeed."""
    uid = _active(conn, settings)
    _row, token = accounts.login(
        conn, settings, email="a@example.test", password=GOOD_PASSWORD
    )
    accounts.change_password(
        conn, settings, user_id=uid,
        current_password=GOOD_PASSWORD, new_password="a-brand-new-passphrase",
    )
    assert not conn.in_transaction, "the transaction was left open"
    assert accounts.current_user(conn, token) is None
    accounts.login(conn, settings, email="a@example.test", password="a-brand-new-passphrase")


def test_guessing_the_current_password_is_rate_limited(conn, settings: Settings):
    """This route takes the current password from inside a live session, so a
    borrowed browser is otherwise an oracle for the password it already has."""
    uid = _active(conn, settings)
    for _ in range(ratelimit.PASSWORD_CHANGE.attempts):
        with pytest.raises(accounts.AuthError):
            accounts.change_password(
                conn, settings, user_id=uid,
                current_password="wrong", new_password="a-brand-new-passphrase",
            )
    with pytest.raises(ratelimit.RateLimited):
        accounts.change_password(
            conn, settings, user_id=uid,
            current_password=GOOD_PASSWORD, new_password="a-brand-new-passphrase",
        )


def test_a_successful_change_clears_the_lockout(conn, settings: Settings):
    """Only failures count. Someone who mistypes once and then gets it right
    must not be left one attempt short of their own account."""
    uid = _active(conn, settings)
    with pytest.raises(accounts.AuthError):
        accounts.change_password(
            conn, settings, user_id=uid,
            current_password="wrong", new_password="a-brand-new-passphrase",
        )
    accounts.change_password(
        conn, settings, user_id=uid,
        current_password=GOOD_PASSWORD, new_password="a-brand-new-passphrase",
    )
    ratelimit.limiter.check("password_change", str(uid), ratelimit.PASSWORD_CHANGE)


def test_expired_sessions_are_actually_deleted(conn, settings: Settings):
    """`purge_expired_sessions` existed and was called from nowhere, so the
    table only ever grew. The lifespan runs it at startup and the poller hourly."""
    uid = _active(conn, settings)
    live = db.create_session(conn, uid, days=30)
    stale = db.create_session(conn, uid, days=-1)

    assert db.session_user(conn, stale) is None, "an expired session must not resolve"
    rows = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
    assert rows == 2, "expired but still on disk, which is the leak"

    assert db.purge_expired_sessions(conn) == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"] == 1
    assert db.session_user(conn, live) is not None, "a live session was purged"


# --- Telling a newcomer whether they have been approved ---------------------


def test_the_approval_lookup_reports_pending_then_active(conn, settings: Settings):
    result = _signup(conn, settings, email="new@example.test")
    assert accounts.approval_status(conn, email="new@example.test", client="10.0.0.1") == "pending"
    db.set_user_status(conn, result.user_id, UserStatus.ACTIVE)
    assert accounts.approval_status(conn, email="NEW@example.test", client="10.0.0.1") == "active"


def test_suspended_and_unknown_addresses_both_read_as_unknown(conn, settings: Settings):
    result = _signup(conn, settings, email="gone@example.test")
    db.set_user_status(conn, result.user_id, UserStatus.SUSPENDED)
    assert accounts.approval_status(conn, email="gone@example.test", client="10.0.0.1") == "unknown"
    assert accounts.approval_status(conn, email="never@example.test", client="10.0.0.1") == "unknown"


def test_polling_a_pending_account_never_uses_up_the_lookup_allowance(conn, settings: Settings):
    """The waiting screen polls every 15 seconds. Pending and active must be free."""
    _signup(conn, settings, email="wait@example.test")
    for _ in range(ratelimit.APPROVAL_LOOKUP.attempts * 3):
        assert accounts.approval_status(conn, email="wait@example.test", client="10.0.0.2") == "pending"
