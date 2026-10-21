"""Spend guardrails.

fal.ai has no native spend cap or budget alert (checked against fal.ai's own
docs, 2026-08). The only backstop fal.ai provides is the prepaid balance hitting
zero and locking the account, with no warning.

So this module is the ceiling. Every submission passes through `check()` before
any money is committed. Four gates, cheapest check first:

  1. Kill switch      — block everything without taking the app down
  2. Per-job          — catches a fat-fingered resolution before it bills
  3. Per-user monthly — one person can't drain the shared budget
  4. Global monthly   — the hard ceiling

Gates 3 and 4 read a running total, so they are only as honest as the write that
follows them. `reserve_within_budget()` is the path that does the read and the
write as one transaction; `check()` on its own is the advisory version and says
so at its call site.

The kill switch and the office cap are readable from the database as well as
from `.env`. The reason is operational: the moment you want the kill switch is
the moment money is leaving, and "edit .env and restart the process" is not a
response to that.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from app import db
from app.config import Settings

log = logging.getLogger(__name__)

# Keys in the app_settings table. An absent row means "use the .env value", so
# the file stays the default and the database records only deliberate overrides.
KILL_SWITCH_KEY = "kill_switch"
MONTHLY_CAP_KEY = "budget_monthly_usd"

_TRUTHY = {"1", "true", "yes", "on"}


class BudgetExceeded(Exception):
    """Raised instead of submitting.

    The message stays explicit about amounts: it is what lands in the job
    ledger and in logs, and a guardrail that hedges about why it fired is
    useless during an incident. `reason` lets a caller present its own wording
    without parsing the string.
    """

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


# --- Runtime controls -------------------------------------------------------


def kill_switch(conn: sqlite3.Connection, settings: Settings) -> bool:
    """The kill switch as it stands right now, not as it stood at boot."""
    raw = db.get_app_setting(conn, KILL_SWITCH_KEY)
    if raw is None:
        return settings.kill_switch
    return raw.strip().lower() in _TRUTHY


def monthly_cap_usd(conn: sqlite3.Connection, settings: Settings) -> float:
    """The office monthly cap, admin override first, `.env` otherwise.

    A stored value that will not parse falls back to `.env` rather than to zero
    or to infinity: both of those are a guess about which way an operator would
    rather be wrong, and neither is defensible without asking.
    """
    raw = db.get_app_setting(conn, MONTHLY_CAP_KEY)
    if raw is None:
        return settings.budget_monthly_usd
    try:
        return float(raw)
    except (TypeError, ValueError):
        log.warning(
            "app_settings.%s holds %r, which is not a number. Using the .env "
            "value of %.2f.", MONTHLY_CAP_KEY, raw, settings.budget_monthly_usd,
        )
        return settings.budget_monthly_usd


def set_kill_switch(
    conn: sqlite3.Connection, *, enabled: bool, actor: str, actor_id: int | None = None
) -> None:
    """Turn spending off (or back on) for the whole office, immediately.

    Logged with who did it. A control that stops the office working is one
    people ask about afterwards, and "it was off when I got in" needs an answer.
    """
    db.set_app_setting(conn, KILL_SWITCH_KEY, "1" if enabled else "0", updated_by=actor_id)
    log.warning("Kill switch turned %s by %s", "ON" if enabled else "OFF", actor)


def set_monthly_cap(
    conn: sqlite3.Connection,
    *,
    cap_usd: float | None,
    actor: str,
    actor_id: int | None = None,
) -> None:
    """Set the office monthly cap. `None` restores the `.env` default."""
    if cap_usd is None:
        db.clear_app_setting(conn, MONTHLY_CAP_KEY)
        log.warning("Office monthly cap reset to the .env default by %s", actor)
        return
    db.set_app_setting(conn, MONTHLY_CAP_KEY, f"{float(cap_usd):.6f}", updated_by=actor_id)
    log.warning("Office monthly cap set to $%.2f by %s", cap_usd, actor)


@dataclass(frozen=True)
class BudgetStatus:
    month_total_usd: float
    month_user_usd: float
    monthly_cap_usd: float
    per_user_cap_usd: float
    kill_switch: bool

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.monthly_cap_usd - self.month_total_usd)

    @property
    def user_remaining_usd(self) -> float:
        return max(0.0, self.per_user_cap_usd - self.month_user_usd)

    @property
    def fraction_used(self) -> float:
        if self.monthly_cap_usd <= 0:
            return 1.0
        return min(1.0, self.month_total_usd / self.monthly_cap_usd)


def status(
    conn: sqlite3.Connection,
    settings: Settings,
    username: str,
    *,
    user_id: int | None = None,
    per_user_cap_usd: float | None = None,
) -> BudgetStatus:
    """Current spend against the ceilings that apply to this person.

    `user_id` makes the per-user gate real. Until accounts existed the ledger
    keyed on a free-text name box, so the cap was bypassed by typing a different
    name. Passing the id keys it to an account instead — and the query still
    counts that person's pre-accounts rows, so activating accounts does not hand
    everyone a fresh budget mid-month.

    `per_user_cap_usd` is the per-person override an admin can set. None means
    the office default, so raising the default raises it for everyone who has
    not been given a figure of their own.
    """
    return BudgetStatus(
        # Renders plus prompt rewrites. The rewriter bills a different vendor
        # out of the same office budget and was previously recorded, shown to
        # admins, and counted against nothing at all.
        month_total_usd=db.committed_spend_this_month(conn),
        month_user_usd=db.committed_spend_this_month(
            conn, username=username, user_id=user_id
        ),
        monthly_cap_usd=monthly_cap_usd(conn, settings),
        per_user_cap_usd=(
            settings.budget_per_user_usd if per_user_cap_usd is None else per_user_cap_usd
        ),
        kill_switch=kill_switch(conn, settings),
    )


def check(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    username: str,
    est_usd: float,
    user_id: int | None = None,
    per_user_cap_usd: float | None = None,
) -> BudgetStatus:
    """Raise BudgetExceeded if this job must not be submitted.

    Returns the current status when the job is allowed, so callers can display it.

    ADVISORY for the two monthly gates. It reads a SUM() and returns; whatever
    writes the reservation does so afterwards, and between the two reads N
    concurrent submissions all see the same pre-spend total and all pass.
    Measured at 40 threads against a $1 cap and $0.10 a job: 40 reservations and
    $4.00 committed. The atomic path is `reserve_within_budget()` below, which
    does the same check and the INSERT inside one BEGIN IMMEDIATE transaction —
    the same measurement through it commits exactly $1.00.

    Keep both, in this order:

        check(...)                    # here, before uploading — fails fast
        ... upload inputs ...
        reserve_within_budget(...)    # at the write — the gate that decides

    This call stays useful as a pre-flight, because refusing before an image is
    uploaded is a better experience than refusing after. It is just not the
    thing that may be trusted to hold the cap.
    """
    if kill_switch(conn, settings):
        raise BudgetExceeded(
            "Rendering is paused for the whole office (kill switch on). No jobs "
            "will be submitted until an administrator lifts it.",
            reason="kill_switch",
        )

    if est_usd > settings.budget_per_job_usd:
        raise BudgetExceeded(
            f"This job estimates at ${est_usd:.3f}, over the per-job limit of "
            f"${settings.budget_per_job_usd:.2f}. Lower the resolution or image "
            f"count, or raise BUDGET_PER_JOB_USD if this is genuinely intended.",
            reason="per_job",
        )

    st = status(
        conn, settings, username, user_id=user_id, per_user_cap_usd=per_user_cap_usd
    )

    if st.month_user_usd + est_usd > st.per_user_cap_usd:
        raise BudgetExceeded(
            f"{username} has spent ${st.month_user_usd:.2f} of a "
            f"${st.per_user_cap_usd:.2f} monthly allowance. This job "
            f"(${est_usd:.3f}) would exceed it.",
            reason="per_user",
        )

    if st.month_total_usd + est_usd > st.monthly_cap_usd:
        raise BudgetExceeded(
            f"Office spend is ${st.month_total_usd:.2f} of a "
            f"${st.monthly_cap_usd:.2f} monthly cap. This job (${est_usd:.3f}) "
            f"would exceed it. Raise BUDGET_MONTHLY_USD in .env if intended.",
            reason="monthly",
        )

    return st


def reserve_within_budget(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    username: str,
    est_usd: float,
    user_id: int | None = None,
    per_user_cap_usd: float | None = None,
    **reserve_kwargs: object,
) -> db.Reservation:
    """Authorise and reserve as one write. THE path a submitter should take.

    Same four gates as `check()`, in the same order, with one difference that
    is the whole point: the two gates that read a running total do so inside the
    transaction that writes the row, so two submissions cannot both be measured
    against the same pre-spend number.

    The cheap gates run first and outside the transaction — refusing a
    fat-fingered resolution should not queue behind another architect's write.

    Raises BudgetExceeded exactly as `check()` does; returns the reservation
    (which may be a duplicate) when the job is allowed. `reserve_kwargs` are
    `db.reserve_job`'s remaining arguments and are passed straight through.
    """
    if kill_switch(conn, settings):
        raise BudgetExceeded(
            "Rendering is paused for the whole office (kill switch on). No jobs "
            "will be submitted until an administrator lifts it.",
            reason="kill_switch",
        )

    if est_usd > settings.budget_per_job_usd:
        raise BudgetExceeded(
            f"This job estimates at ${est_usd:.3f}, over the per-job limit of "
            f"${settings.budget_per_job_usd:.2f}. Lower the resolution or image "
            f"count, or raise BUDGET_PER_JOB_USD if this is genuinely intended.",
            reason="per_job",
        )

    cap_user = (
        settings.budget_per_user_usd if per_user_cap_usd is None else per_user_cap_usd
    )
    cap_month = monthly_cap_usd(conn, settings)

    outcome = db.reserve_job_within_budget(
        conn,
        monthly_cap_usd=cap_month,
        per_user_cap_usd=cap_user,
        est_usd=est_usd,
        username=username,
        user_id=user_id,
        **reserve_kwargs,
    )

    if outcome.refused == "per_user":
        raise BudgetExceeded(
            f"{username} has spent ${outcome.month_user_usd:.2f} of a "
            f"${cap_user:.2f} monthly allowance. This job "
            f"(${est_usd:.3f}) would exceed it.",
            reason="per_user",
        )
    if outcome.refused == "monthly":
        raise BudgetExceeded(
            f"Office spend is ${outcome.month_total_usd:.2f} of a "
            f"${cap_month:.2f} monthly cap. This job (${est_usd:.3f}) "
            f"would exceed it. Raise the cap if that is intended.",
            reason="monthly",
        )
    return outcome


def warning_text(st: BudgetStatus) -> str | None:
    """Nudge before the wall, so the cap is never a surprise."""
    if st.kill_switch:
        return "Kill switch is ON — submissions are blocked."
    used = st.fraction_used
    if used >= 0.9:
        return f"${st.remaining_usd:.2f} left of this month's ${st.monthly_cap_usd:.2f} budget."
    if used >= 0.75:
        return f"{used:.0%} of this month's budget used."
    return None
