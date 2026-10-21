"""Approving accounts, suspending them, and setting what they may spend.

This is where the office's money is actually controlled. Every route requires an
active administrator, checked per request rather than trusted from login.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from api import schemas
from api.deps import AdminDep, ConnDep, SettingsDep
from app import budget, db
from app.models import JobStatus
from app.services import accounts

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _translate(exc: accounts.AuthError) -> HTTPException:
    code = (
        status.HTTP_403_FORBIDDEN
        if exc.reason in {"not_admin", "self", "last_admin"}
        else status.HTTP_400_BAD_REQUEST
    )
    return HTTPException(status_code=code, detail=str(exc))


@router.get("/users", response_model=list[schemas.UserOut])
def list_users(conn: ConnDep, _admin: AdminDep) -> list[schemas.UserOut]:
    return [schemas.user_out(r) for r in db.list_users(conn)]


@router.get("/users/pending", response_model=list[schemas.UserOut])
def pending(conn: ConnDep, _admin: AdminDep) -> list[schemas.UserOut]:
    return [schemas.user_out(r) for r in accounts.pending_accounts(conn)]


@router.post("/users/{user_id}/approve", response_model=schemas.UserOut)
def approve(user_id: int, conn: ConnDep, admin: AdminDep) -> schemas.UserOut:
    try:
        accounts.approve(conn, admin_id=int(admin["id"]), user_id=user_id)
    except accounts.AuthError as exc:
        raise _translate(exc) from exc
    row = db.get_user(conn, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such account.")
    return schemas.user_out(row)


@router.post("/users/{user_id}/suspend", response_model=schemas.UserOut)
def suspend(user_id: int, conn: ConnDep, admin: AdminDep) -> schemas.UserOut:
    """Suspends and signs that person out of every device immediately."""
    try:
        accounts.suspend(conn, admin_id=int(admin["id"]), user_id=user_id)
    except accounts.AuthError as exc:
        raise _translate(exc) from exc
    row = db.get_user(conn, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such account.")
    return schemas.user_out(row)


@router.post("/users/{user_id}/reinstate", response_model=schemas.UserOut)
def reinstate(user_id: int, conn: ConnDep, admin: AdminDep) -> schemas.UserOut:
    try:
        accounts.approve(conn, admin_id=int(admin["id"]), user_id=user_id)
    except accounts.AuthError as exc:
        raise _translate(exc) from exc
    row = db.get_user(conn, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such account.")
    return schemas.user_out(row)


@router.put("/users/{user_id}/budget", response_model=schemas.UserOut)
def set_budget(
    user_id: int, body: schemas.BudgetUpdate, conn: ConnDep, admin: AdminDep
) -> schemas.UserOut:
    """null restores the office default. 0 means this person cannot render."""
    try:
        accounts.set_budget(
            conn, admin_id=int(admin["id"]), user_id=user_id,
            budget_usd=body.budget_monthly_usd,
        )
    except accounts.AuthError as exc:
        raise _translate(exc) from exc
    row = db.get_user(conn, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such account.")
    return schemas.user_out(row)


@router.post("/users/{user_id}/admin", response_model=schemas.UserOut)
def grant_admin(user_id: int, conn: ConnDep, admin: AdminDep) -> schemas.UserOut:
    try:
        accounts.grant_admin(conn, admin_id=int(admin["id"]), user_id=user_id)
    except accounts.AuthError as exc:
        raise _translate(exc) from exc
    row = db.get_user(conn, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such account.")
    return schemas.user_out(row)


@router.delete("/users/{user_id}/admin", response_model=schemas.UserOut)
def revoke_admin(user_id: int, conn: ConnDep, admin: AdminDep) -> schemas.UserOut:
    """Refuses to remove the last administrator — that would lock the office out."""
    try:
        accounts.revoke_admin(conn, admin_id=int(admin["id"]), user_id=user_id)
    except accounts.AuthError as exc:
        raise _translate(exc) from exc
    row = db.get_user(conn, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such account.")
    return schemas.user_out(row)


@router.get("/controls", response_model=schemas.ControlsOut)
def controls(conn: ConnDep, settings: SettingsDep, _admin: AdminDep) -> schemas.ControlsOut:
    """The kill switch and the office monthly cap, as they stand now."""
    return schemas.ControlsOut(
        kill_switch=budget.kill_switch(conn, settings),
        monthly_cap_usd=budget.monthly_cap_usd(conn, settings),
    )


@router.put("/controls/kill-switch", response_model=schemas.ControlsOut)
def set_kill_switch(
    body: schemas.KillSwitchRequest,
    conn: ConnDep,
    settings: SettingsDep,
    admin: AdminDep,
) -> schemas.ControlsOut:
    """Stop or resume all spending for the whole office, without a restart.

    Takes effect on the next request; the override lives in app_settings.
    """
    budget.set_kill_switch(
        conn, enabled=body.on, actor=str(admin["email"]), actor_id=int(admin["id"]),
    )
    return controls(conn, settings, admin)


@router.get("/spend")
def spend(conn: ConnDep, settings: SettingsDep, _admin: AdminDep) -> dict[str, object]:
    """This month's spend, whole office and per person.

    Reads the jobs table, which is the ledger — there is no second store to
    drift out of sync with what was actually submitted.
    """
    per_user = []
    for row in db.list_users(conn):
        per_user.append(
            {
                "user_id": int(row["id"]),
                "email": str(row["email"]),
                "name": str(row["name"]),
                "status": str(row["status"]),
                "spent_usd": db.spend_this_month(
                    conn, username=str(row["email"]), user_id=int(row["id"])
                ),
                "cap_usd": (
                    float(row["budget_monthly_usd"])
                    if row["budget_monthly_usd"] is not None
                    else settings.budget_per_user_usd
                ),
            }
        )
    return {
        "month_total_usd": db.spend_this_month(conn),
        # What the caps are measured against: the renders.
        "committed_total_usd": db.committed_spend_this_month(conn),
        "monthly_cap_usd": budget.monthly_cap_usd(conn, settings),
        "kill_switch": budget.kill_switch(conn, settings),
        "per_user": per_user,
    }


@router.get("/activity", response_model=schemas.ActivityOut)
def activity(
    conn: ConnDep,
    _admin: AdminDep,
    # ge=1 for the same reason as the renders list: LIMIT -1 means no limit.
    limit: int = Query(default=100, ge=1, le=500),
    before_id: int | None = Query(default=None, ge=1),
    user_id: int | None = Query(default=None, ge=1),
    status_: str | None = Query(default=None, alias="status"),
) -> schemas.ActivityOut:
    """Everything every user has generated, newest first, with per-person totals.

    The Studio already shows the office's renders to everyone. This is the
    administrator's view of the same work: who asked, what it cost, and how it
    ended, filterable by person and status and paged by job id.
    """
    choices = [s.value for s in JobStatus]
    if status_ is not None and status_ not in choices:
        raise HTTPException(
            status_code=422,
            detail=f"Status must be one of: {', '.join(choices)}.",
        )
    rows = db.activity_jobs(
        conn, limit, before_id=before_id, user_id=user_id, status=status_
    )
    outputs = db.job_outputs_for(conn, [int(r["id"]) for r in rows])
    jobs = [schemas.activity_job_out(r, outputs.get(int(r["id"]), [])) for r in rows]
    return schemas.ActivityOut(
        jobs=jobs,
        users=[schemas.ActivityUserOut(**u) for u in db.activity_totals(conn)],
        next_before_id=int(rows[-1]["id"]) if len(rows) == limit else None,
        month_total_usd=db.spend_this_month(conn),
    )
