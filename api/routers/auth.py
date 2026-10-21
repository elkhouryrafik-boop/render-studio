"""Access code, signup, sign in, sign out.

Cookie policy, in one place so it cannot drift between the two routes that set
it:

  httponly  — JavaScript cannot read the session, so an injected script cannot
              walk off with it
  samesite  — "lax", so another site cannot make an authenticated request on a
              signed-in user's behalf. This is what stands in for CSRF tokens
  secure    — from COOKIE_SECURE. Off on plain http, because a Secure cookie is
              never sent over http and nobody could sign in at all
  path      — "/", because the API and the app share an origin
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Cookie, HTTPException, Query, Response, status

from api import schemas
from api.deps import SESSION_COOKIE, ClientDep, ConnDep, SettingsDep, UserDep
from app import ratelimit
from app.config import Settings
from app.services import accounts

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _clear_session_cookie(response: Response, settings: Settings) -> None:
    """Clear the cookie ON the response that is actually returned.

    Both routes below used to call `response.delete_cookie()` on the injected
    `Response` and then return a *new* `Response(status_code=204)`. FastAPI
    0.115 merges `sub_response.headers` only when the handler returns something
    it has to serialise — return a Response instance and the injected one is
    dropped whole. Measured: logout answered 204 with no `set-cookie` at all, so
    the browser kept a cookie the server had already revoked.
    """
    response.delete_cookie(
        SESSION_COOKIE, path="/", httponly=True,
        samesite="lax", secure=settings.cookie_secure,
    )


def _set_session_cookie(response: Response, settings: Settings, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=settings.session_days * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


def _translate(exc: Exception) -> HTTPException:
    """Service errors become HTTP status codes without leaking anything extra."""
    if isinstance(exc, ratelimit.RateLimited):
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after)},
        )
    if isinstance(exc, accounts.AuthError):
        code = (
            status.HTTP_403_FORBIDDEN
            if exc.reason in {"bad_code", "pending", "not_admin"}
            else status.HTTP_400_BAD_REQUEST
        )
        return HTTPException(status_code=code, detail=str(exc))
    if isinstance(exc, RuntimeError):
        # OFFICE_ACCESS_CODE unset. A configuration fault, not a user error.
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    raise exc


@router.post("/access-code", status_code=status.HTTP_204_NO_CONTENT)
def verify_access_code(
    body: schemas.AccessCodeRequest, settings: SettingsDep, client: ClientDep
) -> Response:
    """Check the code before showing the signup form.

    Purely so the form does not ask for a name, an address and a password only
    to reject the code afterwards. It grants nothing on its own — signup checks
    the code again, because a client-side gate is not a gate.
    """
    try:
        accounts.check_access_code(settings, body.access_code, client=client)
    except Exception as exc:
        raise _translate(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/approval", response_model=schemas.ApprovalOut)
def approval(
    conn: ConnDep,
    client: ClientDep,
    email: Annotated[str, Query(max_length=320)] = "",
) -> schemas.ApprovalOut:
    """Has this address been approved yet? Public, because the waiting screen has
    no session: a newcomer who is not signed in is exactly who needs it.

    Rate limited by client like /access-code, so the lookup cannot be used to
    walk a list of addresses.
    """
    try:
        state = accounts.approval_status(conn, email=email, client=client)
    except Exception as exc:
        raise _translate(exc) from exc
    return schemas.ApprovalOut(status=state)


@router.post("/signup", response_model=schemas.SignupResponse)
def signup(
    body: schemas.SignupRequest,
    conn: ConnDep,
    settings: SettingsDep,
    client: ClientDep,
    response: Response,
) -> schemas.SignupResponse:
    try:
        result = accounts.signup(
            conn, settings,
            email=body.email, name=body.name, password=body.password,
            access_code=body.access_code, client=client,
        )
    except Exception as exc:
        raise _translate(exc) from exc

    if result.needs_approval:
        return schemas.SignupResponse(
            needs_approval=True,
            message=(
                "Account created. An administrator has to approve it before you "
                "can render — you will be able to sign in once they have."
            ),
        )

    # Active immediately (the bootstrap admin, or approval switched off) — sign
    # them in rather than making them type the password they just chose.
    _row, token = accounts.login(
        conn, settings, email=body.email, password=body.password, client=client
    )
    _set_session_cookie(response, settings, token)
    return schemas.SignupResponse(needs_approval=False, message="Signed in.")


@router.post("/login", response_model=schemas.UserOut)
def login(
    body: schemas.LoginRequest,
    conn: ConnDep,
    settings: SettingsDep,
    client: ClientDep,
    response: Response,
) -> schemas.UserOut:
    try:
        row, token = accounts.login(
            conn, settings, email=body.email, password=body.password, client=client
        )
    except Exception as exc:
        raise _translate(exc) from exc

    _set_session_cookie(response, settings, token)
    return schemas.user_out(row)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    conn: ConnDep,
    settings: SettingsDep,
    session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> Response:
    if session:
        accounts.logout(conn, session)
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    _clear_session_cookie(response, settings)
    return response


@router.get("/me", response_model=schemas.MeOut)
def me(user: UserDep, settings: SettingsDep) -> schemas.MeOut:
    base = schemas.user_out(user)
    return schemas.MeOut(**base.model_dump(), engine=settings.render_engine)


@router.post("/password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(
    body: schemas.PasswordChangeRequest,
    conn: ConnDep,
    settings: SettingsDep,
    user: UserDep,
) -> Response:
    """Change the password, then sign every device out — this one included.

    Rate limited on the account. This route takes `current_password`, so a
    borrowed or stolen session is otherwise a password oracle: guess from
    inside a logged-in browser as fast as argon2 will answer. Keyed on the
    account rather than the address, because the session already names it.
    """
    try:
        accounts.change_password(
            conn, settings, user_id=int(user["id"]),
            current_password=body.current_password,
            new_password=body.new_password,
        )
    except Exception as exc:
        raise _translate(exc) from exc

    # Every session was just revoked, including this one. Clearing the cookie
    # means the browser is told rather than left to discover it on the next call.
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    _clear_session_cookie(response, settings)
    return response
