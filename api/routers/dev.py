"""Development hooks. They exist only while the fake engine is running.

Registered always and guarded per request, because the engine is a setting and
tests flip it with a dependency override. Under any other engine every route here
answers 404 "Not available.", before the caller's role is even looked at, so the
route is indistinguishable from one that does not exist.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from api.deps import AdminDep, SettingsDep
from app import fal_gateway

router = APIRouter(prefix="/api/dev", tags=["dev"])


def _fake_engine_only(settings: SettingsDep) -> None:
    if settings.render_engine != "fake":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not available.")


@router.post(
    "/fail-next",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(_fake_engine_only)],
)
def fail_next(_admin: AdminDep, settings: SettingsDep) -> Response:
    """Make the next request the fake engine completes come back with no image,
    so the Studio's failure path can be exercised without a real failure."""
    fal_gateway._client(settings).fail_next = True
    return Response(status_code=status.HTTP_204_NO_CONTENT)
