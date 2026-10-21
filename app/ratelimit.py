"""In-memory sliding-window rate limiting.

Guessing a shared office access code over a LAN is a script and five minutes.
Argon2 makes each password attempt expensive, but the access code check is a
string comparison and would otherwise answer as fast as the network allows.

In-memory, not Redis. One process serves one office; a limiter that needs its
own server to protect a single-file SQLite app is the wrong shape. The trade-off
is stated rather than hidden: **counters reset when the process restarts**, so an
attacker who can also restart the app can clear their own lockout. On a machine
where they could do that, they already have the database.

Not a general-purpose limiter. It buys time against online guessing; it is not a
defence against someone with a copy of the disk.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class Limit:
    """`attempts` tries per `window_seconds`, then locked out for `window_seconds`."""

    attempts: int
    window_seconds: float


# Deliberately strict. A person mistyping their own password five times in a
# quarter of an hour is rare; a script trying five hundred is not.
ACCESS_CODE = Limit(attempts=10, window_seconds=15 * 60)
LOGIN = Limit(attempts=10, window_seconds=15 * 60)
SIGNUP = Limit(attempts=5, window_seconds=60 * 60)

# The approval lookup is polled every 15 seconds by a waiting newcomer, so only
# the lookups that find nothing count against it: an address that is unknown or
# suspended is the enumeration signal, and it gets the same allowance as a
# wrong access code. Pending and active answers are free.
APPROVAL_LOOKUP = Limit(attempts=ACCESS_CODE.attempts, window_seconds=ACCESS_CODE.window_seconds)
# Every lookup counts here, whatever it answers. The pending poll stays free
# against the bucket above; this one stops a script from walking a list of
# known addresses with no unknown answers at all.
APPROVAL_LOOKUP_ANY = Limit(attempts=120, window_seconds=15 * 60)

# Changing a password requires the current one, so a stolen or borrowed session
# is a password oracle: guess `current_password` from a logged-in browser as
# fast as argon2 allows. Same shape as LOGIN because it is the same attack,
# keyed on the account rather than the address being tried.
PASSWORD_CHANGE = Limit(attempts=10, window_seconds=15 * 60)

# The limits below are not about guessing. They bound how fast one person can
# make the office spend money or occupy the process, which the budget caps do
# not: a script inside a valid session can commit a month's budget in seconds,
# and the caps only notice once it is gone.

# A render is a deliberate act. Twenty a minute is far past what anyone does by
# hand and well short of what a loop would manage.
RENDER_SUBMIT = Limit(attempts=20, window_seconds=60)

# Every open tab polls this every six seconds while anything is in flight. The
# real protection is the shared throttle in render_service.refresh(); this is
# the backstop for a runaway client, set high enough that a person with several
# tabs open never meets it.
RENDER_REFRESH = Limit(attempts=60, window_seconds=60)

# A depth preview is a local ONNX inference on the request thread, a few seconds
# each. Cached results cost nothing, so a person re-previewing the same view is
# not throttled in practice; the limit only bounds a loop over new images.
DEPTH_PREVIEW = Limit(attempts=30, window_seconds=60)

# 100 MB per upload, hashed and written to disk.
CAD_UPLOAD = Limit(attempts=10, window_seconds=60)

# A preview or a confirm is seconds of local rasterisation on the request
# thread — the most expensive thing this app does without calling fal.ai.
CAD_RENDER = Limit(attempts=30, window_seconds=60)


class RateLimited(Exception):
    """Too many attempts. `retry_after` is whole seconds."""

    def __init__(self, retry_after: float) -> None:
        self.retry_after = int(retry_after) + 1
        super().__init__(
            f"Too many attempts. Try again in {max(1, self.retry_after // 60)} minute(s)."
        )


@dataclass
class _Bucket:
    hits: list[float] = field(default_factory=list)


class RateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        # FastAPI serves from a thread pool and the poller runs on its own
        # thread, so the counters need a lock or two simultaneous attempts
        # can each see a stale count.
        self._lock = threading.Lock()

    def check(self, scope: str, key: str, limit: Limit, *, now: float | None = None) -> None:
        """Raise RateLimited if this key is over its allowance. Does not consume.

        Separate from `hit()` so a *successful* login does not burn an attempt.
        Only failures count, which is what keeps a legitimate user who signs in
        forty times a day from locking themselves out.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._live_hits(scope, key, limit, now)
            if len(hits) >= limit.attempts:
                raise RateLimited(limit.window_seconds - (now - hits[0]))

    def hit(self, scope: str, key: str, limit: Limit, *, now: float | None = None) -> None:
        """Record one failed attempt."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._live_hits(scope, key, limit, now)
            hits.append(now)

    def reset(self, scope: str, key: str) -> None:
        """Clear a key's history — called after a success."""
        with self._lock:
            self._buckets.pop((scope, key), None)

    def _live_hits(self, scope: str, key: str, limit: Limit, now: float) -> list[float]:
        bucket = self._buckets.setdefault((scope, key), _Bucket())
        cutoff = now - limit.window_seconds
        bucket.hits = [t for t in bucket.hits if t > cutoff]
        return bucket.hits

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()


# One shared limiter per process. Module-level rather than injected because the
# limit must be global — a per-request instance would count to one, forever.
limiter = RateLimiter()
