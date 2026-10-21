"""A zero-spend stand-in for the fal.ai client.

`app/fal_gateway.py` routes every call through `_client(settings)`. When
`RENDER_ENGINE=fake` that is an instance of `FakeFalClient` rather than the
`fal_client` module, so the whole job lifecycle (reserve, submit, poll, save,
serve) runs with no network and no key. Nothing here contacts fal.ai.

The surface matches what the gateway uses: `upload_file`, `submit`, `status`,
`result`. Status is driven by a clock: Queued for the first half of the delay,
InProgress until the delay, then Completed. The clock is injectable so tests
can step it exactly.

State lives in memory on the instance. A restarted server forgets its requests,
which the gateway already treats as expired (NOT_FOUND), so nothing hangs.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import fal_client
from PIL import Image, ImageDraw, ImageFont

DEFAULT_SIZE = (1024, 768)
PRESET_SIZES: dict[str, tuple[int, int]] = {
    "square_hd": (1024, 1024),
    "landscape_4_3": (1280, 960),
    "portrait_4_3": (960, 1280),
}


class FakeNotFound(Exception):
    """An unknown request id. The text carries NOT_FOUND, as fal's 404 does."""


@dataclass(frozen=True)
class FakeHandle:
    request_id: str


@dataclass
class _Request:
    application: str
    arguments: dict[str, Any]
    submitted_at: float


@dataclass
class FakeFalClient:
    delay_s: float = 4.0
    clock: Callable[[], float] | None = None
    # Set True to make the next status call report Completed and the result for
    # that request come back with no images. The gateway then ends the job
    # `failed` through its existing "no image URL" path. Cleared once used.
    fail_next: bool = False
    _requests: dict[str, _Request] = field(default_factory=dict, repr=False)
    _doomed: set[str] = field(default_factory=set, repr=False)
    # Requests the gateway cancelled. Their status is NOT_FOUND from then on, as
    # fal's is for a request it has dropped.
    cancelled: set[str] = field(default_factory=set, repr=False)
    _counter: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _now(self) -> float:
        return self.clock() if self.clock is not None else time.monotonic()

    def upload_file(self, path: str | Path) -> str:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return f"fake://upload/{digest}"

    def submit(self, application: str, arguments: dict[str, Any] | None = None, **_kw: Any) -> FakeHandle:
        with self._lock:
            self._counter += 1
            request_id = f"fake-{self._counter}"
            self._requests[request_id] = _Request(
                application=application,
                arguments=dict(arguments or {}),
                submitted_at=self._now(),
            )
        return FakeHandle(request_id=request_id)

    def cancel(self, application: str, request_id: str) -> None:
        with self._lock:
            self._get(request_id)
            self.cancelled.add(request_id)

    def status(self, application: str, request_id: str, *, with_logs: bool = False) -> Any:
        with self._lock:
            req = self._get(request_id)
            if request_id in self.cancelled:
                raise FakeNotFound(f"NOT_FOUND: fake request {request_id} was cancelled")
            if self.fail_next:
                self.fail_next = False
                self._doomed.add(request_id)
            if request_id in self._doomed:
                return fal_client.Completed(logs=[], metrics={})
            elapsed = self._now() - req.submitted_at
        if elapsed < self.delay_s / 2:
            return fal_client.Queued(position=0)
        if elapsed < self.delay_s:
            return fal_client.InProgress(logs=[])
        return fal_client.Completed(logs=[], metrics={})

    def result(self, application: str, request_id: str) -> dict[str, Any]:
        with self._lock:
            req = self._get(request_id)
            if request_id in self.cancelled:
                raise FakeNotFound(f"NOT_FOUND: fake request {request_id} was cancelled")
            if request_id in self._doomed:
                self._doomed.discard(request_id)
                return {"images": []}
            arguments = req.arguments
        width, height = _requested_size(arguments)
        count = int(arguments.get("num_images", 1) or 1)
        return {
            "images": [
                {"url": f"fake://result/{request_id}/{i}", "width": width, "height": height}
                for i in range(count)
            ],
            "seed": 1234,
        }

    def _get(self, request_id: str) -> _Request:
        req = self._requests.get(request_id)
        if req is None:
            raise FakeNotFound(f"NOT_FOUND: fake request {request_id} is unknown or expired")
        return req


def _requested_size(arguments: dict[str, Any]) -> tuple[int, int]:
    size = arguments.get("image_size")
    if isinstance(size, dict) and size.get("width") and size.get("height"):
        return int(size["width"]), int(size["height"])
    if isinstance(size, str) and size in PRESET_SIZES:
        return PRESET_SIZES[size]
    if arguments.get("width") and arguments.get("height"):
        return int(arguments["width"]), int(arguments["height"])
    return DEFAULT_SIZE


_TOP = (242, 238, 230)  # warm paper grey, top of the gradient
_BOTTOM = (221, 216, 205)  # and bottom
_INK = (88, 84, 78)
_FRAME = (168, 162, 152)
_FRAME_INSET = 24


def placeholder_png(*, width: int, height: int, request_id: str, index: int) -> bytes:
    """A labelled stand-in image, at the requested size, as PNG bytes.

    Paper-grey vertical gradient, a thin frame inset 24 px, the label
    "FAKE <request_id> #<index>" top-left and the size bottom-right. Uses
    Pillow's default font, so there are no font files to ship.
    """
    img = Image.new("RGB", (width, height), _TOP)
    draw = ImageDraw.Draw(img)
    span = max(height - 1, 1)
    for y in range(height):
        t = y / span
        colour = tuple(round(a + (b - a) * t) for a, b in zip(_TOP, _BOTTOM))
        draw.line([(0, y), (width, y)], fill=colour)
    if width > 2 * _FRAME_INSET and height > 2 * _FRAME_INSET:
        draw.rectangle(
            [_FRAME_INSET, _FRAME_INSET, width - 1 - _FRAME_INSET, height - 1 - _FRAME_INSET],
            outline=_FRAME,
            width=2,
        )
    font = ImageFont.load_default()
    label = f"FAKE  {request_id}  #{index}"
    size_label = f"{width} x {height}"
    margin = _FRAME_INSET * 2
    draw.text((margin, margin), label, fill=_INK, font=font)
    left, top, right, bottom = draw.textbbox((0, 0), size_label, font=font)
    draw.text(
        (width - margin - (right - left), height - margin - (bottom - top)),
        size_label,
        fill=_INK,
        font=font,
    )
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


__all__ = [
    "DEFAULT_SIZE",
    "PRESET_SIZES",
    "FakeFalClient",
    "FakeHandle",
    "FakeNotFound",
    "placeholder_png",
]
