"""Shared vocabulary.

These are the words the database, the services and both front-ends have to agree
on. They live here rather than as bare strings scattered through the codebase,
because a typo in a status string is a silent bug: a row written as "actve" is
simply never returned, and nothing raises.

`StrEnum` means these compare equal to their own text, so they can be handed
straight to SQLite and read back out of a row without conversion ceremony.
"""

from __future__ import annotations

from enum import StrEnum


class UserStatus(StrEnum):
    """An account's standing.

    New accounts land in PENDING and can do nothing at all. The office access
    code gates who may *ask* for an account; approval gates who may spend money.
    """

    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"


class AssetKind(StrEnum):
    """Where an image came from.

    The distinction matters for cleanup and for the UI: uploads are the
    architect's own material, render outputs are ours to regenerate, and masks
    are working files nobody wants to see in a gallery.

    CAD_UPLOAD is the raw DXF/SKP file, kept for re-conversion (a different
    layer selection, a different camera) — it is never itself an image and
    never enters a render. The PNG a conversion produces is ingested
    separately as an ordinary UPLOAD once the architect confirms it.
    """

    UPLOAD = "upload"
    RENDER_OUTPUT = "render_output"
    MASK = "mask"
    CAD_UPLOAD = "cad_upload"
    # A conditioning input produced by a converter — today the true depth pass
    # from an IFC conversion. Real image bytes, but its meaning is "feed me to
    # a specific easycontrols entry", not "show me in a gallery".
    CONTROL = "control"


class ImageRole(StrEnum):
    """What a reference image is *for*.

    Handing a model four pictures tells it nothing. The role decides the order
    images enter `image_urls`, the sentence the prompt builder writes about each
    one, and which image a conditioning engine reads as its control input.

    GEOMETRY is privileged: it is the shape the render must obey, and on
    conditioning engines it is the only role that can be honoured, so it always
    sorts first.
    """

    GEOMETRY = "geometry"
    STYLE = "style"
    MATERIAL = "material"
    CONTEXT = "context"
    ENTOURAGE = "entourage"
    MASK = "mask"

    @property
    def instruction(self) -> str:
        """The phrase the prompt builder uses to introduce this image."""
        return _ROLE_INSTRUCTIONS[self]


# Written as sentences an image model can act on, not as labels. These are the
# difference between "here are three images" and a usable brief.
_ROLE_INSTRUCTIONS: dict[ImageRole, str] = {
    ImageRole.GEOMETRY: (
        "the geometry to obey — its massing, proportions, openings and camera "
        "angle are correct and must not be redesigned"
    ),
    ImageRole.STYLE: (
        "the approved appearance — copy its materials, colour palette, lighting "
        "condition, season and rendering style"
    ),
    ImageRole.MATERIAL: (
        "a material sample — apply this finish where the instruction says to, "
        "and nowhere else"
    ),
    ImageRole.CONTEXT: (
        "the surrounding site — neighbouring buildings, ground and skyline to "
        "sit the building into"
    ),
    ImageRole.ENTOURAGE: (
        "a reference for people, planting and vehicles — match its scale and "
        "character, not its composition"
    ),
    ImageRole.MASK: "the area to change; everything outside it must be left untouched",
}

# The order references are fed to a multi-image endpoint. Geometry first because
# the instruction text refers to images by position ("image 1 is..."), and the
# shape of the building is the thing that must not drift.
ROLE_ORDER: tuple[ImageRole, ...] = (
    ImageRole.GEOMETRY,
    ImageRole.STYLE,
    ImageRole.MATERIAL,
    ImageRole.CONTEXT,
    ImageRole.ENTOURAGE,
)


class JobStatus(StrEnum):
    """Lifecycle of one submission.

    RESERVED is written before fal.ai is called — that ordering is what makes
    the budget guard honest, and it is why a crash mid-submit cannot lose track
    of money already committed.
    """

    RESERVED = "reserved"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUSED = "refused"
    CANCELLED = "cancelled"


# Kept in sync with db.TERMINAL, which predates this module and is what the
# polling loop actually reads.
TERMINAL_STATUSES = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.REFUSED, JobStatus.CANCELLED}
)
