"""Turning what the architect asked for into a render the service can submit.

An intent (make it real, change something, change a region, material render,
enlarge) is resolved here to a task, an engine, a composed prompt, a result
count and a geometry hold. The routers call this and then hand the answer to
`render_service`, which owns spending.

Reference rules are checked on the roles of the pictures, not on their asset
ids, so a plan can be made before anything is uploaded or converted. Every
refusal is an `IntentError` carrying a sentence for the architect.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.models import ImageRole
from app.routing import IntentError, RoutingTable

# Intents that take exactly one picture, the view being worked on.
_ONE_PICTURE = frozenset({"make_real", "material", "region", "enlarge"})


@dataclass(frozen=True)
class IntentPlan:
    """The resolved request. `meta` is what the job records in params_json."""

    intent: str
    view: str | None
    style: str | None
    line: str | None
    task: str
    engine: str
    prompt: str
    num_images: int
    geometry_hold: int | None

    @property
    def meta(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "view": self.view,
            "style": self.style,
            "line": self.line,
            "geometry_hold": self.geometry_hold,
        }


def _clean_line(line: str | None) -> str | None:
    text = " ".join((line or "").split())
    return text or None


def _check_roles(table: RoutingTable, intent: str, label: str, roles: Sequence[ImageRole]) -> None:
    """Refuse a set of pictures the intent cannot use. Sentences, not codes.

    Every intent in routing.yaml has a case below. An intent added to
    routing.yaml without one would be accepted with no picture rules, so a new
    intent needs a case here before it ships.
    """
    if intent == "change":
        if not roles:
            raise IntentError("Change something needs the picture to change. Attach it first.")
        if roles[0] != ImageRole.GEOMETRY:
            raise IntentError("The picture to change must be sent as the geometry reference.")
        if any(role != ImageRole.STYLE for role in roles[1:]):
            raise IntentError("The extra pictures for change something are style references.")
        limit = table.intent_spec(intent)["max_references"]
        if len(roles) - 1 > limit:
            raise IntentError(
                f"Change something takes up to {limit} reference pictures besides the base."
            )
        return

    if intent in _ONE_PICTURE:
        if not roles:
            raise IntentError(f"{label} needs a picture. Attach the one to work on.")
        if len(roles) > 1:
            raise IntentError(f"{label} takes one picture. Remove the extra ones.")
        if roles[0] != ImageRole.GEOMETRY:
            raise IntentError("The picture to work on must be sent as the geometry reference.")


def intent_count(table: RoutingTable, intent: str, view: str | None, requested: int | None) -> int:
    """How many results to ask for: the intent's count, or the architect's, never above the engine's.

    Left out, it is what the intent asks for (four, or one on an engine that
    returns one). Asked for, it must be one to the office maximum, and is then
    capped by what the engine can return, so enlarge is always one.
    """
    cap = table.intent_num_images(intent, view)
    if requested is None:
        return cap
    if not 1 <= requested <= table.num_images_max:
        raise IntentError(f"Ask for one to {table.num_images_max} results.")
    return min(requested, cap)


def plan_intent(
    table: RoutingTable,
    *,
    intent: str,
    view: str | None = None,
    style: str | None = None,
    line: str | None = None,
    num_images: int | None = None,
    geometry_hold: int | None = None,
    roles: Sequence[ImageRole] = (),
    has_mask: bool = False,
) -> IntentPlan:
    """Resolve an intent into a task, engine, prompt and count.

    `roles` are the roles of the pictures the render will carry, in order, with
    the base first (the view for submit; the parent's output for continue).
    `geometry_hold` is honoured only when the intent has a slider; otherwise it
    is dropped. `enlarge` ignores style and line and always returns one image.
    """
    spec = table.intent_spec(intent)
    label = spec["label"]
    task, engine = table.resolve_intent(intent, view)
    resolved_view = (view or spec["default_view"]) if spec["views"] else None

    _check_roles(table, intent, label, list(roles))

    if spec["needs_mask"] and not has_mask:
        raise IntentError(spec["missing_mask"] or f"{label} needs a marked region.")
    if has_mask and not spec["needs_mask"]:
        raise IntentError(f"{label} takes no mask. Remove the mask.")

    if intent == "enlarge":
        style_key: str | None = None
        line_text: str | None = None
    else:
        style_key = style or table.default_style
        line_text = _clean_line(line)
        if spec["needs_line"] and not line_text:
            raise IntentError(spec["missing_line"] or f"{label} needs a line of words.")

    count = intent_count(table, intent, view, num_images)

    hold: int | None = None
    if spec["slider"] and (geometry_hold is not None or table.takes_geometry_hold(task)):
        requested = table.hold_default if geometry_hold is None else geometry_hold
        hold = max(0, min(100, int(requested)))

    prompt = table.compose_prompt(task, style_key, line_text)
    if not prompt:
        # Only reachable if routing.yaml gives the task neither a scaffold nor a style.
        raise IntentError(f"{label} has no prompt to send. Check routing.yaml.")

    return IntentPlan(
        intent=intent,
        view=resolved_view,
        style=style_key,
        line=line_text,
        task=task,
        engine=engine,
        prompt=prompt,
        num_images=count,
        geometry_hold=hold,
    )


def intent_meta_from_params(params: dict[str, Any]) -> dict[str, Any]:
    """The intent fields a job recorded, in the shape `plan_intent` produced them.

    Empty for a job that was not made from an intent, so a legacy retry stays
    a legacy retry.
    """
    if not params.get("intent"):
        return {}
    return {
        "intent": params.get("intent"),
        "view": params.get("view"),
        "style": params.get("style"),
        "line": params.get("line"),
        "geometry_hold": params.get("geometry_hold"),
    }
