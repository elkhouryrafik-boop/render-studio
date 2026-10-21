"""Deterministic model routing.

The "automatically chooses the best model" requirement, implemented as a lookup
table loaded from routing.yaml. No inference on the common path — the user picks
a task (a button), the table picks the model.

Two hard rules enforced here:
  1. A task must be enabled. Disabled tasks carry a reason and are refused.
  2. The resolved model must be on the allowlist. This is the backstop that stops
     any routing mistake — including a future LLM intent-parser — from reaching
     an endpoint nobody budgeted for.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def experimental_enabled() -> bool:
    """ENABLE_EXPERIMENTAL_TASKS=1 (or true/yes/on) unlocks experimental tasks.

    Read on every call, so tests and operators can flip it without a restart.
    """
    return os.environ.get("ENABLE_EXPERIMENTAL_TASKS", "").strip().lower() in {"1", "true", "yes", "on"}


class RoutingError(Exception):
    """Raised when a task cannot be routed. Never falls through to a default model."""


class IntentError(RoutingError):
    """An intent, view or style the architect asked for does not exist, or the request is incomplete.

    The message is a sentence for the architect; the router sends it as a 422.
    """


@dataclass(frozen=True)
class Route:
    task: str
    label: str
    model_id: str
    engine: str = "proven"
    engine_label: str = ""
    engine_note: str = ""
    conditioning: list[dict[str, Any]] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)
    lora_url: str | None = None


@dataclass(frozen=True)
class Engine:
    """One of the choices offered for a task, with the sentence that explains it."""

    key: str
    label: str
    note: str
    model_id: str


# Keys in a geometry_hold snap that describe it for the screen, not conditioning.
_SNAP_PRESENTATION = frozenset({"value", "label", "hint"})


class RoutingTable:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw
        self._allowlist: set[str] = set(raw.get("allowlist", []))
        self._costs: dict[str, dict[str, Any]] = raw.get("costs", {})
        self._payloads: dict[str, dict[str, Any]] = raw.get("payloads", {})
        self._tasks: dict[str, dict[str, Any]] = raw.get("tasks", {})
        self.priced_on: str = raw.get("priced_on", "unknown")
        self._snaps: list[tuple[int, dict[str, float]]] = self._parse_hold(raw)
        self._snap_info: list[dict[str, Any]] = self._parse_snap_info(raw)
        self._hold_default = int((raw.get("geometry_hold") or {}).get("default", 50))
        self._styles: dict[str, dict[str, str]] = self._parse_styles(raw)
        self._sizes: list[dict[str, Any]] = self._parse_sizes(raw)
        self._intent_defaults: dict[str, Any] = dict(raw.get("intent_defaults") or {})
        self._intents: dict[str, dict[str, Any]] = self._parse_intents(raw)
        self._validate()
        self._validate_intents()

    @classmethod
    def load(cls, path: Path) -> RoutingTable:
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    @staticmethod
    def _parse_hold(raw: dict[str, Any]) -> list[tuple[int, dict[str, float]]]:
        """The `geometry_hold.snaps` block as [(value, {type: scale})] sorted by value."""
        snaps = (raw.get("geometry_hold") or {}).get("snaps") or {}
        parsed: list[tuple[int, dict[str, float]]] = []
        for name, spec in snaps.items():
            try:
                value = int(spec["value"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RoutingError(
                    f"geometry_hold snap {name!r} needs an integer `value` (0-100)."
                ) from exc
            if not 0 <= value <= 100:
                raise RoutingError(f"geometry_hold snap {name!r} value {value} is outside 0-100.")
            scales = {
                str(k): float(v) for k, v in spec.items() if k not in _SNAP_PRESENTATION
            }
            parsed.append((value, scales))
        parsed.sort(key=lambda p: p[0])
        if parsed and len({tuple(sorted(sc)) for _, sc in parsed}) != 1:
            raise RoutingError("geometry_hold snaps must all define the same conditioning types.")
        return parsed

    @staticmethod
    def _parse_snap_info(raw: dict[str, Any]) -> list[dict[str, Any]]:
        """The snaps as the one-screen slider shows them: key, value, label, hint."""
        snaps = (raw.get("geometry_hold") or {}).get("snaps") or {}
        info = [
            {
                "key": str(name),
                "value": int(spec["value"]),
                "label": str(spec.get("label", name)),
                "hint": " ".join(str(spec.get("hint", "")).split()),
            }
            for name, spec in snaps.items()
        ]
        return sorted(info, key=lambda d: d["value"])

    @staticmethod
    def _parse_styles(raw: dict[str, Any]) -> dict[str, dict[str, str]]:
        styles: dict[str, dict[str, str]] = {}
        for key, spec in (raw.get("styles") or {}).items():
            spec = spec or {}
            prompt = " ".join(str(spec.get("prompt", "")).split())
            if not prompt:
                raise RoutingError(f"Style {key!r} needs a non-empty `prompt`.")
            styles[str(key)] = {"label": str(spec.get("label", key)), "prompt": prompt}
        return styles

    @staticmethod
    def _parse_sizes(raw: dict[str, Any]) -> list[dict[str, Any]]:
        sizes: list[dict[str, Any]] = []
        for spec in raw.get("sizes") or []:
            try:
                w, h = int(spec["width"]), int(spec["height"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RoutingError("Each sizes entry needs integer width and height.") from exc
            sizes.append({
                "key": f"{w}x{h}",
                "label": str(spec.get("label", f"{w} × {h}")),
                "width": w,
                "height": h,
            })
        return sizes

    @staticmethod
    def _parse_intents(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
        intents: dict[str, dict[str, Any]] = {}
        for key, spec in (raw.get("intents") or {}).items():
            spec = spec or {}
            views = {
                str(vk): {"label": str(v.get("label", vk)), "task": str(v.get("task", ""))}
                for vk, v in (spec.get("views") or {}).items()
            }
            intents[str(key)] = {
                "label": str(spec.get("label", key)),
                "hint": " ".join(str(spec.get("hint", "")).split()),
                "card": bool(spec.get("card", True)),
                "slider": bool(spec.get("slider", False)),
                "needs_mask": bool(spec.get("needs_mask", False)),
                "needs_line": bool(spec.get("needs_line", False)),
                "max_references": int(spec.get("max_references", 0)),
                "requires": spec.get("requires") or None,
                "engine": str(spec.get("engine") or ""),
                "task": str(spec.get("task") or ""),
                "views": views,
                "default_view": spec.get("default_view"),
                "missing_mask": " ".join(str(spec.get("missing_mask", "")).split()),
                "missing_line": " ".join(str(spec.get("missing_line", "")).split()),
            }
        return intents

    def _validate_intents(self) -> None:
        """Catch a broken intents block at startup. Every name must resolve.

        A table with neither an intents nor a styles block (a minimal table, as
        some tests build) has nothing to validate and is accepted as it is.
        """
        if not self._intents and not self._styles:
            return
        if not self._intents:
            raise RoutingError("routing.yaml has styles but no intents block.")
        style = str(self._intent_defaults.get("style", ""))
        if style not in self._styles:
            raise RoutingError(
                f"intent_defaults.style {style!r} is not one of the styles: {sorted(self._styles)}."
            )
        n_default = int(self._intent_defaults.get("num_images", 0))
        n_max = int(self._intent_defaults.get("num_images_max", 0))
        if not 1 <= n_default <= n_max:
            raise RoutingError("intent_defaults needs 1 <= num_images <= num_images_max.")
        for key, spec in self._intents.items():
            if spec["views"]:
                if spec["task"]:
                    raise RoutingError(f"Intent {key!r} has both views and a task. Pick one.")
                if spec["default_view"] not in spec["views"]:
                    raise RoutingError(
                        f"Intent {key!r} default_view {spec['default_view']!r} is not one of its views."
                    )
                for vk, view in spec["views"].items():
                    self._check_intent_target(
                        f"intent {key!r} view {vk!r}", view["task"], spec["engine"]
                    )
            else:
                if spec["default_view"] is not None:
                    raise RoutingError(f"Intent {key!r} names a default_view but has no views.")
                self._check_intent_target(f"intent {key!r}", spec["task"], spec["engine"])

    def _check_intent_target(self, where: str, task: str, engine: str) -> None:
        spec = self._tasks.get(task)
        if spec is None:
            raise RoutingError(f"{where} names task {task!r}, which is not in tasks:.")
        if engine not in (spec.get("engines") or {}):
            raise RoutingError(
                f"{where} names engine {engine!r}, which task {task!r} does not offer. "
                f"Its engines: {sorted(spec.get('engines') or {})}."
            )

    def scales_for_hold(self, value: int) -> dict[str, float]:
        """Conditioning scales for a 0-100 geometry hold, by linear interpolation.

        0 = loose, 50 = hold (today's shipped scales), 100 = exact, with the
        points defined in routing.yaml. Out-of-range values are clamped.
        """
        if not self._snaps:
            raise RoutingError("routing.yaml has no geometry_hold.snaps block.")
        v = max(0, min(100, int(value)))
        pts = self._snaps
        if v <= pts[0][0]:
            return dict(pts[0][1])
        for (v0, s0), (v1, s1) in itertools.pairwise(pts):
            if v <= v1:
                t = (v - v0) / (v1 - v0)
                return {k: s0[k] + (s1[k] - s0[k]) * t for k in s0}
        return dict(pts[-1][1])

    def _validate(self) -> None:
        """Catch a broken table at startup, not at submit time."""
        if not self._allowlist:
            raise RoutingError("routing.yaml has an empty allowlist — refusing to start.")
        for model_id in self._allowlist:
            if model_id not in self._costs:
                raise RoutingError(
                    f"Model {model_id!r} is allowlisted but has no cost entry. "
                    "Every allowlisted model needs a price or the budget guard is blind."
                )
            if model_id not in self._payloads:
                raise RoutingError(
                    f"Model {model_id!r} is allowlisted but has no payload entry. "
                    "These endpoints do not share a request shape — declare it "
                    "in routing.yaml rather than letting the gateway guess."
                )
        for name, spec in self._tasks.items():
            if not spec.get("enabled", False):
                continue
            engines = spec.get("engines") or {}
            if not engines:
                raise RoutingError(f"Task {name!r} is enabled but offers no engines.")
            for key, eng in engines.items():
                model_id = eng.get("model")
                if model_id not in self._allowlist:
                    raise RoutingError(
                        f"Task {name!r} engine {key!r} routes to {model_id!r}, "
                        "which is not on the allowlist."
                    )
            default = spec.get("default_engine")
            if default not in engines:
                raise RoutingError(
                    f"Task {name!r} has default_engine {default!r}, "
                    f"which is not one of its engines: {sorted(engines)}."
                )

    # --- Public API ---------------------------------------------------------

    def resolve_intent(self, intent: str, view: str | None = None) -> tuple[str, str]:
        """(task_key, engine_key) for an intent, and for a view where the intent has views.

        A view left out takes the intent's default_view. Asking for a view the
        intent does not have, or a view on an intent that takes none, is refused
        with a sentence.
        """
        spec = self._intent_spec(intent)
        if spec["views"]:
            key = view or spec["default_view"]
            if key not in spec["views"]:
                raise IntentError(
                    f"{spec['label']} does not work on a {view!r} view. "
                    f"Use one of: {', '.join(spec['views'])}."
                )
            return spec["views"][key]["task"], spec["engine"]
        if view:
            raise IntentError(f"{spec['label']} takes no view. Leave the view out.")
        return spec["task"], spec["engine"]

    def intent_num_images(self, intent: str, view: str | None = None) -> int:
        """How many results an intent asks for: the default, or 1 if its engine returns one image."""
        task, engine = self.resolve_intent(intent, view)
        model_id = self._tasks[task]["engines"][engine]["model"]
        if not self.supports_num_images(model_id):
            return 1
        return self.num_images_default

    def intent_spec(self, intent: str) -> dict[str, Any]:
        """The normalised intent, for the router and the reference rules."""
        return dict(self._intent_spec(intent))

    def _intent_spec(self, intent: str) -> dict[str, Any]:
        spec = self._intents.get(intent)
        if spec is None:
            raise IntentError(f"Unknown intent {intent!r}. Known: {sorted(self._intents)}.")
        return spec

    @property
    def default_style(self) -> str:
        return str(self._intent_defaults["style"])

    @property
    def num_images_default(self) -> int:
        return int(self._intent_defaults["num_images"])

    @property
    def num_images_max(self) -> int:
        return int(self._intent_defaults["num_images_max"])

    @property
    def hold_default(self) -> int:
        return self._hold_default

    def takes_geometry_hold(self, task: str) -> bool:
        """True when the task is a perspective view that the geometry hold slider moves."""
        return bool((self._tasks.get(task) or {}).get("geometry_hold"))

    def compose_prompt(self, task: str, style: str | None, line: str | None) -> str:
        """`<task scaffold> <style prompt> <line>`, trimmed, with single spaces.

        Any part may be empty: a task with no scaffold, no style (enlarge), or
        no line. Whitespace inside each part is collapsed, so a line pasted with
        newlines reads as one sentence.

        Two per-task switches in routing.yaml shape this: a scaffold containing
        `{line}` takes the architect's words inside the sentence instead of
        after it, and `prompt_style: false` leaves the look preset out. Both
        exist for the masked repair: with "photoreal architectural photograph"
        appended and the words tacked on the end, flux-pro/v1/fill painted a
        framed photograph with text inside the mask (paid smoke, 2026-10-10).
        """
        if task not in self._tasks:
            raise RoutingError(f"Unknown task {task!r}.")
        spec = self._tasks[task]
        scaffold = str(spec.get("prompt_scaffold") or "")
        words = " ".join((line or "").split())
        style_text = ""
        if style is not None and spec.get("prompt_style", True):
            if style not in self._styles:
                raise IntentError(f"Unknown style {style!r}. Known: {sorted(self._styles)}.")
            style_text = self._styles[style]["prompt"]
        elif style is not None and style not in self._styles:
            raise IntentError(f"Unknown style {style!r}. Known: {sorted(self._styles)}.")
        if "{line}" in scaffold:
            scaffold = scaffold.replace("{line}", words or "the same as its surroundings")
            words = ""
        parts = (scaffold, style_text, words)
        return " ".join(" ".join(p.split()) for p in parts if p and p.strip())

    def intents_out(self, *, depth_available: bool = False) -> dict[str, Any]:
        """The payload of GET /api/intents, as the contract shows it."""
        intents = []
        for key, spec in self._intents.items():
            target = self._tasks.get(spec["task"] or "") if spec.get("task") else None
            if target is not None and self._hidden_experimental(target):
                continue
            intents.append({
                "key": key,
                "label": spec["label"],
                "hint": spec["hint"],
                "card": spec["card"],
                "slider": spec["slider"],
                "needs_mask": spec["needs_mask"],
                "needs_line": spec["needs_line"],
                "max_references": spec["max_references"],
                "requires": spec["requires"],
                "views": [{"key": vk, "label": v["label"]} for vk, v in spec["views"].items()],
                "default_view": spec["default_view"],
            })
        return {
            "intents": intents,
            "styles": [{"key": k, "label": v["label"]} for k, v in self._styles.items()],
            "default_style": self.default_style,
            "geometry_hold": {"default": self._hold_default, "snaps": self._snap_info},
            "sizes": self._sizes,
            "depth_available": bool(depth_available),
            "num_images_default": self.num_images_default,
            "num_images_max": self.num_images_max,
        }

    def available_tasks(self) -> list[tuple[str, str]]:
        """(task_name, label) for every enabled task — drives the UI buttons.

        Experimental tasks are left out unless ENABLE_EXPERIMENTAL_TASKS=1.
        """
        return [
            (name, spec.get("label", name))
            for name, spec in self._tasks.items()
            if spec.get("enabled", False) and not self._hidden_experimental(spec)
        ]

    @staticmethod
    def _hidden_experimental(spec: dict[str, Any]) -> bool:
        """True for an `experimental: true` task while the env flag is off."""
        return bool(spec.get("experimental")) and not experimental_enabled()

    def engines(self, task: str) -> list[Engine]:
        """The choices offered for a task, in the order the UI should show them."""
        spec = self._enabled_spec(task)
        return [
            Engine(
                key=key,
                label=eng.get("label", key),
                note=" ".join((eng.get("note") or "").split()),
                model_id=eng["model"],
            )
            for key, eng in (spec.get("engines") or {}).items()
        ]

    def default_engine(self, task: str) -> str:
        return str(self._enabled_spec(task).get("default_engine", ""))

    def _enabled_spec(self, task: str) -> dict[str, Any]:
        spec = self._tasks.get(task)
        if spec is None:
            raise RoutingError(f"Unknown task {task!r}. Known: {sorted(self._tasks)}")
        if not spec.get("enabled", False):
            reason = spec.get("disabled_reason", "no reason recorded").strip()
            raise RoutingError(f"Task {task!r} is disabled on purpose.\n\n{reason}")
        return spec

    def in_development(self, task: str) -> str:
        """Why this task is listed but not offered, or "" if it is offered.

        Distinct from `enabled: false`, which hides a task outright. A task in
        development is one the office will look for and should be told about:
        it shows in the list, greyed, carrying this sentence.
        """
        spec = self._tasks.get(task) or {}
        if not spec.get("in_development"):
            return ""
        if spec.get("experimental") and experimental_enabled():
            return ""
        reason = " ".join((spec.get("in_development_reason") or "").split())
        return reason or "In development. Not available yet."

    def resolve(self, task: str, engine: str | None = None) -> Route:
        """Resolve a task to a model. Falls back to the task's default engine."""
        spec = self._enabled_spec(task)
        # Enforced here rather than only in the UI. Every path that spends
        # money — submit, estimate, continue-from — goes through resolve(), so
        # a stale browser tab, a replayed request or a direct API call is
        # refused by the same rule that greys the control.
        held = self.in_development(task)
        if held:
            raise RoutingError(f"Task {task!r} is not available yet.\n\n{held}")
        engines = spec.get("engines") or {}

        key = engine or spec.get("default_engine")
        eng = engines.get(str(key))
        if eng is None:
            raise RoutingError(
                f"Task {task!r} has no engine {key!r}. Known: {sorted(engines)}."
            )

        model_id = eng.get("model")
        if model_id not in self._allowlist:
            # Defence in depth — _validate() should have caught this at startup.
            raise RoutingError(f"Task {task!r} routes to non-allowlisted model {model_id!r}.")

        return Route(
            task=task,
            label=spec.get("label", task),
            model_id=model_id,
            engine=str(key),
            engine_label=eng.get("label", str(key)),
            engine_note=" ".join((eng.get("note") or "").split()),
            conditioning=eng.get("conditioning", []) or [],
            defaults=eng.get("defaults", {}) or {},
            # The LoRA belongs to the task, not to one engine.
            lora_url=spec.get("lora_url"),
        )

    def is_allowed(self, model_id: str) -> bool:
        return model_id in self._allowlist

    def payload_spec(self, model_id: str) -> dict[str, Any]:
        """How this endpoint wants its request shaped. Never guesses a default."""
        spec = self._payloads.get(model_id)
        if spec is None:
            raise RoutingError(
                f"No payload entry for {model_id!r} — refusing to build a request. "
                "Add its shape to routing.yaml, read off the endpoint's schema page."
            )
        return spec

    def supports_num_images(self, model_id: str) -> bool:
        """False for endpoints that return exactly one image per call.

        The UI needs this to stop offering variants that would be silently
        dropped, and the estimate needs it so the cost matches the request.
        """
        return bool(self.payload_spec(model_id).get("num_images", True))

    def payload_verified(self, model_id: str) -> str:
        return str(self.payload_spec(model_id).get("verified", "UNVERIFIED"))

    def estimate_usd(self, model_id: str, width: int, height: int, num_images: int = 1) -> float:
        """Pre-flight cost estimate. Advisory — fal.ai does the real billing.

        Deliberately does NOT fall back to a default price for an unknown model:
        an unpriced model must be refused, not guessed at.
        """
        cost = self._costs.get(model_id)
        if cost is None:
            raise RoutingError(
                f"No cost entry for {model_id!r} — refusing to estimate. "
                "Add it to routing.yaml before using this model."
            )
        unit = cost["unit"]
        usd = float(cost["usd"])
        if unit == "per_image":
            return usd * num_images
        if unit == "per_megapixel":
            megapixels = (width * height) / 1_000_000
            return usd * megapixels * num_images
        raise RoutingError(f"Unknown cost unit {unit!r} for {model_id!r}.")
