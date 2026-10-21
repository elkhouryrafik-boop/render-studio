"""Two things the office is told, and one of them must also be enforced.

A task in development is listed but held back, and the server refuses it
whatever the greyed control does. The real routing table no longer carries a
held task of its own, so these tests write a small table of their own. They
do not depend on whichever real task is held back next.

The SKP bake estimate is the other half: a 16-minute first parse with no number
on screen reads as a hang, and someone will kill the app mid-bake.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.routing import RoutingError, RoutingTable
from app.services import cad_convert

# Only what RoutingTable needs to load: one allowlisted model, one task held
# back, one task offered, one task disabled on purpose.
TEST_ROUTING = """
allowlist: [fal-ai/test-model]
costs:
  fal-ai/test-model: {}
payloads:
  fal-ai/test-model: {}
tasks:
  held_task:
    label: "Held task"
    description: "A task the office can see but cannot run yet."
    enabled: true
    in_development: true
    in_development_reason: >
      In development. Multi-view consistency is not solved, so this stays held
      back until a person has checked the massing by hand.
    default_engine: proven
    engines:
      proven:
        label: "Proven"
        note: "The only engine in the test table."
        model: fal-ai/test-model
        defaults: {}
  plain_task:
    label: "Plain task"
    description: "A task that is offered."
    enabled: true
    default_engine: proven
    engines:
      proven:
        label: "Proven"
        note: "The only engine in the test table."
        model: fal-ai/test-model
        defaults: {}
  parked_task:
    label: "Parked task"
    enabled: false
    disabled_reason: "Parked for the test."
    engines: {}
"""


@pytest.fixture
def table(tmp_path: Path) -> RoutingTable:
    path = tmp_path / "routing.yaml"
    path.write_text(TEST_ROUTING, encoding="utf-8")
    return RoutingTable.load(path)


# --- listed, but not offered -------------------------------------------------


def test_a_held_task_is_still_listed(table: RoutingTable):
    """Visible on purpose. `enabled: false` would remove it from the list, and
    an absent task reads as "this tool cannot do that" rather than "not yet"."""
    assert "held_task" in dict(table.available_tasks())


def test_a_held_task_carries_a_reason_a_person_can_read(table: RoutingTable):
    reason = table.in_development("held_task")
    assert reason, "a held-back task with no explanation is just a broken button"
    assert len(reason) > 60, "one word is not an explanation"
    # It must say what to do instead, not only what is broken.
    assert "massing" in reason.lower()


def test_resolving_it_is_refused_server_side(table: RoutingTable):
    """The actual gate. A stale tab, a replayed request or a direct API call
    must not be able to spend money on a task the UI shows as unavailable."""
    with pytest.raises(RoutingError) as excinfo:
        table.resolve("held_task")
    assert "not available yet" in str(excinfo.value)


def test_an_explicit_engine_is_refused_too(table: RoutingTable):
    with pytest.raises(RoutingError):
        table.resolve("held_task", "proven")


def test_every_other_task_still_resolves(table: RoutingTable):
    """The block must be surgical. Nothing else may become unreachable."""
    for key, _label in table.available_tasks():
        if table.in_development(key):
            continue
        assert table.resolve(key).model_id


def test_a_task_with_no_flag_is_not_held_back(table: RoutingTable):
    assert table.in_development("plain_task") == ""


def test_a_task_disabled_on_purpose_is_refused_with_its_reason(table: RoutingTable):
    with pytest.raises(RoutingError) as excinfo:
        table.resolve("parked_task")
    assert "disabled on purpose" in str(excinfo.value).lower()
    assert "Parked for the test." in str(excinfo.value)


# --- the one-off conversion wait ---------------------------------------------


def test_the_estimate_matches_the_measured_file(tmp_path: Path):
    """974 seconds for 27.3 MB, measured 2026-08-27 on a test
    model. The estimate exists to put a real number in front of that wait."""
    f = tmp_path / "site.skp"
    f.write_bytes(b"\0" * 27_300_000)
    seconds = cad_convert.estimated_bake_seconds(f)
    assert 900 < seconds < 1050, seconds
    assert round(seconds / 60) == 16


def test_the_estimate_scales_with_size(tmp_path: Path):
    small = tmp_path / "small.skp"
    small.write_bytes(b"\0" * 5_000_000)
    big = tmp_path / "big.skp"
    big.write_bytes(b"\0" * 50_000_000)
    assert cad_convert.estimated_bake_seconds(big) > cad_convert.estimated_bake_seconds(small)


def test_a_tiny_file_still_reports_a_floor_not_zero(tmp_path: Path):
    """Zero would render as "about 0 minutes", which is worse than no estimate."""
    f = tmp_path / "tiny.skp"
    f.write_bytes(b"\0" * 1000)
    assert cad_convert.estimated_bake_seconds(f) >= 30.0


def test_a_missing_file_reports_unknown_rather_than_raising(tmp_path: Path):
    """This is advisory copy. It must never be the thing that breaks an upload."""
    assert cad_convert.estimated_bake_seconds(tmp_path / "nope.skp") == 0.0


# --- experimental tasks: ENABLE_EXPERIMENTAL_TASKS ---------------------------

EXPERIMENTAL = ("entourage", "skp_material_render")


@pytest.fixture
def real_table() -> RoutingTable:
    return RoutingTable.load(Path(__file__).resolve().parent.parent / "routing.yaml")


def test_experimental_tasks_hidden_and_refused_by_default(real_table, monkeypatch):
    monkeypatch.delenv("ENABLE_EXPERIMENTAL_TASKS", raising=False)
    listed = dict(real_table.available_tasks())
    for key in EXPERIMENTAL:
        assert key not in listed
        assert real_table.in_development(key)
        with pytest.raises(RoutingError) as excinfo:
            real_table.resolve(key)
        assert "ENABLE_EXPERIMENTAL_TASKS" in str(excinfo.value)
    assert "material" not in [i["key"] for i in real_table.intents_out()["intents"]]
    assert real_table.resolve("massing_to_render").model_id


def test_experimental_tasks_allowed_when_flag_set(real_table, monkeypatch):
    monkeypatch.setenv("ENABLE_EXPERIMENTAL_TASKS", "1")
    listed = dict(real_table.available_tasks())
    for key in EXPERIMENTAL:
        assert key in listed
        assert real_table.in_development(key) == ""
        assert real_table.resolve(key).model_id
    assert "material" in [i["key"] for i in real_table.intents_out()["intents"]]


def test_flag_value_zero_keeps_them_held(real_table, monkeypatch):
    monkeypatch.setenv("ENABLE_EXPERIMENTAL_TASKS", "0")
    assert "entourage" not in dict(real_table.available_tasks())
