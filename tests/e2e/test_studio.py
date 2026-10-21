"""The Studio, end to end, in headless Chromium against the fake engine.

The steps run in file order on one signed-in page: gate, view, render, plan
view, region edit, forced failure and retry, lightbox, the admin stop switch and
the phone width. Each step saves a screenshot to examples/out/e2e/. Selectors are
roles and visible text the brief defines; no test ids are needed. A step that
fails stops the journey, because every later step depends on the sheets before it.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect, sync_playwright

from tests.e2e.conftest import ACCESS_CODE, ADMIN_EMAIL, ADMIN_PASSWORD, REPO

SHOTS = REPO / "examples" / "out" / "e2e"
TILES = ".sheet-panel .grid article.tile"
FAILED_SENTENCE = "The render came back without an image. Retry sends it again."
RAW_STATUS = re.compile(r"\b(in_progress|completed|reserved)\b")

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def ui(e2e_base_url: str) -> Iterator[Page]:
    """One signed-out browser page at 1440x900, closed after the module."""
    SHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.set_default_timeout(15_000)
        page.goto(e2e_base_url)
        yield page
        context.close()
        browser.close()


@pytest.fixture(scope="module")
def png_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from PIL import Image, ImageDraw

    path = tmp_path_factory.mktemp("upload") / "view.png"
    img = Image.new("RGB", (1024, 768), (226, 230, 236))
    draw = ImageDraw.Draw(img)
    draw.rectangle((180, 240, 840, 700), fill=(190, 150, 110), outline=(60, 50, 40), width=6)
    draw.polygon([(160, 240), (512, 110), (860, 240)], fill=(120, 60, 50))
    img.save(path, "PNG")
    return path


def shot(page: Page, name: str) -> None:
    """Save a viewport screenshot; fall back to JPEG q70 if the PNG is over 300 KB."""
    SHOTS.mkdir(parents=True, exist_ok=True)
    target = SHOTS / f"{name}.png"
    page.screenshot(path=str(target))
    if target.stat().st_size > 300 * 1024:
        target.unlink()
        page.screenshot(path=str(SHOTS / f"{name}.jpg"), type="jpeg", quality=70)


def api(page: Page, base: str) -> list[dict]:
    return page.request.get(f"{base}/api/renders").json()


def wait_job(page: Page, base: str, after_id: int, outputs: int, status: str = "completed") -> dict:
    """Poll the API until a job newer than `after_id` reaches `status`."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        for job in api(page, base):
            enough = status != "completed" or len(job["outputs"]) >= outputs
            if job["id"] > after_id and job["status"] == status and enough:
                return job
        time.sleep(0.5)
    raise AssertionError(
        f"No job after {after_id} reached {status} with {outputs} outputs in 60 s."
    )


def newest_id(page: Page, base: str) -> int:
    return max((j["id"] for j in api(page, base)), default=0)


def expect_tiles(page: Page, count: int) -> None:
    """Wait until the current sheet shows `count` images that actually loaded."""
    page.wait_for_function(
        """([sel, n]) => {
            const imgs = [...document.querySelectorAll(sel + ' img')];
            return imgs.length === n && imgs.every((i) => i.complete && i.naturalWidth > 0);
        }""",
        arg=[TILES, count],
        timeout=60_000,
    )


def expect_no_raw_copy(page: Page) -> None:
    text = page.inner_text("body")
    assert "fal-ai/" not in page.content(), "A model id reached the page."
    assert not RAW_STATUS.search(text), "A raw status word reached the page."


def render(page: Page) -> None:
    page.get_by_role("button", name="Render", exact=True).click()


def test_a_gate_signs_in_the_bootstrap_admin(ui: Page, e2e_base_url: str) -> None:
    ui.get_by_role("button", name="Create an account").click()
    ui.get_by_label("Office access code").fill(ACCESS_CODE)
    ui.get_by_role("button", name="Continue").click()
    ui.get_by_label("Name", exact=True).fill("E2E Admin")
    ui.get_by_label("Email", exact=True).fill(ADMIN_EMAIL)
    ui.get_by_label("Password", exact=True).fill(ADMIN_PASSWORD)
    ui.get_by_role("button", name="Create account").click()
    expect(ui.get_by_role("button", name="Render", exact=True)).to_be_visible()
    expect(ui.get_by_text("Drop a view", exact=False).first).to_be_visible()
    shot(ui, "01-gate")


def test_b_drop_a_view_shows_the_depth_chip(ui: Page, png_path: Path) -> None:
    """The view loads either way; which chip appears depends on the depth model.

    depth-anything-v2-small is a large optional download (`scripts/fetch_depth_model.py`),
    so a fresh checkout runs in edges-only mode. Both states are correct - asserting only
    the estimated-depth chip made this suite fail 1 of 9 on a clean clone, and made the
    README's "9 passed" claim unreproducible.
    """
    ui.locator("input[type=file][aria-hidden='true']").set_input_files(str(png_path))
    chip = ui.get_by_text(re.compile(r"^(Depth: estimated|Edges only)$"))
    expect(chip).to_be_visible(timeout=30_000)
    expect(ui.get_by_alt_text("The view")).to_be_visible()
    shot(ui, "02-view")


def test_c_make_it_real_returns_four_sheets(ui: Page, e2e_base_url: str) -> None:
    ui.get_by_role("radio", name=re.compile(r"^Make it real")).click()
    ui.get_by_role("radio", name="Model view", exact=True).click()
    expect(ui.get_by_role("slider", name="How closely to hold the drawing")).to_have_attribute(
        "aria-valuetext", "Hold"
    )
    ui.get_by_role("radio", name="Daylight", exact=True).click()
    before = newest_id(ui, e2e_base_url)
    render(ui)
    job = wait_job(ui, e2e_base_url, before, outputs=4)
    expect(ui.get_by_role("heading", name=re.compile(rf"Sheet {job['id']:04d}"))).to_be_visible()
    expect_tiles(ui, 4)
    expect_no_raw_copy(ui)
    shot(ui, "03-sheet-four-tiles")


def test_d_plan_view_hides_the_slider(ui: Page) -> None:
    ui.get_by_role("radio", name="Plan", exact=True).click()
    expect(ui.get_by_role("slider", name="How closely to hold the drawing")).to_have_count(0)
    expect(ui.get_by_text("Drawings keep their tuned line hold.")).to_be_visible()
    shot(ui, "04-plan-view")


def test_e_change_a_region_on_tile_one(ui: Page, e2e_base_url: str) -> None:
    # Hover opens the tile actions; the first tile is the sheet from step c.
    tile = ui.locator(TILES).first
    tile.hover()
    tile.get_by_role("button", name="Change a region, image 1").click()

    panel = ui.get_by_label("Mark the region")
    expect(panel).to_be_visible()
    canvas = panel.locator("canvas")
    box = canvas.bounding_box()
    assert box is not None
    x0, y0 = box["x"] + box["width"] * 0.3, box["y"] + box["height"] * 0.35
    ui.mouse.move(x0, y0)
    ui.mouse.down()
    for dx, dy in [(60, 10), (120, 30), (160, 50), (120, 80)]:
        ui.mouse.move(x0 + dx, y0 + dy, steps=6)
    ui.mouse.up()
    shot(ui, "05-region-mask")

    panel.get_by_role("button", name="Done marking").click()
    expect(ui.get_by_label("What the marked part becomes")).to_be_visible()
    ui.get_by_label("What the marked part becomes").fill("a slate roof with two dormers")

    before = newest_id(ui, e2e_base_url)
    render(ui)
    job = wait_job(ui, e2e_base_url, before, outputs=1)
    expect(ui.get_by_role("heading", name=re.compile(rf"Sheet {job['id']:04d}"))).to_be_visible()
    expect_no_raw_copy(ui)
    shot(ui, "06-region-sheet")


def test_f_forced_failure_shows_a_sentence_and_retry(ui: Page, e2e_base_url: str) -> None:
    assert ui.request.post(f"{e2e_base_url}/api/dev/fail-next").status == 204
    ui.get_by_role("radio", name=re.compile(r"^Make it real")).click()

    before = newest_id(ui, e2e_base_url)
    render(ui)
    failed = wait_job(ui, e2e_base_url, before, outputs=0, status="failed")

    alert = ui.get_by_role("alert")
    expect(alert).to_contain_text(FAILED_SENTENCE)
    expect(alert.get_by_role("button", name="Retry")).to_be_visible()
    expect(ui.get_by_text("failed", exact=True).first).to_be_visible()
    shot(ui, "07-failed-with-retry")

    # Open the sheet before it so the failed job drops to the queue, where a
    # failed row shows the same sentence and its own Retry.
    ui.get_by_role("button", name=re.compile(r"^Sheet 0002,")).click()
    row = ui.locator(".queue-row").filter(has_text=FAILED_SENTENCE)
    expect(row).to_be_visible()
    expect(row.get_by_role("button", name="Retry")).to_be_visible()
    shot(ui, "07b-failed-queue-row")

    row.get_by_role("button", name="Retry").click()
    retried = wait_job(ui, e2e_base_url, failed["id"], outputs=4)
    expect(
        ui.get_by_role("heading", name=re.compile(rf"Sheet {retried['id']:04d}"))
    ).to_be_visible()
    expect_tiles(ui, 4)
    expect_no_raw_copy(ui)
    shot(ui, "08-retry-four-tiles")


def test_g_key_two_opens_the_lightbox_and_escape_closes_it(ui: Page) -> None:
    ui.evaluate("document.activeElement instanceof HTMLElement && document.activeElement.blur()")
    ui.keyboard.press("2")
    dialog = ui.get_by_role("dialog")
    expect(dialog).to_be_visible()
    expect(dialog).to_have_attribute("aria-label", re.compile(r"image 2$"))
    shot(ui, "09-lightbox")
    ui.keyboard.press("Escape")
    expect(ui.get_by_role("dialog")).to_have_count(0)


def test_h_admin_stop_all_spending_on_and_off(ui: Page) -> None:
    ui.get_by_role("button", name=re.compile(r"^Accounts")).click()
    # The switch is named by its own state sentence, so it is found by its panel.
    switch = ui.locator(".stop-panel").get_by_role("switch")
    expect(switch).to_have_attribute("aria-checked", "false")
    switch.click()
    expect(switch).to_have_attribute("aria-checked", "true")
    expect(ui.get_by_text("Spending is stopped for everyone.")).to_be_visible()
    shot(ui, "10-stop-spending-on")
    switch.click()
    expect(switch).to_have_attribute("aria-checked", "false")
    expect(ui.get_by_text("Spending is open again.")).to_be_visible()
    ui.get_by_role("button", name="Studio", exact=True).click()


def test_i_phone_width_stacks_the_controls(ui: Page) -> None:
    ui.set_viewport_size({"width": 390, "height": 844})
    ui.reload()
    expect(ui.get_by_role("button", name="Render", exact=True)).to_be_visible()
    notes = ui.locator("form.controls section.note")
    expect(notes.first).to_be_visible()
    boxes = [notes.nth(i).bounding_box() for i in range(notes.count())]
    assert len(boxes) >= 3
    assert all(b is not None for b in boxes)
    lefts = [b["x"] for b in boxes]
    tops = [b["y"] for b in boxes]
    assert max(lefts) - min(lefts) < 2, "Notes are not stacked in one column."
    assert tops == sorted(tops), "Notes are out of order on the phone."
    assert ui.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    shot(ui, "11-phone")
