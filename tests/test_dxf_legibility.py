"""Every colour in a rasterized DXF must survive the move to a white page.

ACI colours are chosen against AutoCAD's black model space. On white, three of
the seven standard ones vanish: yellow at 1.07:1, green at 1.37:1, cyan at
1.25:1 (measured 2026-08-27). Anything drawn on those layers was present in the
file and effectively absent from the image the render conditioned on.

The fix is a plot style, not monochrome: colour stays meaningful for the plan
tasks, and anything below 3:1 is darkened along its own hue until it clears it.
"""

from __future__ import annotations

import io

import ezdxf
from PIL import Image

from app.services import cad_convert
from app.services.cad_convert import (
    _MIN_PLOT_CONTRAST,
    _contrast_on_white,
    _ensure_plot_legibility,
    _legible_rgb,
)

YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
GREEN = (0, 255, 0)


def test_the_three_invisible_standards_become_legible():
    for rgb in (YELLOW, CYAN, GREEN):
        fixed = _legible_rgb(rgb)
        assert _contrast_on_white(fixed) >= _MIN_PLOT_CONTRAST, (rgb, fixed)


def test_darkening_preserves_the_hue():
    r, g, b = _legible_rgb(YELLOW)
    assert r == g and b < r, "darkened yellow should stay yellow-family (olive)"
    r, g, b = _legible_rgb(CYAN)
    assert g == b and r < g, "darkened cyan should stay cyan-family (teal)"


def test_already_legible_colours_are_untouched():
    for rgb in ((0, 0, 0), (180, 30, 30), (0, 0, 255), (60, 60, 60)):
        assert _legible_rgb(rgb) == rgb


def _drawing(tmp_path, layer_color_aci: int):
    doc = ezdxf.new("R2018")
    doc.layers.add("A-TEST", color=layer_color_aci)
    msp = doc.modelspace()
    for i in range(6):
        msp.add_lwpolyline(
            [(0, i), (30.0, i), (30.0, i + 0.5), (0, i + 0.5)],
            close=True,
            dxfattribs={"layer": "A-TEST"},
        )
    path = tmp_path / "one_layer.dxf"
    doc.saveas(path)
    return path


def test_the_layer_table_is_remapped(tmp_path):
    doc = ezdxf.readfile(_drawing(tmp_path, layer_color_aci=2))  # ACI 2 = yellow
    _ensure_plot_legibility(doc)
    layer = doc.layers.get("A-TEST")
    assert layer.rgb is not None
    assert _contrast_on_white(layer.rgb) >= _MIN_PLOT_CONTRAST


def test_entity_colour_overrides_are_remapped(tmp_path):
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    line = msp.add_line((0, 0), (10, 0), dxfattribs={"color": 4})  # explicit cyan
    _ensure_plot_legibility(doc)
    assert line.rgb is not None
    assert _contrast_on_white(line.rgb) >= _MIN_PLOT_CONTRAST


def test_yellow_linework_is_actually_visible_in_the_raster(tmp_path):
    """The end-to-end claim: a drawing whose only content sits on a yellow
    layer must produce a raster with real ink in it, not a blank white page."""
    path = _drawing(tmp_path, layer_color_aci=2)
    png = cad_convert.converter_for(".dxf").convert(path).view_png
    image = Image.open(io.BytesIO(png)).convert("L")
    darkest = int(min(image.get_flattened_data())) if hasattr(image, "get_flattened_data") else min(image.getdata())
    # Yellow on white rasterized to ~229 luminance before the fix; the floor
    # of 3:1 lands at or below ~180.
    assert darkest <= 190, f"darkest pixel {darkest}: the linework is still invisible"
