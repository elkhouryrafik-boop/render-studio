"""The DXF raster must be the size we chose, not the size the units imply.

Found 2026-08-27 by converting one drawing authored four ways. `_page()` sizes
the page from the drawing's extents treated as millimetres, so before the fix
the same plan came out at 2400px in millimetres, 2400px in inches, 659px in
feet and 242px in metres. Metres is the default for most Revit and ArchiCAD
exports outside the US, and at 242px every room name and dimension string is
unreadable. A control image that small is how a conditioned render ends up
inventing a building instead of tracing one, which is the exact failure the
2026-08-27 render review documents.
"""

from __future__ import annotations

import io

import ezdxf
import pytest
from ezdxf.math import Matrix44
from PIL import Image

from app.services import cad_convert

# One unit in the drawing is this many metres. The building below is 18m long,
# so these span 18 units (metres) to 18000 units (millimetres).
UNIT_SCALES = {
    "metres": 1.0,
    "centimetres": 100.0,
    "millimetres": 1000.0,
    "feet": 3.28084,
    "inches": 39.3701,
}


def _plan(tmp_path, scale: float, name: str):
    """A small floor plan with walls, an opening and a label, at `scale`."""
    doc = ezdxf.new("R2018", setup=True)
    doc.layers.add("A-WALL", color=7)
    doc.layers.add("A-TEXT", color=7)
    msp = doc.modelspace()

    def box(x0, y0, x1, y1):
        msp.add_lwpolyline(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            close=True,
            dxfattribs={"layer": "A-WALL"},
        )

    box(0, 0, 18.0, 12.0)
    box(0.3, 0.3, 17.7, 11.7)
    msp.add_line((9.0, 0.3), (9.0, 11.7), dxfattribs={"layer": "A-WALL"})
    msp.add_text("STUDIO", height=0.4, dxfattribs={"layer": "A-TEXT"}).set_placement(
        (4.5, 6.0)
    )

    for entity in list(msp):
        entity.transform(Matrix44.scale(scale, scale, scale))

    path = tmp_path / f"{name}.dxf"
    doc.saveas(path)
    return path


def _size(png: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(png)) as im:
        return im.size


@pytest.mark.parametrize("units", sorted(UNIT_SCALES))
def test_raster_long_edge_is_unit_independent(tmp_path, units):
    path = _plan(tmp_path, UNIT_SCALES[units], units)
    result = cad_convert.converter_for(".dxf").convert(path)
    long_edge = max(_size(result.view_png))

    target = cad_convert._TARGET_LONG_EDGE_PX
    assert long_edge == pytest.approx(target, rel=cad_convert._LONG_EDGE_TOLERANCE), (
        f"a plan authored in {units} rasterized to {long_edge}px, "
        f"but the target is {target}px whatever the units"
    )


def test_every_unit_system_agrees_with_every_other(tmp_path):
    """Not just near the target: near each other, so the aspect is stable too."""
    sizes = {
        units: _size(
            cad_convert.converter_for(".dxf").convert(_plan(tmp_path, scale, units)).view_png
        )
        for units, scale in UNIT_SCALES.items()
    }
    widths = {w for w, _ in sizes.values()}
    heights = {h for _, h in sizes.values()}
    assert max(widths) - min(widths) <= 2, sizes
    assert max(heights) - min(heights) <= 2, sizes


def test_a_layer_subset_keeps_the_same_output_size(tmp_path):
    """Turning layers off must not shrink the raster to whatever is left."""
    path = _plan(tmp_path, UNIT_SCALES["metres"], "subset")
    converter = cad_convert.converter_for(".dxf")
    full = max(_size(converter.convert(path).view_png))
    walls = max(_size(converter.convert(path, layers=["A-WALL"]).view_png))
    assert walls == pytest.approx(full, rel=cad_convert._LONG_EDGE_TOLERANCE)


def test_line_weight_scale_does_not_change_the_output_size(tmp_path):
    path = _plan(tmp_path, UNIT_SCALES["metres"], "weights")
    converter = cad_convert.converter_for(".dxf")
    thin = _size(converter.convert(path, line_weight_scale=0.5).view_png)
    thick = _size(converter.convert(path, line_weight_scale=3.0).view_png)
    assert thin == thick


def test_png_size_reads_the_header_without_decoding(tmp_path):
    path = _plan(tmp_path, UNIT_SCALES["metres"], "header")
    png = cad_convert.converter_for(".dxf").convert(path).view_png
    assert cad_convert._png_size(png) == _size(png)


def test_png_size_refuses_anything_that_is_not_a_png():
    assert cad_convert._png_size(b"") == (0, 0)
    assert cad_convert._png_size(b"<svg xmlns='http://www.w3.org/2000/svg'/>") == (0, 0)
