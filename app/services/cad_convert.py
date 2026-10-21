"""CAD files in, preview and render-ready images out.

fal.ai's endpoints read pixels only — nothing here talks to fal.ai, and nothing
here spends money. This module's whole job is turning a CAD file into a
`ConversionResult` an architect can look at before any credit is spent. Bad
rasterizations are cheap to redo here; they are not cheap once they have gone
through a render.

Three converters share one contract (`ConversionResult`) so the router and the
React UI do not need to know which file type they are looking at:

  * `DxfConverter` — implemented, via ezdxf's PyMuPDF and SVG drawing backends.
  * `IfcConverter` — implemented: ifcopenshell → trimesh → pyrender offscreen,
    producing a clay view AND a true z-buffer depth pass. Revit exports IFC
    natively, which is why this exists — raw .rvt has no open parser and is a
    deliberate non-goal.
  * `SkpConverter` — a stub. The real implementation needs real SKP
    samples to validate against (openskp is reverse-engineered), so it raises
    until those exist. See the class docstring for the planned strategy.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Protocol

import ezdxf
import ezdxf.colors as ezdxf_colors
from ezdxf import bbox
from ezdxf.addons.drawing import Frontend, RenderContext, layout, pymupdf, svg
from ezdxf.addons.drawing.config import BackgroundPolicy, Configuration

from app.config import Settings, get_settings
from app.services.materials import (
    Confidence,
    MaterialClass,
    MaterialSignal,
    change_map_pixel,
    is_entourage,
    is_placeholder_material,
    is_renderable,
    resolve,
    srgb_to_linear,
)

log = logging.getLogger(__name__)

# Long edge of the rasterized PNG, in pixels. Page(0, 0, ...) auto-detects a
# page size from the drawing's extents in its own units (mm, historically),
# and an architectural DXF drawn at 1 unit = 1 mm has extents in the
# thousands — feeding that straight to PyMuPDF at any real DPI asks it for a
# multi-gigapixel image and it raises "Overly large image". Capping the page
# in mm, derived from a pixel budget and this fixed DPI, bounds that.
#
# The cap alone is only half the problem, and the missing half was a real bug
# (found 2026-08-27, testing the DXF path end to end). The page is sized from
# the extents *as millimetres*, so the output resolution is a function of the
# units the drawing happens to be authored in. One drawing, four unit systems,
# measured:
#
#     millimetres  ->  2400 x 1859     (hits the cap, correct)
#     inches       ->  2400 x 1863     (hits the cap, correct)
#     feet         ->   659 x  523     (too small)
#     metres       ->   242 x  200     (unusable: every room name is mush)
#
# Metres is not an exotic choice — it is the default for most Revit and
# ArchiCAD exports outside the US — and a 242px control image is exactly how a
# conditioned render ends up inventing a building instead of tracing one. So
# the DPI is now solved per file, from the page the drawing actually produced,
# to land the long edge near the target whatever the units.
_DPI = 150
_MAX_LONG_EDGE_PX = 2400
_MAX_PAGE_MM = _MAX_LONG_EDGE_PX / _DPI * 25.4

# What the long edge should come out at. Below this, conditioning stops
# holding fine linework; above it, the file is large for no visible gain.
_TARGET_LONG_EDGE_PX = 2000
# Bounds on the solved DPI. The floor is PyMuPDF's own sane minimum; the
# ceiling stops a drawing with tiny extents (a detail authored in metres, a
# single door leaf) from asking for a gigapixel raster.
_MIN_SOLVED_DPI = 24
_MAX_SOLVED_DPI = 4800
# Re-render only when the first attempt is off by more than this. A drawing
# that already lands close pays for one rasterization, not two.
_LONG_EDGE_TOLERANCE = 0.15
_PAGE_MARGIN_MM = 5.0

# ACI 7 is "the adaptive one" — white on a dark viewer, black on a white page.
# ezdxf resolves it against the background at draw time, but a layer swatch
# shown in the browser is drawn on a white UI, independent of that resolution,
# so it needs its own answer. Black is correct for every DXF this app will
# ever rasterize, because the output is always a white-background PNG.
_ACI_ADAPTIVE = 7

# ACI colours are chosen to be readable on AutoCAD's BLACK model space, and
# three of the seven standard ones do not survive the move to a white page:
# yellow measures 1.07:1 against white, green 1.37:1, cyan 1.25:1 (WCAG
# contrast, measured 2026-08-27). Anything an office puts on those layers was
# present in the file, present in the SVG markup, and effectively absent from
# the image the render conditioned on.
#
# The fix is a plot style, not a monochrome switch. Colour is meaningful input
# to floorplan_prettify and masterplan_prettify — the model reads a coloured
# plan — so flattening everything to ink would trade one loss for another.
# Instead, any colour below the floor is darkened along its own hue until it
# clears it, exactly what AutoCAD's own print style tables do when plotting
# yellow on white paper: yellow becomes olive, cyan becomes teal, and the
# layer still reads as itself.
_MIN_PLOT_CONTRAST = 3.0
_LEGIBILITY_DARKEN_STEP = 0.92

# A DXF's lineweight is a real-world plotting width — often ~0.25mm, correct
# on a printed sheet. ezdxf's SVG backend renders it as that same fraction of
# the page regardless of page size, so a ~400mm architectural drawing squeezed
# into a ~400px browser preview reduces a 0.25mm line to a quarter of a CSS
# pixel: present in the markup, invisible on screen. `min_lineweight` is a
# floor, in 1/300", under which no line is allowed to shrink — thick lines are
# untouched, only the vanishing ones are rescued. The PNG path needs no such
# floor: PyMuPDF rasterizes at a real DPI, where the same 0.25mm line is
# already a visible ~1.5px.
_SVG_MIN_LINEWEIGHT = 14


class CadConversionError(Exception):
    """The file could not be turned into a preview or a render input.

    Same idiom as `assets.AssetError`: raised only for something the architect
    can act on — a corrupt file, an empty drawing, a layer selection that
    leaves nothing to draw. Nothing has been rasterized when this is raised.
    """


class CadConversionPending(Exception):
    """Not an error — a background bake is running and has not reached this
    file yet. SKP's initial parse+bake takes minutes (about 14 on a ~28 MB file;
    proportionally longer on a larger one), so nothing waits on it inline.
    Call the same method again in a few seconds; `stage` and
    `elapsed_seconds` are enough to show a progress message meanwhile.
    """

    def __init__(self, message: str, *, stage: str, elapsed_seconds: float,
                 estimated_seconds: float = 0.0) -> None:
        super().__init__(message)
        self.stage = stage
        self.elapsed_seconds = elapsed_seconds
        # How long the whole bake is expected to take, so the caller can say
        # "about 16 minutes" instead of "several minutes". 0 means unknown.
        self.estimated_seconds = estimated_seconds


@dataclass(frozen=True)
class LayerInfo:
    name: str
    color: str | None
    visible: bool = True


@dataclass(frozen=True)
class CameraMeta:
    """The camera a converter used. 3D formats only — a DXF has no camera."""

    preset: str


# The camera presets a 3D converter offers. IFC (and SKP, later) files carry
# no usable camera of their own — openskp extracts none, and an IFC's contexts
# are not views — so there is no "the" view to render, only chosen ones. The
# preset names are the contract between converter, router, and the UI buttons.
CAMERA_PRESETS = ("iso", "front", "top")


@dataclass(frozen=True)
class MaterialUsage:
    """One resolved appearance's share of a render — a row in the finish
    schedule the materials panel shows. `share_of_visible_pixels` excludes the
    synthetic ground plane and the sky/background, so the rows in one
    `ConversionResult.materials` sum to (approximately) 1.0.
    """

    material_class: str
    confidence: str
    albedo: tuple[float, float, float]
    share_of_visible_pixels: float
    source: str
    tile_width_mm: float | None = None


@dataclass
class ConversionResult:
    """What a CAD file becomes on its way into the normal render pipeline.

    `view_png` is what the architect previews and what becomes the `upload`
    asset on confirm. `depth_png`, `albedo_png` and `change_png` are SKP's
    true-depth, materially-correct-but-unlit, and per-pixel-confidence passes
    — always `None` for DXF, since a 2D drawing has no z-axis or material
    resolution to export. `semantic_png` is reserved for a future layer/class
    segmentation pass and is not produced by any converter yet. `camera`
    records how a perspective view was produced, so re-converting at a
    different angle is reproducible. `materials` is the finish schedule —
    empty except for SKP, the only converter that resolves surface materials.
    """

    view_png: bytes
    depth_png: bytes | None = None
    semantic_png: bytes | None = None
    albedo_png: bytes | None = None
    change_png: bytes | None = None
    camera: CameraMeta | None = None
    layers: list[LayerInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    materials: list[MaterialUsage] = field(default_factory=list)


class CadConverter(Protocol):
    """What every converter must offer, so callers do not branch on file type."""

    def layers(self, path: Path) -> list[LayerInfo]:
        """The layer manifest, read without rasterizing anything."""
        ...

    def preview(
        self,
        path: Path,
        *,
        layers: Sequence[str] | None = None,
        camera: str = "iso",
    ) -> bytes:
        """A small preview for the browser — SVG for 2D formats, PNG for 3D.
        The router sniffs which from the bytes. Never billed. `camera` is
        meaningful only to 3D converters; the DXF one ignores it."""
        ...

    def convert(
        self,
        path: Path,
        *,
        layers: Sequence[str] | None = None,
        line_weight_scale: float = 1.0,
        camera: str = "iso",
    ) -> ConversionResult:
        """The final rasterization, produced only once the architect confirms."""
        ...


def _layer_manifest(doc: ezdxf.document.Drawing) -> list[LayerInfo]:
    out: list[LayerInfo] = []
    for layer in doc.layers:
        # DXF encodes "this layer is off" as a negated color, so the sign has
        # to be stripped before the value means a color at all — is_off()
        # below is the readable form of that same bit.
        aci = abs(int(layer.dxf.color))
        color: str | None = None
        if aci == _ACI_ADAPTIVE:
            color = "#000000"
        elif 0 < aci < 256:
            r, g, b = ezdxf_colors.aci2rgb(aci)
            color = f"#{r:02x}{g:02x}{b:02x}"
        out.append(
            LayerInfo(
                name=str(layer.dxf.name),
                color=color,
                visible=not (layer.is_off() or layer.is_frozen()),
            )
        )
    return out


def _layer_filter(layers: Sequence[str] | None):
    """A `filter_func` for `Frontend.draw_layout`, or `None` for no filtering.

    `None` here means "every layer" — the same meaning it has on the
    `ConversionResult`-producing callers — rather than "no layers", which
    would silently rasterize a blank page.
    """
    if layers is None:
        return None
    wanted = {name.lower() for name in layers}
    return lambda entity: entity.dxf.layer.lower() in wanted


def _page() -> layout.Page:
    return layout.Page(
        0, 0,
        margins=layout.Margins.all(_PAGE_MARGIN_MM),
        max_width=_MAX_PAGE_MM,
        max_height=_MAX_PAGE_MM,
    )


def _png_size(data: bytes) -> tuple[int, int]:
    """A PNG's dimensions, read from its IHDR without decoding the pixels."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return (0, 0)
    return (
        int.from_bytes(data[16:20], "big"),
        int.from_bytes(data[20:24], "big"),
    )


def _contrast_on_white(rgb: tuple[int, int, int]) -> float:
    """WCAG contrast ratio of a colour against a white page."""

    def channel(value: int) -> float:
        c = value / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    luminance = (
        0.2126 * channel(rgb[0]) + 0.7152 * channel(rgb[1]) + 0.0722 * channel(rgb[2])
    )
    return 1.05 / (luminance + 0.05)


def _legible_rgb(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """The same hue, darkened until it clears `_MIN_PLOT_CONTRAST` on white."""
    if _contrast_on_white(rgb) >= _MIN_PLOT_CONTRAST:
        return rgb
    r, g, b = (float(c) for c in rgb)
    # Multiplicative darkening preserves the channel ratios, which is the hue.
    # Forty steps of 0.92 reaches black, so this always terminates.
    for _ in range(40):
        r, g, b = r * _LEGIBILITY_DARKEN_STEP, g * _LEGIBILITY_DARKEN_STEP, b * _LEGIBILITY_DARKEN_STEP
        if _contrast_on_white((round(r), round(g), round(b))) >= _MIN_PLOT_CONTRAST:
            break
    return (round(r), round(g), round(b))


def _ensure_plot_legibility(doc: ezdxf.document.Drawing) -> None:
    """Darken any colour that would vanish on the white page. See the note on
    `_MIN_PLOT_CONTRAST` for why this is a plot style rather than monochrome.

    Layer colours cover the BYLAYER case, which is most of an architectural
    drawing; entities with their own colour override (in modelspace and inside
    block definitions) are remapped individually. ACI 7 is skipped — ezdxf
    already resolves it adaptively to black on a white background. The document
    is re-read from disk on every call, so mutating it here touches nothing
    but this render.
    """
    for lyr in doc.layers:
        rgb = lyr.rgb
        if rgb is None:
            aci = abs(int(lyr.dxf.color))
            if aci == 7 or not 1 <= aci <= 255:
                continue
            rgb = ezdxf_colors.aci2rgb(aci)
        fixed = _legible_rgb(rgb)
        if fixed != rgb:
            lyr.rgb = fixed

    for block in doc.blocks:
        for entity in block:
            rgb = getattr(entity, "rgb", None)
            if rgb is None:
                aci = entity.dxf.color if entity.dxf.hasattr("color") else 256
                if aci in (0, 256, 7) or not 1 <= aci <= 255:
                    continue
                rgb = ezdxf_colors.aci2rgb(aci)
            fixed = _legible_rgb(rgb)
            if fixed != rgb:
                entity.rgb = fixed


def _content_aspect(
    doc: ezdxf.document.Drawing, layers: Sequence[str] | None
) -> float | None:
    """Width over height of the drawn geometry, in the drawing's own units.

    Taken from the geometry rather than from a trial raster on purpose. A trial
    raster's bounding box includes stroke widths, which are absolute
    millimetres, so at a small page size they are a large fraction of the image
    and the measured aspect drifts with the drawing's units. That drift was
    worth up to 17% of the short edge.
    """
    keep = _layer_filter(layers)
    entities = [e for e in doc.modelspace() if keep is None or keep(e)]
    if not entities:
        return None
    try:
        box = bbox.extents(entities, fast=True)
    except Exception:  # noqa: BLE001 - any bbox failure just means "fall back"
        return None
    if not box.has_data or box.size.x <= 0 or box.size.y <= 0:
        return None
    return box.size.x / box.size.y


def _rasterize_to_target(
    backend: pymupdf.PyMuPdfBackend,
    doc: ezdxf.document.Drawing,
    layers: Sequence[str] | None,
) -> bytes:
    """Rasterize so the long edge lands on `_TARGET_LONG_EDGE_PX`.

    An auto-sized page (`Page(0, 0, ...)`) takes its millimetre size from the
    drawing's own extents, so the output resolution ends up being a function of
    the units the drawing was authored in rather than anything anyone chose.
    See the note on `_DPI` for the measurements.

    Raising the DPI does not fix it: with an auto-sized page ezdxf re-fits the
    content, and the pixel count is not linear in the DPI (measured: the same
    metre-unit drawing gave 242px at 150 dpi and 4754px at 1240 dpi, where
    linear would have been 2050). So the page is made explicit instead, sized
    from the geometry's aspect ratio, and ezdxf fits the content to it.
    """
    aspect = _content_aspect(doc, layers)
    if aspect is None:
        return backend.get_pixmap_bytes(_page(), dpi=_DPI, fmt="png")

    long_mm = _TARGET_LONG_EDGE_PX / _DPI * 25.4
    short_mm = long_mm / aspect if aspect >= 1.0 else long_mm * aspect
    page_w, page_h = (long_mm, short_mm) if aspect >= 1.0 else (short_mm, long_mm)
    page = layout.Page(
        page_w + 2 * _PAGE_MARGIN_MM,
        page_h + 2 * _PAGE_MARGIN_MM,
        margins=layout.Margins.all(_PAGE_MARGIN_MM),
    )
    try:
        return backend.get_pixmap_bytes(page, dpi=_DPI, fmt="png")
    except (ValueError, RuntimeError, MemoryError):
        # A refused page is not worth failing the conversion over. PyMuPDF
        # raises RuntimeError for one it will not rasterize and ValueError for
        # one it will not accept; the auto-sized page always works, it is just
        # the wrong size.
        log.warning("DXF raster onto an explicit page failed; falling back to the auto page")
        return backend.get_pixmap_bytes(_page(), dpi=_DPI, fmt="png")


def _warnings_for(doc: ezdxf.document.Drawing, requested_layers: Sequence[str] | None) -> list[str]:
    known = {layer.dxf.name.lower() for layer in doc.layers}
    return [
        f"Layer {name!r} was requested but does not exist in this file."
        for name in requested_layers or []
        if name.lower() not in known
    ]


def _require_visible_entities(doc: ezdxf.document.Drawing, layers: Sequence[str] | None) -> None:
    """Refuse before rasterizing rather than crashing inside it.

    An empty set of entities to draw — an empty drawing, or every selected
    layer turning out to be empty — leaves the page's content extents
    undefined. `Page(0, 0, ...)` auto-detects its size from those extents, so
    PyMuPDF is asked to size a page from NaN and raises deep inside its own
    coordinate math. Catching the empty case here turns that into a sentence
    the architect can act on.
    """
    filt = _layer_filter(layers)
    msp = doc.modelspace()
    has_entities = any(filt(e) for e in msp) if filt else len(msp) > 0
    if not has_entities:
        reason = (
            "None of the selected layers have anything on them. Choose at "
            "least one layer with geometry."
            if layers is not None
            else "This drawing's model space has no entities to show."
        )
        raise CadConversionError(reason)


def _read(path: Path) -> ezdxf.document.Drawing:
    """Load a DXF, turning ezdxf's own exceptions into `CadConversionError`.

    `readfile` raises `OSError` for a file that fails its DXF sniff test and
    `ezdxf.DXFError` for one that passes it but is malformed further in — both
    mean the same thing to the architect: this file cannot be read.
    """
    try:
        return ezdxf.readfile(str(path))
    except (OSError, ezdxf.DXFError) as exc:
        raise CadConversionError(f"That file could not be read as DXF: {exc}") from exc


class DxfConverter:
    """AutoCAD DXF → PNG/SVG, via ezdxf's PyMuPDF and SVG drawing backends.

    Uses `PyMuPdfBackend` rather than the matplotlib backend ezdxf's older
    examples show: it is the documented native PNG path (DPI control, page
    layout, no matplotlib in the dependency tree), and the same drawing
    frontend also drives `SVGBackend` for a preview that costs nothing to
    generate and re-generate as the architect toggles layers.
    """

    def layers(self, path: Path) -> list[LayerInfo]:
        return _layer_manifest(_read(path))

    def preview(
        self,
        path: Path,
        *,
        layers: Sequence[str] | None = None,
        camera: str = "iso",
    ) -> bytes:
        doc = _read(path)
        _require_visible_entities(doc, layers)
        _ensure_plot_legibility(doc)
        ctx = RenderContext(doc)
        backend = svg.SVGBackend()
        config = Configuration(
            background_policy=BackgroundPolicy.WHITE,
            min_lineweight=_SVG_MIN_LINEWEIGHT,
        )
        frontend = Frontend(ctx, backend, config=config)
        frontend.draw_layout(doc.modelspace(), filter_func=_layer_filter(layers))
        return backend.get_string(_page()).encode("utf-8")

    def convert(
        self,
        path: Path,
        *,
        layers: Sequence[str] | None = None,
        line_weight_scale: float = 1.0,
        camera: str = "iso",
    ) -> ConversionResult:
        doc = _read(path)
        _require_visible_entities(doc, layers)
        _ensure_plot_legibility(doc)
        ctx = RenderContext(doc)
        backend = pymupdf.PyMuPdfBackend()
        config = Configuration(
            background_policy=BackgroundPolicy.WHITE,
            lineweight_scaling=line_weight_scale,
        )
        frontend = Frontend(ctx, backend, config=config)
        frontend.draw_layout(doc.modelspace(), filter_func=_layer_filter(layers))
        view_png = _rasterize_to_target(backend, doc, layers)

        return ConversionResult(
            view_png=view_png,
            layers=_layer_manifest(doc),
            warnings=_warnings_for(doc, layers),
        )


# SkpConverter is defined after the IFC section — it reuses `_look_at` and
# `_RENDER_LOCK` from there rather than duplicating them.


# --- IFC (the Revit path) -----------------------------------------------------


_IFC_PRESET_DIRECTIONS = {
    # Unit-ish direction from the model's centre toward the camera, Z up.
    "iso": (1.0, -1.0, 0.8),
    "front": (0.0, -1.0, 0.12),
    "top": (0.0, -1e-3, 1.0),  # not exactly vertical, so the up-vector stays defined
}

# Output sizes. Convert matches the app's default render aspect (4:3); the
# preview is half-size because it is regenerated on every storey toggle and
# camera change, and a browser <img> a few hundred px wide cannot show the
# difference anyway.
_IFC_CONVERT_SIZE = (1600, 1200)
_IFC_PREVIEW_SIZE = (800, 600)


class IfcConverter:
    """IFC → clay view + true depth, via ifcopenshell → trimesh → pyrender.

    This is the Revit path. Raw .rvt is a closed format with no open parser
    and only a cloud-upload conversion route (Autodesk APS) — wrong shape for
    a LAN-only tool — but Revit exports IFC natively, so "send the IFC" is
    the ask. Unlike SKP this ships immediately: the format is an open
    standard and ifcopenshell is a trusted parser, so there is no
    validation gamble to wait on.

    "Layers" for an IFC file are its building storeys — that is the unit an
    architect actually thinks in when choosing what to show. Elements outside
    any storey are grouped under a pseudo-layer so nothing silently vanishes.

    The depth pass is the whole reason this converter exists: pyrender's
    z-buffer is *true* depth, which `massing_to_render`'s `prefer_supplied:
    true` conditioning has anticipated since before this module existed. It
    leaves here already in the ControlNet MiDaS convention — 8-bit, near =
    bright, background black — because the render service should never need
    to know what convention a control image was produced under.
    """

    _UNASSIGNED = "(no storey)"

    def layers(self, path: Path) -> list[LayerInfo]:
        model = _ifc_read(path)
        names, has_unassigned = _ifc_storey_names(model)
        out = [LayerInfo(name=n, color=None, visible=True) for n in names]
        if has_unassigned:
            out.append(LayerInfo(name=self._UNASSIGNED, color=None, visible=True))
        return out

    def preview(
        self,
        path: Path,
        *,
        layers: Sequence[str] | None = None,
        camera: str = "iso",
    ) -> bytes:
        model = _ifc_read(path)
        view_png, _depth, _warnings = _ifc_render(
            model, layers=layers, camera=camera,
            size=_IFC_PREVIEW_SIZE, with_depth=False,
        )
        return view_png

    def convert(
        self,
        path: Path,
        *,
        layers: Sequence[str] | None = None,
        line_weight_scale: float = 1.0,  # meaningless for a mesh render; accepted, ignored
        camera: str = "iso",
    ) -> ConversionResult:
        model = _ifc_read(path)
        names, has_unassigned = _ifc_storey_names(model)
        known = list(names) + ([self._UNASSIGNED] if has_unassigned else [])
        warnings = [
            f"Storey {name!r} was requested but does not exist in this file."
            for name in layers or []
            if name.lower() not in {k.lower() for k in known}
        ]
        view_png, depth_png, mesh_warnings = _ifc_render(
            model, layers=layers, camera=camera,
            size=_IFC_CONVERT_SIZE, with_depth=True,
        )
        return ConversionResult(
            view_png=view_png,
            depth_png=depth_png,
            camera=CameraMeta(preset=camera),
            layers=[LayerInfo(name=n, color=None, visible=True) for n in known],
            warnings=warnings + mesh_warnings,
        )


def _ifc_read(path: Path):
    import ifcopenshell

    try:
        model = ifcopenshell.open(str(path))
    except Exception as exc:  # ifcopenshell raises bare Exception/IOError for bad files
        raise CadConversionError(f"That file could not be read as IFC: {exc}") from exc
    if model is None:
        raise CadConversionError("That file could not be read as IFC.")
    return model


def _ifc_storey_names(model) -> tuple[list[str], bool]:
    """Storey names in file order, plus whether any product lives outside one."""
    import ifcopenshell.util.element

    names = [str(s.Name or f"Storey {s.id()}") for s in model.by_type("IfcBuildingStorey")]
    has_unassigned = False
    for product in model.by_type("IfcProduct"):
        if not product.Representation:
            continue
        container = ifcopenshell.util.element.get_container(product)
        if container is None or not container.is_a("IfcBuildingStorey"):
            has_unassigned = True
            break
    return names, has_unassigned


def _ifc_meshes(model, layers: Sequence[str] | None) -> tuple[list, list[str]]:
    """(trimesh meshes, warnings) for every representable product that passes
    the storey filter. A product ifcopenshell cannot triangulate becomes a
    warning rather than sinking the whole conversion — one broken proxy
    object must not cost the architect the building."""
    import ifcopenshell.geom
    import ifcopenshell.util.element
    import numpy as np
    import trimesh

    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)

    wanted = {n.lower() for n in layers} if layers is not None else None
    meshes: list = []
    warnings: list[str] = []
    for product in model.by_type("IfcProduct"):
        if not product.Representation:
            continue
        if product.is_a("IfcSpace") or product.is_a("IfcOpeningElement"):
            # Spaces are air and openings are subtractions — rendering either
            # as solid geometry draws boxes where the model has rooms/windows.
            continue
        if wanted is not None:
            container = ifcopenshell.util.element.get_container(product)
            in_storey = container is not None and container.is_a("IfcBuildingStorey")
            name = str(container.Name or f"Storey {container.id()}") if in_storey else IfcConverter._UNASSIGNED
            if name.lower() not in wanted:
                continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            verts = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
            faces = np.array(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
        except Exception as exc:  # noqa: BLE001 — per-product, reported not fatal
            warnings.append(
                f"Could not read {product.is_a()} {product.Name or product.GlobalId!r}; "
                f"it is missing from the image: {exc}"
            )
            continue
        if len(verts) and len(faces):
            meshes.append(trimesh.Trimesh(vertices=verts, faces=faces))
    return meshes, warnings


def _look_at(eye, target, up):
    """Camera pose from eye/target/up. Shared by IFC and SKP.

    Recovers from `up` parallel to the view direction (a straight-down `top`
    view) rather than emitting NaNs into the pose: SKP's up-axis is only
    known at render time, so a caller cannot always guarantee a non-parallel
    up in advance the way the IFC presets do.
    """
    import numpy as np

    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    norm = np.linalg.norm(right)
    if norm < 1e-9:
        up = np.roll(np.abs(fwd), 1)
        right = np.cross(fwd, up)
        norm = np.linalg.norm(right)
    right = right / norm
    true_up = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, true_up, -fwd, eye
    return pose


# pyrender's offscreen renderer holds process-global GL state through pyglet.
# FastAPI serves sync endpoints from a thread pool, so two conversions can
# genuinely overlap; serializing them costs a queue at office scale and
# removes a class of crash that presents as "sometimes the preview 500s".
_RENDER_LOCK = threading.Lock()


def _ifc_render(
    model,
    *,
    layers: Sequence[str] | None,
    camera: str,
    size: tuple[int, int],
    with_depth: bool,
) -> tuple[bytes, bytes | None, list[str]]:
    """Rasterize a parsed IFC model to (view_png, depth_png|None, warnings)."""
    import numpy as np

    direction = _IFC_PRESET_DIRECTIONS.get(camera)
    if direction is None:
        raise CadConversionError(
            f"Unknown camera preset {camera!r}. Choose one of: {', '.join(CAMERA_PRESETS)}."
        )

    meshes, mesh_warnings = _ifc_meshes(model, layers)
    if not meshes:
        raise CadConversionError(
            "None of the selected storeys have anything on them. Choose at "
            "least one storey with geometry."
            if layers is not None
            else "This file has no renderable geometry."
        )

    import pyrender
    import trimesh
    from PIL import Image

    combined = trimesh.util.concatenate(meshes)
    center = combined.bounds.mean(axis=0)
    extent = float(np.linalg.norm(combined.bounds[1] - combined.bounds[0]))
    if extent <= 0:
        raise CadConversionError("This file's geometry has no size to frame a camera around.")

    d = np.array(direction, dtype=np.float64)
    eye = center + d / np.linalg.norm(d) * extent * 1.6
    up = np.array([0.0, 0.0, 1.0])

    with _RENDER_LOCK:
        scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.35, 0.35, 0.35])
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.82, 0.81, 0.78, 1.0], metallicFactor=0.0, roughnessFactor=0.9
        )
        scene.add(pyrender.Mesh.from_trimesh(combined, material=material, smooth=False))
        cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(45))
        scene.add(cam, pose=_look_at(eye, center, up))
        # Keyed to the camera so every preset is lit, plus a slight offset so
        # faces facing the camera are not uniformly flat.
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=_look_at(eye + np.array([2.0, 1.0, 4.0]), center, up))

        renderer = pyrender.OffscreenRenderer(*size)
        try:
            color, depth = renderer.render(scene)
        finally:
            renderer.delete()

    buf = BytesIO()
    Image.fromarray(color).save(buf, format="PNG")
    view_png = buf.getvalue()

    depth_png: bytes | None = None
    if with_depth:
        # True z-buffer → ControlNet depth convention: 8-bit, near = bright,
        # empty background = 0 (far). Normalized over the hit pixels only, so
        # the building uses the full value range whatever its distance.
        mask = depth > 0
        out = np.zeros_like(depth, dtype=np.uint8)
        if mask.any():
            dmin, dmax = float(depth[mask].min()), float(depth[mask].max())
            out[mask] = (
                255 * (1.0 - (depth[mask] - dmin) / max(dmax - dmin, 1e-9))
            ).astype(np.uint8)
        buf = BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        depth_png = buf.getvalue()

    return view_png, depth_png, mesh_warnings


# --- SKP (SketchUp, via openskp) ---------------------------------------------
#
# Validated against a large (~28 MB) SketchUp test model. Produces four
# passes per camera:
#
#   view    — clay. Geometry and light, no material information. The
#             conditioning plate.
#   depth   — true z-buffer, ControlNet convention (8-bit, near = bright).
#   albedo  — the same geometry painted with each surface's RESOLVED
#             material. Flat, unlit, and materially correct — the base image
#             a render refines from.
#   change  — per-pixel strength for differential diffusion, in fal's
#             convention: BRIGHT where a surface's material is known (hold
#             what we painted), DARK where it is guessed (repaint it). See
#             the polarity note at the encode site — shipping this the other
#             way round is what made the task return its own input.
#
# The parse+bake (openskp SkpFile.open().parse() + build_scene()) is the slow
# part — minutes, not seconds, and proportional to file size. It runs in a
# background thread and is cached to disk keyed by content hash, so a second
# conversion of the same file is fast. Everything after the bake — resolving
# materials, merging primitives, fitting a camera, rendering — is seconds.

# Camera presets as (ground_1, ground_2, up) components in the DETECTED frame,
# not world vectors: for a Y-up bake the ground axes are X and Z, for Z-up
# they are X and Y. The same three numbers produce the same view either way,
# which is what lets a preset survive up-axis detection instead of fighting
# it.
_SKP_PRESET_COMPONENTS: dict[str, tuple[float, float, float]] = {
    "iso": (1.0, -1.0, 1.15),
    "front": (0.0, -1.0, 0.12),
    "top": (0.0, 0.0, 1.0),
}

# "buildings-iso" frames on the built mass rather than the whole site slab, at
# a documented dolly-in factor (E1's building-camera renders, 2026-08-07). The
# shared `CAMERA_PRESETS` tuple (DXF/IFC's contract) is deliberately left
# alone — this is a preset name, not a new axis on the converter Protocol, so
# DXF and IFC stay oblivious to it. `camera` on the API is a plain string, not
# an enum, so this needs no schema change either.
_SKP_SUBJECT_CAMERAS: dict[str, tuple[str, float]] = {
    "buildings-iso": ("iso", 2.2),  # (base preset, zoom)
}
SKP_CAMERA_PRESETS = (*CAMERA_PRESETS, *_SKP_SUBJECT_CAMERAS)

# FLUX requires image dimensions that are multiples of 16.
_SKP_CONVERT_SIZE = (1600, 1200)
assert _SKP_CONVERT_SIZE[0] % 16 == 0 and _SKP_CONVERT_SIZE[1] % 16 == 0
_SKP_PREVIEW_SIZE = (800, 600)  # never sent to fal.ai — no multiple-of-16 need

_SKP_YFOV_DEG = 45.0
_SKP_FRAME_MARGIN = 1.03  # bbox fits the frame with ~3% air

# What stands above the ground rather than on it: standing 15% of the model's
# height above local grade is what a building does and a carriageway does not
# — material class alone cannot separate them ("concrete" is both).
_SKP_SUBJECT_MIN_HEIGHT_FRAC = 0.15

# Entourage is left unpainted in the albedo pass, so the change map must hand
# it to the model at full freedom regardless of how confidently its *name*
# resolved — a hole held at 0.5 stays a hole.
_SKP_ENTOURAGE_CHANGE_STRENGTH = 0.80

# The clay view's tonal range after the stretch. The raw pyrender pass put the
# whole scene — lit faces, shadowed faces, ground — inside roughly a fifth of
# the 8-bit range (pale grey on pale grey, measured 2026-08-27 on a large
# test model), which made the confirm view hard to judge and starved the canny
# pass that conditions the render of edges. The stretch maps the rendered
# geometry's 2nd..98th luminance percentiles onto this range; the background is
# untouched, and the transform is the same for all three channels so clay
# stays neutral.
_SKP_CLAY_RANGE = (60.0, 245.0)

# Below this many visible pixels the buildings are too small a sample to set
# the depth range from, so the pass falls back to all real geometry.
_SKP_DEPTH_MIN_SUBJECT_PX = 2000
# Ground pixels needed before the ground plane is fitted and removed. A
# three-parameter least squares on fewer than this is fitting noise, and the
# fallback below is the correct answer for a view with no ground in it.
_SKP_DEPTH_MIN_GROUND_PX = 5000
# Height-above-ground relief, as a fraction of the ground plane's own
# disparity span across the frame, below which there is nothing to show and
# the range-scaling fallback is the honest answer. Scale-free on purpose.
_SKP_DEPTH_MIN_RELIEF_FRACTION = 0.01
# Camera-space units. A subject flatter than this (one roof plane seen from
# directly above) has no depth of its own to scale to.
_SKP_DEPTH_MIN_SPREAD = 1e-3

_SKP_BUILT_CLASSES = frozenset({
    MaterialClass.CONCRETE, MaterialClass.GLASS, MaterialClass.METAL,
    MaterialClass.WOOD, MaterialClass.MASONRY, MaterialClass.BRICK,
    MaterialClass.CERAMIC, MaterialClass.STONE, MaterialClass.PLASTER,
    MaterialClass.ROOFING, MaterialClass.FABRIC, MaterialClass.PLASTIC,
    MaterialClass.UNKNOWN,
})
_SKP_GROUND_CLASSES = frozenset({
    MaterialClass.SOIL, MaterialClass.ASPHALT, MaterialClass.GRASS,
    MaterialClass.GRAVEL, MaterialClass.WATER,
})
_SKP_GROUND_LAYER_HINTS = ("terrain", "earthwork", "topo", "grading", "surface")


# --- bake: parse + build_scene, backgrounded and cached by content hash ------


def _skp_cache_key(path: Path) -> str:
    """The content hash this file is stored under.

    `asset_service.ingest_cad` writes every CAD upload to
    `inputs/<sha[:2]>/<sha><suffix>` — content-addressed, so identical bytes
    always occupy the same path. `path.stem` is therefore already the content
    hash for any file this converter is ever called with, and re-hashing a
    (possibly 200+ MB) file on every layers/preview/confirm call would be
    real, avoidable cost. This is a property of that storage convention, not
    of this file in isolation — if that convention ever changes, this needs
    to hash the bytes instead.
    """
    return path.stem


def _skp_cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    """Where one baked SKP scene lives: the payload, and its status file.

    `.npz` rather than the `.pkl` this used to be. The bake cache sits in
    DATA_DIR, which is a directory on a shared office PC, and `pickle.load` on a
    file anyone with write access can replace is code execution in the server
    process. npz holds arrays and one JSON blob and executes nothing.

    Old `.pkl` files are simply never read again — the key is a content hash, so
    the first request for that model re-bakes it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.npz", cache_dir / f"{key}.status.json"


def _skp_write_payload(payload: dict, npz_path: Path) -> None:
    """Serialise a baked scene without pickling anything.

    Geometry goes in as arrays (two per primitive) and everything else as one
    JSON string, so the loader can refuse pickles outright. The two dictionary
    shapes JSON cannot hold directly — an RGB tuple key, and a 2-tuple value —
    are converted here and converted back in `_skp_read_payload`; they are the
    only lossy-looking part, and both round-trip exactly.
    """
    import numpy as np

    # Everything is concatenated into four arrays plus offsets, NOT one array
    # per primitive.
    #
    # A .npz is a zip, and every member costs a local header, a central
    # directory entry and a 128-byte .npy header. A large (~27MB) test model
    # bakes to 52,017 primitives, so the naive one-array-each layout wrote
    # 104,034 zip members: 19.9MB of actual array data in a 98.1MB file, nearly
    # 80MB of pure container overhead, against 28.8MB for the pickle this
    # replaced. Measured 2026-08-27. Concatenating brings it to four members.
    #
    # Indices are int32, not int64. They address vertices within their own
    # primitive; the largest value in that same 52k-primitive model was 34,602.
    prims: list[dict] = []
    pos_parts: list[object] = []
    idx_parts: list[object] = []
    pos_offsets = [0]
    idx_offsets = [0]
    for prim in payload["primitives"]:
        positions = np.asarray(prim["positions"], dtype=np.float32).reshape(-1, 3)
        indices = np.asarray(prim["indices"], dtype=np.int32).reshape(-1)
        pos_parts.append(positions)
        idx_parts.append(indices)
        pos_offsets.append(pos_offsets[-1] + len(positions))
        idx_offsets.append(idx_offsets[-1] + len(indices))
        prims.append(
            {
                "material_index": int(prim["material_index"]),
                "geom_name": str(prim["geom_name"]),
            }
        )

    arrays: dict[str, object] = {
        "positions": (
            np.concatenate(pos_parts) if pos_parts
            else np.zeros((0, 3), dtype=np.float32)
        ),
        "indices": (
            np.concatenate(idx_parts) if idx_parts
            else np.zeros(0, dtype=np.int32)
        ),
        "positions_offsets": np.asarray(pos_offsets, dtype=np.int64),
        "indices_offsets": np.asarray(idx_offsets, dtype=np.int64),
    }

    meta = {
        "layout": 2,
        "primitives": prims,
        "gltf_materials": payload["gltf_materials"],
        "layer_by_geom": payload["layer_by_geom"],
        # RGB tuple keys, as "r,g,b".
        "names_by_color": {
            ",".join(str(int(c)) for c in rgb): list(names)
            for rgb, names in payload["names_by_color"].items()
        },
        # (filename, tile_width_mm) pairs, as two-element lists.
        "texture_by_name": {
            str(name): [value[0], value[1]]
            for name, value in payload["texture_by_name"].items()
        },
        "layers": list(payload["layers"]),
    }
    arrays["meta"] = np.array(json.dumps(meta))

    # Compressed, because `meta` is one JSON string holding a record per
    # primitive: 52.46MB raw for a large test model, 0.26MB deflated, a
    # 200x difference for 0.2s of write time on a bake that already takes 16
    # minutes. The geometry arrays are incompressible float32 and deflate
    # simply passes them through.
    tmp = npz_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp.replace(npz_path)  # atomic — a poller never sees a half-written cache

    # The pickle this file supersedes is dead the moment the npz lands. It is
    # never read again, but it was never deleted either, so every model left a
    # stale copy of itself behind (~29 MB for one large test model).
    stale = npz_path.with_suffix(".pkl")
    try:
        stale.unlink(missing_ok=True)
    except OSError:
        log.warning("Could not remove the superseded pickle cache at %s", stale)


def _skp_read_payload(npz_path: Path) -> dict:
    """Read a baked scene back. `allow_pickle=False` is the whole point."""
    import numpy as np

    with np.load(npz_path, allow_pickle=False) as bundle:
        meta = json.loads(str(bundle["meta"]))
        if meta.get("layout") == 2:
            positions = bundle["positions"]
            indices = bundle["indices"]
            pos_off = bundle["positions_offsets"]
            idx_off = bundle["indices_offsets"]
            primitives = [
                {
                    "positions": positions[pos_off[i] : pos_off[i + 1]],
                    "indices": indices[idx_off[i] : idx_off[i + 1]],
                    "material_index": prim["material_index"],
                    "geom_name": prim["geom_name"],
                }
                for i, prim in enumerate(meta["primitives"])
            ]
        else:
            # Layout 1: one zip member per array. Still read so an existing
            # cache is not thrown away, but never written again.
            primitives = [
                {
                    "positions": bundle[f"positions_{i}"],
                    "indices": bundle[f"indices_{i}"],
                    "material_index": prim["material_index"],
                    "geom_name": prim["geom_name"],
                }
                for i, prim in enumerate(meta["primitives"])
            ]

    return {
        "primitives": primitives,
        "gltf_materials": meta["gltf_materials"],
        "layer_by_geom": meta["layer_by_geom"],
        "names_by_color": {
            tuple(int(c) for c in key.split(",")): names
            for key, names in meta["names_by_color"].items()
        },
        "texture_by_name": {
            name: (value[0], value[1]) for name, value in meta["texture_by_name"].items()
        },
        "layers": meta["layers"],
    }


def _skp_write_status(status_path: Path, **fields: object) -> None:
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(fields), encoding="utf-8")
    tmp.replace(status_path)


def _skp_read_status(status_path: Path) -> dict | None:
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# One bake per content hash, however many requests arrive while it runs.
_SKP_BAKE_THREADS: dict[str, threading.Thread] = {}
_SKP_BAKE_REGISTRY_LOCK = threading.Lock()


def _skp_payload_from_scene(model, scene) -> dict:
    """The bake's cacheable payload: numpy arrays and plain data only.

    `build_scene` keys materials by RGB (openskp `scene.py:145`,
    `color_to_material_index`), so the material NAME does not survive the
    bake. Names are the only universally-preserved field across CAD formats
    and tier-1 resolution needs them, so they are joined back by colour here
    — lossy where two materials share an RGB, which is recorded rather than
    hidden (an ambiguous colour falls through to the layer).
    """
    import numpy as np

    by_color: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    tex_by_name: dict[str, tuple[str | None, float | None]] = {}
    for mat in model.materials:
        color = getattr(mat, "color", None)
        name = str(getattr(mat, "name", "") or "")
        if color is None or not name:
            continue
        rgb = tuple(int(c) for c in tuple(color)[:3])
        by_color[rgb].append(name)
        texture = getattr(mat, "texture", None)
        if texture is not None:
            # SketchUp texture width/height are the real-world tile size in
            # INCHES (Sketchup::Texture docs) — converted to mm once, here, so
            # nothing downstream has to remember that.
            width_in = getattr(texture, "width", None)
            tex_by_name[name] = (
                str(getattr(texture, "filename", "") or "") or None,
                float(width_in) * 25.4 if width_in else None,
            )

    return {
        "primitives": [
            {
                "positions": np.asarray(p.positions, dtype=np.float32),
                "indices": np.asarray(p.indices, dtype=np.int64),
                "material_index": p.material_index,
                "geom_name": p.geom_name,
            }
            for p in scene.glb_primitives
        ],
        "gltf_materials": scene.gltf_materials,
        "layer_by_geom": {k: getattr(v, "layer", "") for k, v in scene.mesh_index.items()},
        "names_by_color": dict(by_color),
        "texture_by_name": tex_by_name,
        "layers": [str(getattr(x, "name", x)) for x in model.layers],
    }


def _skp_bake_worker(path: Path, cache_path: Path, status_path: Path) -> None:
    """Runs off the request thread. Never raises — failure is reported
    through the status file, which is the only channel a poller reads."""
    started = datetime.now(UTC).isoformat()

    def status(**fields: object) -> None:
        _skp_write_status(
            status_path, started_at=started, updated_at=datetime.now(UTC).isoformat(), **fields
        )

    try:
        status(status="baking", stage="parsing", error=None)
        from openskp import SkpFile

        handle = SkpFile.open(path)
        model = handle.parse()
        log.info("SKP parsed %s: %d layers, %d materials", path.name, len(model.layers), len(model.materials))

        status(status="baking", stage="baking_scene", error=None)
        scene = handle.build_scene()
        log.info("SKP baked %s: %d primitives", path.name, len(scene.glb_primitives))

        status(status="baking", stage="caching", error=None)
        payload = _skp_payload_from_scene(model, scene)
        _skp_write_payload(payload, cache_path)

        status(status="completed", stage="done", error=None)
    except Exception as exc:
        log.exception("SKP bake failed for %s", path)
        status(status="failed", stage="parsing", error=str(exc)[:500])
    finally:
        with _SKP_BAKE_REGISTRY_LOCK:
            _SKP_BAKE_THREADS.pop(str(cache_path), None)


# Seconds of first-bake per megabyte of SKP. Measured 2026-08-27: a
# 27.3 MB test model took 974 seconds — 16 minutes — single-threaded, dominated
# by `baking_scene`, which is roughly linear in geometry and so in file size.
# One datapoint, so this is an estimate offered as one; it exists because
# "several minutes" in front of a 16-minute wait reads as a hang.
_SKP_BAKE_SECONDS_PER_MB = 974.0 / 27.3


def estimated_bake_seconds(path: Path) -> float:
    """Roughly how long this file's first bake will take. 0 if unknowable.

    Only the FIRST look at a given file costs this: the bake is cached by
    content hash, so every later conversion of the same file — another camera,
    another layer selection, another task — is seconds.
    """
    try:
        size_mb = path.stat().st_size / 1_000_000
    except OSError:
        return 0.0
    return max(size_mb * _SKP_BAKE_SECONDS_PER_MB, 30.0)


def is_bake_cached(path: Path, settings: Settings) -> bool:
    """Whether this exact file has already been baked — i.e. no wait ahead."""
    key = _skp_cache_key(path)
    cache_path, _status = _skp_cache_paths(settings.skp_cache_dir(), key)
    return cache_path.exists()


def _ensure_skp_baked(path: Path, settings: Settings) -> dict:
    """The cached bake payload, or a `CadConversionPending`/`CadConversionError`.

    Starts the background bake at most once per content hash — a second
    request while one is running joins the same job rather than starting a
    duplicate 14-minute parse. `settings` comes from the caller (request-scoped
    dependency injection in the API, or the office's real settings singleton
    in the UI) rather than from a module-global lookup, the same way every
    other service in this codebase receives it — so tests can point the cache
    at an isolated `tmp_path` instead of the real `data/skp_cache`.
    """
    key = _skp_cache_key(path)
    cache_path, status_path = _skp_cache_paths(settings.skp_cache_dir(), key)
    if cache_path.exists():
        return _skp_read_payload(cache_path)

    registry_key = str(cache_path)
    with _SKP_BAKE_REGISTRY_LOCK:
        thread = _SKP_BAKE_THREADS.get(registry_key)
        if thread is None or not thread.is_alive():
            # A finished thread is gone from the registry either way (success
            # writes the cache file, caught above, or failure leaves only the
            # status file) — so getting here with a *failed* status on disk
            # means the previous attempt is done, not still running. Without
            # this check, every subsequent call would silently start a fresh
            # doomed parse instead of remembering that this file cannot be
            # read: a corrupt upload would burn a bake attempt on every page
            # load forever.
            previous = _skp_read_status(status_path)
            if previous is not None and previous.get("status") == "failed":
                raise CadConversionError(
                    f"That file could not be parsed as SKP: "
                    f"{previous.get('error') or 'unknown error'}"
                )
            thread = threading.Thread(
                target=_skp_bake_worker, args=(path, cache_path, status_path), daemon=True
            )
            _SKP_BAKE_THREADS[registry_key] = thread
            thread.start()

    status = _skp_read_status(status_path) or {
        "status": "baking", "stage": "starting", "started_at": datetime.now(UTC).isoformat(),
    }
    if status.get("status") == "failed":
        raise CadConversionError(
            f"That file could not be parsed as SKP: {status.get('error') or 'unknown error'}"
        )
    started = datetime.fromisoformat(str(status["started_at"]))
    elapsed = (datetime.now(UTC) - started).total_seconds()
    estimate = estimated_bake_seconds(path)
    raise CadConversionPending(
        f"Parsing and baking {path.name} — about {round(estimate / 60)} minutes "
        "for a file this size, and only this once. Try again shortly.",
        stage=str(status.get("stage", "starting")),
        elapsed_seconds=elapsed,
        estimated_seconds=estimate,
    )


# --- up-axis detection --------------------------------------------------------


def _skp_detect_up_axis(prims: list[dict], layer_of) -> int:
    """Which axis is up, decided from geometry rather than assumed.

    The baked glTF geometry is Y-up (glTF's convention) while SketchUp's own
    frame is Z-up — assuming either one put a camera edge-on to a large
    test file. Two independent witnesses:

      1. Ground-sheet layers (`Terrain source` etc.) are horizontal
         by construction, so their thinnest extent axis is up. Survives
         tower-shaped models, where the whole-model bbox does not.
      2. The whole-model bbox: sites are much wider than tall, so the
         thinnest axis is up. Fails on towers, hence witness 1 outranks it.

    Y is the fallback only when neither witness is conclusive.
    """
    import numpy as np

    ground = [p for p in prims if any(h in layer_of(p).lower() for h in _SKP_GROUND_LAYER_HINTS)]
    ground_axis: int | None = None
    if ground:
        pts = np.concatenate([p["positions"].reshape(-1, 3) for p in ground])
        ext = pts.max(axis=0) - pts.min(axis=0)
        order = np.argsort(ext)
        if ext[order[0]] < 0.5 * max(ext[order[1]], 1e-9):
            ground_axis = int(order[0])

    all_pts = np.concatenate([p["positions"].reshape(-1, 3) for p in prims])
    ext_all = all_pts.max(axis=0) - all_pts.min(axis=0)
    order_all = np.argsort(ext_all)
    bbox_axis: int | None = None
    if ext_all[order_all[0]] < 0.5 * max(ext_all[order_all[1]], 1e-9):
        bbox_axis = int(order_all[0])

    if ground_axis is not None:
        if bbox_axis is not None and bbox_axis != ground_axis:
            log.info(
                "SKP up-axis witnesses disagree (ground %d, bbox %d); trusting "
                "the ground sheet", ground_axis, bbox_axis,
            )
        return ground_axis
    if bbox_axis is not None:
        return bbox_axis
    return 1


# --- material resolution -------------------------------------------------------


def _skp_join_material_names(data: dict) -> dict[int, tuple]:
    """material_index -> (name, texture, tile_mm, albedo, pseudo_layers).

    Two things a naive colour->name join gets wrong:

      * openskp synthesises a pseudo-material `Layer_<name>` per layer and
        mixes it into `model.materials`. A colour matching one of those is a
        LAYER signal, not a material signal, and is split out and carried as
        a layer-name hint rather than a name candidate.
      * A colour shared by a placeholder name (`<auto>12`, `Heather_Soles`,
        `*`) and a real one is a real signal — the placeholder could not have
        claimed the surface, so it does not get to veto the name that can.
    """
    gltf = data["gltf_materials"]
    names_by_color = data["names_by_color"]
    tex_by_name = data["texture_by_name"]

    per_index: dict[int, tuple] = {}
    for idx, gm in enumerate(gltf):
        pbr = gm.get("pbrMetallicRoughness", {})
        base = pbr.get("baseColorFactor", [0.6, 0.6, 0.6, 1.0])
        albedo = (float(base[0]), float(base[1]), float(base[2]))
        rgb = tuple(round(c * 255) for c in albedo)
        candidates = names_by_color.get(rgb, [])

        pseudo_layers = tuple(
            sorted({c[len("Layer_"):] for c in candidates if c.startswith("Layer_")})
        )
        real = [
            c for c in candidates
            if not c.startswith("Layer_") and not is_placeholder_material(c)
        ]
        name = real[0] if len(real) == 1 else None
        texture, tile_mm = tex_by_name.get(name or "", (None, None))
        per_index[idx] = (name, texture, tile_mm, albedo, pseudo_layers)
    return per_index


def _skp_resolve_primitives(data: dict, layers: Sequence[str] | None) -> list[tuple[dict, MaterialSignal]]:
    """Attach a `MaterialSignal` to every primitive that passes the layer
    filter. Memoised per (material_index, layer): tens of thousands of
    primitives collapse to a few dozen distinct resolution inputs."""
    per_index = _skp_join_material_names(data)
    layer_by_geom = data["layer_by_geom"]
    wanted = {n.lower() for n in layers} if layers is not None else None

    cache: dict[tuple[int, str], MaterialSignal] = {}

    def _resolve_one(idx: int, layer: str) -> MaterialSignal:
        key = (idx, layer)
        if key in cache:
            return cache[key]
        name, texture, tile_mm, albedo, pseudo_layers = per_index[idx]
        signal = resolve(
            material_name=name, layer_name=layer,
            base_color=albedo if name else None,
            texture_name=texture, texture_width_mm=tile_mm,
        )
        if signal.confidence is Confidence.UNKNOWN:
            # The primitive's own layer said nothing (nested geometry
            # collapses onto Layer0), but the colour matched a Layer_X
            # pseudo-material — X is the FACE's layer, a deeper signal than
            # the instance layer. No base_color: a SketchUp tag colour is a
            # display palette, not an appearance.
            for pseudo in pseudo_layers:
                candidate = resolve(layer_name=pseudo)
                if candidate.confidence is not Confidence.UNKNOWN:
                    signal = candidate
                    break
        cache[key] = signal
        return signal

    resolved = []
    for prim in data["primitives"]:
        layer = layer_by_geom.get(prim["geom_name"], "")
        if not layer:
            layer = prim["geom_name"].rsplit("_", 1)[-1]
        if wanted is not None and layer.lower() not in wanted:
            continue
        resolved.append((prim, _resolve_one(prim["material_index"], layer)))
    return resolved


def _skp_build_groups(resolved: list[tuple[dict, MaterialSignal]]) -> list[dict]:
    """Merge primitives sharing a resolved appearance into single meshes.

    One pyrender mesh per primitive was ~5 minutes of render time on a large
    test file; grouped by (class, tier, appearance), tens of thousands of
    primitives collapse to a couple dozen meshes and rendering takes seconds.
    """
    import numpy as np

    groups: dict[tuple, dict] = {}
    for prim, signal in resolved:
        if not is_renderable(signal):
            # Drafting furniture is deleted, not painted — a revision cloud
            # has no correct appearance, and leaving it in puts floating
            # annotation text into an architectural photograph.
            continue
        entourage = is_entourage(signal)
        strength = _SKP_ENTOURAGE_CHANGE_STRENGTH if entourage else signal.change_strength
        key = (
            signal.material_class, signal.confidence, signal.albedo,
            round(signal.roughness, 3), round(signal.metallic, 3), entourage,
        )
        group = groups.setdefault(key, {
            "material_class": signal.material_class,
            "confidence": signal.confidence,
            "albedo": signal.albedo,
            "roughness": signal.roughness,
            "metallic": signal.metallic,
            "entourage": entourage,
            "change_strength": strength,
            "source": signal.source,
            "tile_width_mm": signal.tile_width_mm,
            "verts": [], "faces": [], "offset": 0, "prims": 0, "tris": 0,
        })
        verts = prim["positions"].reshape(-1, 3)
        faces = prim["indices"].reshape(-1, 3)
        group["verts"].append(verts)
        group["faces"].append(faces + group["offset"])
        group["offset"] += len(verts)
        group["prims"] += 1
        group["tris"] += len(faces)

    out = []
    for group in groups.values():
        group["vertices"] = np.concatenate(group["verts"])
        group["indices"] = np.concatenate(group["faces"])
        del group["verts"], group["faces"], group["offset"]
        out.append(group)
    out.sort(key=lambda g: -g["tris"])  # big groups first: stable-ish ID pass
    return out


# --- camera --------------------------------------------------------------------


def _skp_axis_components_to_vector(components: tuple[float, float, float], up_axis: int):
    import numpy as np

    ground_axes = [i for i in range(3) if i != up_axis]
    vec = np.zeros(3)
    vec[ground_axes[0]] = components[0]
    vec[ground_axes[1]] = components[1]
    vec[up_axis] = components[2]
    return vec


def _skp_up_vector_for(direction, up_axis: int):
    """World up-vector, or a ground-axis fallback when looking straight down
    (`_look_at` degenerates when the view direction is parallel to up)."""
    import numpy as np

    up = np.zeros(3)
    up[up_axis] = 1.0
    if np.linalg.norm(np.cross(direction / np.linalg.norm(direction), up)) < 1e-6:
        ground_axes = [i for i in range(3) if i != up_axis]
        up = np.zeros(3)
        up[ground_axes[0]] = 1.0
    return up


def _skp_robust_bbox(centers, areas, lo_q: float = 0.02, hi_q: float = 0.98, pad: float = 0.02):
    """Bbox of where the surface area actually is, not of every stray vertex.

    Per-axis percentiles of triangle centroids weighted by triangle AREA, so
    long thin outliers (an off-site column grid, import strays) cannot drag
    the frame while the surface that carries almost all the area cannot be
    cropped. Padded a few percent so the cut never shaves a real edge.
    """
    import numpy as np

    total = areas.sum()
    lo = np.empty(3)
    hi = np.empty(3)
    for axis in range(3):
        order = np.argsort(centers[:, axis])
        cum = np.cumsum(areas[order]) / max(total, 1e-12)
        lo[axis] = centers[order, axis][np.searchsorted(cum, lo_q)]
        hi[axis] = centers[order, axis][min(np.searchsorted(cum, hi_q), len(order) - 1)]
    margin = pad * float(np.linalg.norm(hi - lo))
    return lo - margin, hi + margin


def _skp_fit_camera(corners, center, direction, up, yfov: float, aspect: float):
    """Camera pose that fits every bbox corner in frame with a small margin.

    Derived, not tuned: for each corner, the distance at which it grazes the
    horizontal or vertical frustum plane is |lateral| / tan(half-fov) plus its
    offset toward the camera. The camera sits at the maximum over corners.
    """
    import numpy as np

    d_unit = direction / np.linalg.norm(direction)
    fwd = -d_unit
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, fwd)

    tan_y = np.tan(yfov / 2.0)
    tan_x = tan_y * aspect
    rel = corners - center
    toward = rel @ d_unit
    need_x = np.abs(rel @ right) / tan_x + toward
    need_y = np.abs(rel @ cam_up) / tan_y + toward
    dist = _SKP_FRAME_MARGIN * max(float(need_x.max()), float(need_y.max()), 1e-3)
    eye = center + d_unit * dist
    return _look_at(eye, center, up), dist


def _skp_weighted_median(values, weights) -> float:
    import numpy as np

    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, 0.5 * cum[-1])])


def _skp_ground_plane(groups, tms, pose, extent, site_lo, site_hi, up_axis: int, aspect: float):
    """A ground annulus from the site edge out past every frame edge.

    Without this, `controlnet_unions` reads the finite site slab's edge as a
    hard depth silhouette and renders the scheme as a physical diorama
    floating in the desert (measured, E1, 2026-08-07) — the slab drops
    straight to background where it ends, which is exactly an object edge to
    a depth-conditioned model.

    Level: the area-weighted median up-coordinate of up-facing ground-class
    faces (soil, asphalt, grass, gravel, water) — the slab is a solid box, so
    this MUST land on its TOP surface, or the plane hides the building rather
    than the cliff beneath it.

    Shape: an annulus, not a full quad. A quad at top-of-slab level would roof
    over any real terrain that dips below it (a gully channel); the ring
    starts just inside the slab edge and its hole leaves real terrain in
    charge everywhere a real surface exists.

    Size: every frame-border ray is intersected with the plane; the annulus
    extends 10% past the farthest hit, so its own edge is asserted to never
    land inside the frame.

    Returns (trimesh, group_dict, far_corner_distance) or None.
    """
    import numpy as np
    import trimesh

    ground_axes = [a for a in range(3) if a != up_axis]
    eye = pose[:3, 3]
    rot = pose[:3, :3]

    tri_list = []
    src = None
    src_area = 0.0
    for g, tm in zip(groups, tms):
        if g["entourage"] or g["material_class"] not in _SKP_GROUND_CLASSES:
            continue
        tri = np.asarray(tm.triangles, dtype=np.float64)
        if not len(tri):
            continue
        tri_list.append(tri)
        area = float(
            0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        )
        # Water is never the plane's source material: an infinite water plane
        # is a lake, not a desert floor.
        if g["material_class"] is not MaterialClass.WATER and area > src_area:
            src, src_area = g, area
    if not tri_list:
        return None

    tri = np.concatenate(tri_list)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(normals, axis=1)
    ok = areas > 1e-12
    tri, normals, areas = tri[ok], normals[ok], areas[ok]
    cents_up = tri.mean(axis=1)[:, up_axis]
    up_comp = normals[:, up_axis] / (2.0 * areas)

    up_facing = up_comp > 0.5
    if areas[up_facing].sum() < 0.05 * areas.sum():
        up_facing = np.abs(up_comp) > 0.5
    level = (
        _skp_weighted_median(cents_up[up_facing], areas[up_facing])
        if up_facing.any()
        else _skp_weighted_median(cents_up, areas)
    )
    if eye[up_axis] <= level + 1e-6:
        return None

    tan_y = np.tan(np.deg2rad(_SKP_YFOV_DEG) / 2.0)
    tan_x = tan_y * aspect
    ts = np.linspace(-1.0, 1.0, 65)
    ones = np.ones_like(ts)
    border = np.concatenate([
        np.stack([ts, -ones], axis=1), np.stack([ts, ones], axis=1),
        np.stack([-ones, ts], axis=1), np.stack([ones, ts], axis=1),
    ])
    dirs = np.stack([border[:, 0] * tan_x, border[:, 1] * tan_y, -np.ones(len(border))], axis=1)
    dirs = dirs @ rot.T
    down = dirs[:, up_axis] < -1e-9
    if not down.any():
        return None
    tvals = (level - eye[up_axis]) / dirs[down, up_axis]
    hits = eye[None, :] + tvals[:, None] * dirs[down]

    center = (site_lo + site_hi) / 2.0
    span = site_hi - site_lo
    cap = 60.0 * extent
    half = np.empty(2)
    for k, axis in enumerate(ground_axes):
        reach = 1.10 * float(np.abs(hits[:, axis] - center[axis]).max()) if len(hits) else 0.0
        # The ring must always enclose its own hole, however tight the frame.
        half[k] = max(min(reach, cap), 0.55 * float(span[axis]))

    drop = 5e-4 * extent  # hides the ring under coincident real ground; sub-pixel
    inset = 0.04 * span  # ring starts inside the slab edge so no gap can show
    g0, g1 = ground_axes

    def pt(u: float, v: float):
        p = np.empty(3)
        p[up_axis] = level - drop
        p[g0] = u
        p[g1] = v
        return p

    outer = [
        pt(center[g0] - half[0], center[g1] - half[1]),
        pt(center[g0] + half[0], center[g1] - half[1]),
        pt(center[g0] + half[0], center[g1] + half[1]),
        pt(center[g0] - half[0], center[g1] + half[1]),
    ]
    in_lo, in_hi = site_lo + inset, site_hi - inset
    inner = [
        pt(in_lo[g0], in_lo[g1]), pt(in_hi[g0], in_lo[g1]),
        pt(in_hi[g0], in_hi[g1]), pt(in_lo[g0], in_hi[g1]),
    ]
    verts = np.array(outer + inner)
    faces = np.array(
        [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
        dtype=np.int64,
    )
    # Faces must wind so normals point up: pyrender culls backfaces, and a
    # plane visible only from below is no plane at all.
    tri_p = verts[faces]
    flip = np.cross(tri_p[:, 1] - tri_p[:, 0], tri_p[:, 2] - tri_p[:, 0])[:, up_axis] < 0
    faces[flip] = faces[flip][:, ::-1]

    for c in outer:
        v = rot.T @ (c - eye)
        if v[2] < -1e-9:
            sx, sy = v[0] / (-v[2] * tan_x), v[1] / (-v[2] * tan_y)
            if abs(sx) <= 1.0 and abs(sy) <= 1.0:
                # A corner of the ground plane inside the frame means the plane
                # is too small and its own edge becomes a silhouette in the very
                # pass it exists to smooth. Cosmetic, so it is reported and the
                # render continues — this was an `assert`, which said nothing at
                # all under `python -O`.
                log.warning(
                    "Ground-plane corner lands inside the frame at (%.2f, %.2f); "
                    "the plane edge may show in the depth pass.", sx, sy,
                )

    if src is not None:
        albedo, rough, metal = src["albedo"], src["roughness"], src["metallic"]
    else:  # only water resolved as ground; fall back to the resolver's soil defaults
        albedo, rough, metal = (0.36, 0.28, 0.20), 0.95, 0.0

    group = {
        "material_class": src["material_class"] if src is not None else MaterialClass.SOIL,
        "confidence": Confidence.UNKNOWN,
        "albedo": tuple(albedo),
        "roughness": float(rough),
        "metallic": float(metal),
        "entourage": False,
        # Invented context, not a claim from the architect's file: the model
        # must be entirely free to put whatever desert it likes here.
        "change_strength": 1.0,
        "source": "synthetic ground plane (invented context)",
        "tile_width_mm": None,
        "synthetic": True,
        "prims": 1,
        "tris": len(faces),
    }
    far = float(np.linalg.norm(np.array(outer) - eye, axis=1).max())
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False), group, far


def _skp_subject_bbox(centers, areas, built_mask, up_axis: int, site_lo, site_hi):
    """Robust bbox of the built mass, from material class AND elevation.

    Neither signal alone works: elevation fails on undulating terrain (a gully
    can put 56% of faces above a single global grade), and class alone fails
    because "concrete" is both the building frame and every kerb and culvert
    on the site. A face is the subject if it resolved to a built class AND
    stands clear of the LOCAL ground beneath it — a coarse grid over the
    ground axes, each cell taking its own low percentile, survives terrain
    that is nowhere flat.
    """
    import numpy as np

    up_extent = float(site_hi[up_axis] - site_lo[up_axis])
    ground_axes = [a for a in range(3) if a != up_axis]

    n = 24
    lo2 = site_lo[ground_axes]
    span = np.maximum(site_hi[ground_axes] - lo2, 1e-9)
    cell = np.clip(((centers[:, ground_axes] - lo2) / span * n).astype(int), 0, n - 1)
    flat = cell[:, 0] * n + cell[:, 1]
    global_grade = float(np.percentile(centers[:, up_axis], 10))
    grade_of = np.full(n * n, global_grade)
    for c in np.unique(flat):
        sel = centers[flat == c, up_axis]
        if sel.size >= 8:
            grade_of[c] = float(np.percentile(sel, 10))

    clear = centers[:, up_axis] > grade_of[flat] + _SKP_SUBJECT_MIN_HEIGHT_FRAC * up_extent
    subject = clear & built_mask
    if subject.sum() < 64:
        return site_lo, site_hi, (site_lo + site_hi) / 2.0

    lo, hi = _skp_robust_bbox(centers[subject], areas[subject], lo_q=0.05, hi_q=0.95)
    # Pull the bottom down to the ground the buildings stand on, so the frame
    # is of a building on a site rather than a floating slab of wall.
    lo[up_axis] = min(lo[up_axis], global_grade)
    # Aim at where the built area actually is, not the middle of its bbox —
    # two bars either side of a gully put the bbox centre in the empty channel
    # between them.
    w = areas[subject]
    target = (centers[subject] * w[:, None]).sum(axis=0) / max(w.sum(), 1e-12)
    return lo, hi, target


# --- render --------------------------------------------------------------------


def _render_skp_passes(
    groups: list[dict],
    camera: str,
    up_axis: int,
    *,
    subject: str = "site",
    zoom: float = 1.0,
    size: tuple[int, int] = _SKP_CONVERT_SIZE,
    full: bool = True,
) -> dict[str, object]:
    """(view, depth, albedo, change) PNGs plus the materials-panel data.

    `full=False` (the preview path) skips the ID pass and the albedo pass —
    the two extra `OffscreenRenderer` calls a materials-correct render needs
    but a "just show me the massing" preview does not.
    """
    import numpy as np
    import pyrender
    import trimesh
    from PIL import Image

    cents_list = []
    areas_list = []
    for g in groups:
        tri = g["vertices"][g["indices"]]
        cents_list.append(tri.mean(axis=1))
        areas_list.append(
            0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        )

    cents_all = np.concatenate(cents_list)
    areas_all = np.concatenate(areas_list)
    all_lo, all_hi = _skp_robust_bbox(cents_all, areas_all)
    extent = float(np.linalg.norm(all_hi - all_lo))
    if extent <= 0:
        raise CadConversionError("This file's geometry has no size to frame a camera around.")

    if subject == "buildings":
        built_mask = np.concatenate([
            np.full(len(c), g["material_class"] in _SKP_BUILT_CLASSES and not g["entourage"])
            for g, c in zip(groups, cents_list)
        ])
        frame_lo, frame_hi, center = _skp_subject_bbox(
            cents_all, areas_all, built_mask, up_axis, all_lo, all_hi
        )
    else:
        frame_lo, frame_hi = all_lo, all_hi
        center = (frame_lo + frame_hi) / 2.0
    corners = np.array(
        [[frame_lo[i] if bit & (1 << i) else frame_hi[i] for i in range(3)] for bit in range(8)]
    )

    # Cull geometry far outside the framed volume — leaving off-site strays in
    # the scene lets a low camera end up standing among them.
    cull_lo = all_lo - 0.10 * (all_hi - all_lo)
    cull_hi = all_hi + 0.10 * (all_hi - all_lo)
    active: list[dict] = []
    tms = []
    for g, cents in zip(groups, cents_list):
        keep_faces = np.all((cents >= cull_lo) & (cents <= cull_hi), axis=1)
        if not keep_faces.any():
            continue
        active.append(g)
        tms.append(trimesh.Trimesh(vertices=g["vertices"], faces=g["indices"][keep_faces], process=False))
    groups = active

    direction = _skp_axis_components_to_vector(_SKP_PRESET_COMPONENTS[camera], up_axis)
    up = _skp_up_vector_for(direction, up_axis)
    aspect = size[0] / size[1]
    pose, dist = _skp_fit_camera(corners, center, direction, up, np.deg2rad(_SKP_YFOV_DEG), aspect)
    if zoom != 1.0:
        dist /= zoom
        pose = _look_at(center + direction / np.linalg.norm(direction) * dist, center, up)

    plane = _skp_ground_plane(groups, tms, pose, extent, all_lo, all_hi, up_axis, aspect)
    far_plane = 0.0
    if plane is not None:
        plane_tm, plane_group, far_plane = plane
        groups = groups + [plane_group]
        tms = tms + [plane_tm]

    if len(groups) > 255:
        # The ID pass encodes the appearance group in 8 bits, so group 256 would
        # silently be read back as group 0 and paint one material with another's
        # albedo. This was an `assert`, which `python -O` strips — leaving the
        # corruption with nothing to stop it.
        raise CadConversionError(
            f"That model resolves to {len(groups)} distinct appearances, and the "
            "material pass can encode 255. Hide some layers and convert again."
        )

    cam = pyrender.PerspectiveCamera(
        yfov=np.deg2rad(_SKP_YFOV_DEG),
        znear=max(0.05, dist * 0.002),
        # zfar must reach the plane's far corners or the clip would cut a NEW
        # silhouette into the very pass the plane exists to smooth.
        zfar=max(dist + extent * 2.0, 1.05 * far_plane),
    )
    # Cross-light: ~90 deg in azimuth from the camera, mid-morning elevation.
    # Sun in the camera's own quadrant lit every facade identically and the
    # clay read as a single pale mass; cross-lit, the massing reads.
    sun_vec = _skp_axis_components_to_vector((0.3, -1.5, 1.0), up_axis)
    sun_pose = _look_at(center + sun_vec / np.linalg.norm(sun_vec) * dist, center, up)

    def _render(mesh_specs, bg, ambient, seg=False, sun=True):
        scene = pyrender.Scene(bg_color=bg, ambient_light=ambient)
        seg_map = {}
        for i, (tm, material) in enumerate(mesh_specs):
            mesh = pyrender.Mesh.from_trimesh(tm, material=material, smooth=False)
            node = scene.add(mesh)
            seg_map[node] = (i + 1, i + 1, i + 1)
        scene.add(cam, pose=pose)
        if sun and not seg:
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=sun_pose)
        renderer = pyrender.OffscreenRenderer(*size)
        try:
            if seg:
                color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SEG, seg_node_map=seg_map)
            else:
                # RenderFlags.SHADOWS_DIRECTIONAL crashes inside pyrender's
                # shadow-map path on this Windows GL stack (ctypes
                # ArgumentError, measured on the test machine). Clay ships unshadowed.
                color, depth = renderer.render(scene)
        finally:
            renderer.delete()
        return color, depth

    def _png(arr) -> bytes:
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    with _RENDER_LOCK:
        clay_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.82, 0.81, 0.78, 1.0], metallicFactor=0.0, roughnessFactor=0.9
        )
        color, depth = _render([(tm, clay_mat) for tm in tms], [255, 255, 255, 255], [0.25] * 3)

        mask = depth > 0

        # Stretch the clay's tonal range. See the note on _SKP_CLAY_RANGE.
        if mask.any():
            lum = color[..., :3].astype(np.float32) @ np.array(
                [0.2126, 0.7152, 0.0722], dtype=np.float32
            )
            lo = float(np.percentile(lum[mask], 2))
            hi = float(np.percentile(lum[mask], 98))
            if hi - lo >= 1.0:
                t_lo, t_hi = _SKP_CLAY_RANGE
                stretched = color.astype(np.float32)
                # Colour channels only — an alpha channel, if present, must
                # not be remapped into translucency.
                region = stretched[..., :3]
                region[mask] = np.clip(
                    (region[mask] - lo) * ((t_hi - t_lo) / (hi - lo)) + t_lo,
                    0.0,
                    255.0,
                )
                color = stretched.astype(np.uint8)

        out: dict[str, object] = {"view": _png(color), "materials": []}
        dimg = np.zeros_like(depth, dtype=np.uint8)
        real_lut = np.zeros(len(groups) + 1, dtype=bool)
        for i, g in enumerate(groups):
            real_lut[i + 1] = not g.get("synthetic", False)

        if not full:
            if mask.any():
                dmin, dmax = float(depth[mask].min()), float(depth[mask].max())
                dimg[mask] = (255 * (1.0 - (depth[mask] - dmin) / max(dmax - dmin, 1e-9))).astype(np.uint8)
            out["depth"] = _png(dimg)
            return out

        # ID pass: exact per-group pixel attribution. Masks the depth
        # normalization, is the change-map lookup, and is the pixel count
        # behind the materials-panel data — unlit and exact in all three
        # roles (pyrender SEG mode: no lighting, no multisampling).
        id_color, _ = _render([(tm, None) for tm in tms], [0, 0, 0, 255], [1.0] * 3, seg=True)
        ids = id_color[..., 0].astype(np.intp)
        real_px = real_lut[ids]

        if mask.any():
            # Normalize over the BUILT subject, not over everything real.
            #
            # Excluding only the synthetic ground plane is not enough, because
            # the architect's own file models the site: paving, kerbs, soil,
            # contours, all of it real geometry running to the horizon. On a
            # 27MB test model that put 99% of the 8-bit range on the
            # ground and left the buildings separated from the tarmac beside
            # them by 19.9 grey levels, 7.8% of the range (measured
            # 2026-08-27). A depth ControlNet reading that sees a smooth
            # gradient with a faint bump in it, which is the opposite of the
            # "hold the geometry" promise this pass exists to keep.
            #
            # Scaling to the buildings gives the massing the range and lets the
            # distant ground clamp flat, which is the correct trade for a
            # control image: near-field building depth is the signal, the
            # receding site is not.
            built_lut = np.zeros(len(groups) + 1, dtype=bool)
            for i, g in enumerate(groups):
                built_lut[i + 1] = (
                    not g.get("synthetic", False)
                    and not g["entourage"]
                    and g["material_class"] in _SKP_BUILT_CLASSES
                )
            built_mask = mask & built_lut[ids]
            built_depth = depth[built_mask]
            real_depth = depth[mask & real_px]

            # ...except that on an oblique or aerial camera, rescaling cannot
            # fix it either, because there is nothing to rescale: a building
            # and the tarmac beside it are at the SAME distance, so no choice
            # of dmin/dmax separates them. Measured on this exact model
            # (2026-08-31): inside the geometry only 18.2% of pixels differed
            # by more than 8 levels from a plain per-row ground ramp, and
            # outside it the pass was a ramp to within a standard deviation of
            # 0.07 levels. What reached fal was, over 82% of the frame, an
            # instruction reading "empty plane receding to the horizon".
            #
            # What a depth ControlNet needs here is height above ground, not
            # distance from camera. For a planar ground under perspective,
            # INVERSE depth is exactly an affine function of pixel coordinates
            # — not approximately, exactly — so the ground can be fitted as a
            # plane in 1/z and subtracted, leaving each pixel's disparity above
            # ground. Fitting rather than assuming a horizontal horizon keeps
            # this correct for a rotated or tilted camera.
            ground_mask = mask & real_px & ~built_mask
            deramped = None
            if ground_mask.sum() >= _SKP_DEPTH_MIN_GROUND_PX and built_depth.size:
                yy, xx = np.nonzero(mask)
                gy, gx = np.nonzero(ground_mask)
                # Least squares on 1/z over ground pixels. rcond=None silences
                # the numpy>=1.14 deprecation and takes machine-precision rank.
                A = np.column_stack([gx, gy, np.ones(gx.size)]).astype(np.float64)
                inv_ground = 1.0 / np.maximum(depth[ground_mask], 1e-6)
                coef, _res, rank, _sv = np.linalg.lstsq(A, inv_ground, rcond=None)
                if rank == 3:
                    inv_all = 1.0 / np.maximum(depth[mask], 1e-6)
                    plane = coef[0] * xx + coef[1] * yy + coef[2]
                    deramped = inv_all - plane      # >0 = above the ground plane
                    plane_span = float(plane.max() - plane.min())

            wrote_deramped = False
            if deramped is not None:
                ref_res = deramped[built_lut[ids][mask]]
                if ref_res.size >= _SKP_DEPTH_MIN_SUBJECT_PX:
                    lo = float(np.percentile(ref_res, 1.0))
                    hi = float(np.percentile(ref_res, 99.0))
                    # Relative, not absolute. `_SKP_DEPTH_MIN_SPREAD` is in raw
                    # depth units and these residuals are disparities — on this
                    # model the built relief is 2.3e-4 against a 1e-3 threshold,
                    # so an absolute test silently sent every oblique view to
                    # the fallback and produced the ground ramp this branch
                    # exists to remove. What "too flat to normalize" actually
                    # means is that the subject stands no meaningful height
                    # above the ground, so it is measured against the ground
                    # plane's own span across the frame.
                    if hi - lo >= _SKP_DEPTH_MIN_RELIEF_FRACTION * plane_span:
                        # Near = bright, as the ControlNet convention wants:
                        # a residual above the ground plane is nearer than the
                        # ground, so it is already the right way round.
                        dimg[mask] = (
                            255 * np.clip((deramped - lo) / (hi - lo), 0, 1)
                        ).astype(np.uint8)
                        wrote_deramped = True

            # Fallback: no usable ground plane (a pure elevation, an interior,
            # a model that is all building). Range-scale to the built subject,
            # which is the right answer when there is no ground ramp to remove.
            if not wrote_deramped:
                ref = (
                    built_depth if built_depth.size >= _SKP_DEPTH_MIN_SUBJECT_PX
                    else real_depth if real_depth.size
                    else depth[mask]
                )
                dmin = float(np.percentile(ref, 0.5))
                dmax = float(np.percentile(ref, 99.5))
                # A perfectly flat subject (a single roof plane, seen from
                # above) would divide by nothing; widen to the real geometry.
                if dmax - dmin < _SKP_DEPTH_MIN_SPREAD and real_depth.size:
                    dmin = float(np.percentile(real_depth, 0.5))
                    dmax = float(np.percentile(real_depth, 99.5))
                clamped = np.clip(depth[mask], dmin, dmax)
                dimg[mask] = (
                    255 * (1.0 - (clamped - dmin) / max(dmax - dmin, 1e-9))
                ).astype(np.uint8)
        out["depth"] = _png(dimg)

        # Albedo: unlit — a shadowed face rendered near-black would smuggle a
        # lighting claim into a material pass. Entourage is skipped rather
        # than painted; a flat-shaded SketchUp cutout survives refinement as
        # a flat-shaded cutout, and leaving the region to the model produces
        # better people than this pipeline can paint.
        #
        # COLOUR SPACE: `g["albedo"]` is sRGB, `baseColorFactor` is linear
        # (glTF spec). Passing the first as the second gamma-encodes it twice
        # and lifts the whole pass toward white — on a large test
        # model that painted mid-tone colours visibly lighter, i.e. a flat cream image, which is
        # exactly the "washed-out" render `skp_material_render` was blamed for.
        # Measured 2026-08-31; see `materials.srgb_to_linear`.
        #
        # The clay `view` pass above is deliberately NOT converted: it is a lit
        # geometry plate, not a material claim, its grey is a styling choice
        # rather than a measured colour, and it is the conditioning input for
        # tasks that are already launch-ready on their current tuning.
        albedo_specs = [
            (tm, pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*srgb_to_linear(g["albedo"]), 1.0],
                metallicFactor=g["metallic"], roughnessFactor=g["roughness"],
            ))
            for tm, g in zip(tms, groups) if not g["entourage"]
        ]
        albedo_color, _ = _render(albedo_specs, [255, 255, 255, 255], [1.0] * 3, sun=False)
        out["albedo"] = _png(albedo_color)

        # Change map: per-group confidence, looked up through the same ID
        # pass — unlit and exact, because a shadow falling across a wall must
        # not change how freely the model may repaint it.
        #
        # POLARITY: fal's change_map_image_url is BLACK = repaint,
        # WHITE = preserve. That is the opposite of the reading this file and
        # materials.py shipped with, and it is why every render of this task
        # came back as the albedo handed straight back. Measured 2026-08-31,
        # six probes at 640x480, one fixed seed, one variable each:
        #
        #   change map all WHITE  -> MAE 2.45 vs the input albedo   (preserved)
        #   change map all BLACK  -> MAE 14.77, std 27.2            (rendered)
        #
        # and with the old polarity nothing else could move the result:
        # de-ramping the depth pass, dropping depth conditioning entirely, and
        # rewriting the prompt all landed within MAE 0.1 of the baseline,
        # because a preserved image has nothing left for them to act on. The
        # earlier "depth scale 0.6 vs 0.3 shows real improvement" reading was
        # seed noise — the two differ by less than either differs from the
        # input.
        #
        # `change_strength` stays as it reads everywhere else in the codebase
        # (1.0 = repaint freely), so the inversion lives here, at the one place
        # the value becomes a fal-convention pixel.
        strength_lut = np.zeros(len(groups) + 1, dtype=np.uint8)
        # sky/background: no material claim, reinvent freely
        strength_lut[0] = change_map_pixel(1.0)
        for i, g in enumerate(groups):
            strength_lut[i + 1] = change_map_pixel(g["change_strength"])
        out["change"] = _png(strength_lut[ids])

        # Materials-panel data: each group's share of VISIBLE pixels, real
        # geometry only (the synthetic ground plane is invented context, not
        # a claim from the architect's file, and is excluded here the same
        # way it is excluded from `depth`'s normalization range above).
        counts = np.bincount(ids.ravel(), minlength=len(groups) + 1)
        synth_px = sum(int(counts[i + 1]) for i, g in enumerate(groups) if g.get("synthetic", False))
        visible = int(counts[1:].sum()) - synth_px
        usage: list[MaterialUsage] = []
        if visible:
            for i, g in enumerate(groups):
                if g.get("synthetic", False) or not counts[i + 1]:
                    continue
                usage.append(MaterialUsage(
                    material_class=str(g["material_class"].value),
                    confidence=str(g["confidence"].value),
                    albedo=tuple(g["albedo"]),
                    share_of_visible_pixels=int(counts[i + 1]) / visible,
                    source=str(g["source"]),
                    tile_width_mm=g.get("tile_width_mm"),
                ))
            usage.sort(key=lambda u: -u.share_of_visible_pixels)
        out["materials"] = usage

    return out


class SkpConverter:
    """SketchUp SKP → clay/depth/albedo/change, via openskp → pyrender.

    Reverse-engineered parser (openskp, MIT), validated against a real
    model file rather than trusted by reading its source; see the module
    docstring above for the pass contract.

    Every method may raise `CadConversionPending` while the first look at a
    given file is still baking in the background — the caller (the API
    router) is expected to turn that into a 202 a client polls past, exactly
    like a fal.ai render job.

    `settings` is optional and resolved lazily to the office's real settings
    on first use if omitted, so `CONVERTERS[".skp"]` (built at import time,
    before `.env` is necessarily loaded) stays cheap to construct. The API
    router passes its request-scoped `settings` explicitly instead of relying
    on that default, so a test's isolated `tmp_path` settings are actually
    used rather than shadowed by the process-wide singleton.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings

    def _resolved_settings(self) -> Settings:
        return self._settings or get_settings()

    def layers(self, path: Path) -> list[LayerInfo]:
        """Only the layers that actually carry geometry.

        A SketchUp file's layer table and its geometry's layer assignments are
        not the same thing. Anything nested inside a group or component reports
        the group's layer, or Layer0, rather than the layer the architect
        assigned in the outliner, and openskp gives us the entity's own answer.

        On a 27MB test model that gap is total: 61 layers declared,
        and exactly 6 of them ever appear on a primitive. Listing all 61 offered
        the user 55 checkboxes that select nothing, and choosing one of
        them refused the whole conversion with "none of the selected layers have
        anything on them" — which reads as a bug in their file rather than a
        limit of the parser. Measured 2026-08-27.

        So the manifest reports what can actually be selected. The layers that
        exist but hold nothing are still in `data["layers"]` for anyone who
        needs them; they are simply not offered as controls.
        """
        data = _ensure_skp_baked(path, self._resolved_settings())
        layer_of = self._layer_of(data)
        populated: dict[str, int] = {}
        for prim in data["primitives"]:
            name = layer_of(prim)
            if name:
                populated[name] = populated.get(name, 0) + 1

        # Declared order first, so the list reads the way the file does, then
        # anything the geometry carries that the layer table never declared.
        declared = [n for n in data["layers"] if n in populated]
        extra = sorted(n for n in populated if n not in set(data["layers"]))
        return [
            LayerInfo(name=n, color=None, visible=True) for n in (*declared, *extra)
        ]

    def preview(
        self,
        path: Path,
        *,
        layers: Sequence[str] | None = None,
        camera: str = "iso",
    ) -> bytes:
        data = _ensure_skp_baked(path, self._resolved_settings())
        base_camera, zoom, subject = self._camera_params(camera)
        resolved = _skp_resolve_primitives(data, layers)
        groups = _skp_build_groups(resolved)
        if not groups:
            raise CadConversionError(
                "None of the selected layers have anything on them. Choose at "
                "least one layer with geometry."
                if layers is not None
                else "This file has no renderable geometry."
            )
        up_axis = _skp_detect_up_axis(data["primitives"], self._layer_of(data))
        passes = _render_skp_passes(
            groups, base_camera, up_axis, subject=subject, zoom=zoom,
            size=_SKP_PREVIEW_SIZE, full=False,
        )
        return passes["view"]  # type: ignore[return-value]

    def convert(
        self,
        path: Path,
        *,
        layers: Sequence[str] | None = None,
        line_weight_scale: float = 1.0,  # meaningless for a mesh render; accepted, ignored
        camera: str = "iso",
    ) -> ConversionResult:
        data = _ensure_skp_baked(path, self._resolved_settings())
        base_camera, zoom, subject = self._camera_params(camera)
        known = set(data["layers"])
        warnings = [
            f"Layer {name!r} was requested but does not exist in this file."
            for name in layers or []
            if name not in known
        ]
        resolved = _skp_resolve_primitives(data, layers)
        groups = _skp_build_groups(resolved)
        if not groups:
            raise CadConversionError(
                "None of the selected layers have anything on them. Choose at "
                "least one layer with geometry."
                if layers is not None
                else "This file has no renderable geometry."
            )
        up_axis = _skp_detect_up_axis(data["primitives"], self._layer_of(data))
        passes = _render_skp_passes(
            groups, base_camera, up_axis, subject=subject, zoom=zoom,
            size=_SKP_CONVERT_SIZE, full=True,
        )
        return ConversionResult(
            view_png=passes["view"],  # type: ignore[arg-type]
            depth_png=passes["depth"],  # type: ignore[arg-type]
            albedo_png=passes["albedo"],  # type: ignore[arg-type]
            change_png=passes["change"],  # type: ignore[arg-type]
            camera=CameraMeta(preset=camera),
            layers=[LayerInfo(name=n, color=None, visible=True) for n in data["layers"]],
            warnings=warnings,
            materials=passes["materials"],  # type: ignore[arg-type]
        )

    @staticmethod
    def _layer_of(data: dict):
        layer_by_geom = data["layer_by_geom"]

        def layer_of(prim: dict) -> str:
            return layer_by_geom.get(prim["geom_name"], "") or prim["geom_name"].rsplit("_", 1)[-1]

        return layer_of

    @staticmethod
    def _camera_params(camera: str) -> tuple[str, float, str]:
        """`camera` -> (base preset, zoom, subject). Raises `CadConversionError`
        for anything outside `SKP_CAMERA_PRESETS`."""
        if camera in _SKP_SUBJECT_CAMERAS:
            base, zoom = _SKP_SUBJECT_CAMERAS[camera]
            return base, zoom, "buildings"
        if camera in _SKP_PRESET_COMPONENTS:
            return camera, 1.0, "site"
        raise CadConversionError(
            f"Unknown camera preset {camera!r}. Choose one of: {', '.join(SKP_CAMERA_PRESETS)}."
        )


CONVERTERS: dict[str, CadConverter] = {
    ".dxf": DxfConverter(),
    ".skp": SkpConverter(),
    ".ifc": IfcConverter(),
}


def converter_for(suffix: str, settings: Settings | None = None) -> CadConverter:
    """The converter for a file suffix.

    `settings` is SKP-only — it needs a request-scoped (or test-isolated)
    settings object for its bake cache, and is otherwise ignored. Passing it
    always returns a fresh `SkpConverter`, cheap to construct, rather than
    mutating the shared `CONVERTERS[".skp"]` singleton in place.
    """
    suffix = suffix.lower()
    if suffix not in CONVERTERS:
        raise ValueError(f"No CAD converter for {suffix!r}. Known: {sorted(CONVERTERS)}.")
    if suffix == ".skp" and settings is not None:
        return SkpConverter(settings)
    return CONVERTERS[suffix]
