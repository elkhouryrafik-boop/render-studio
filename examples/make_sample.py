"""Generate the synthetic sample inputs for render-studio.

    uv run python examples/make_sample.py --out examples/inputs

Everything here is invented. The "building" is a three-volume massing (a podium
and two towers) drawn with Pillow's painter's algorithm; the plan is a made-up
three-room layout; the IFC is a single storey with four walls and a slab.
Output is deterministic for the PNG and DXF files (fixed seed, no clock).

Files written:
    massing-aerial.png     axonometric view, 1536x1024
    massing-street.png     two-point-perspective view, 1536x1024
    sketch-street.png      the street view as a jittered pencil sketch
    elevation-sample.png   orthographic elevation with a window grid
    plan-sample.dxf        3-room plan, layers WALLS DOORS WINDOWS TEXT DIMS
    model-sample.ifc       one storey, 4 walls, slab (skipped if ifcopenshell is missing)
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1536, 1024
CAPTION = "Synthetic massing — render-studio sample"
STOREY = 3.0  # metres

# name, (x0, y0, z0), (x1, y1, z1) in metres. Podium 40x24x6, towers of 12 and 18 storeys.
PODIUM = ((0.0, 0.0, 0.0), (40.0, 24.0, 6.0))
TOWER_A = ((4.0, 6.0, 6.0), (16.0, 18.0, 6.0 + 12 * STOREY))
TOWER_B = ((24.0, 6.0, 6.0), (36.0, 18.0, 6.0 + 18 * STOREY))
BOXES = (PODIUM, TOWER_A, TOWER_B)

Vec = tuple[float, float, float]


# --- tiny vector helpers -----------------------------------------------------


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vec) -> Vec:
    n = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


class Camera:
    """Pinhole (perspective=True) or parallel (perspective=False) camera, z up."""

    def __init__(self, eye: Vec, target: Vec, *, perspective: bool, scale: float,
                 centre: tuple[float, float] = (W / 2, H / 2 - 30)) -> None:
        self.eye = eye
        self.perspective = perspective
        self.scale = scale
        self.centre = centre
        self.fwd = _norm(_sub(target, eye))
        self.right = _norm(_cross(self.fwd, (0.0, 0.0, 1.0)))
        self.up = _cross(self.right, self.fwd)

    def project(self, p: Vec) -> tuple[float, float]:
        d = _sub(p, self.eye)
        x, y, z = _dot(d, self.right), _dot(d, self.up), _dot(d, self.fwd)
        if self.perspective:
            z = max(z, 0.1)
            x, y = x / z, y / z
        return (self.centre[0] + x * self.scale, self.centre[1] - y * self.scale)

    def depth(self, p: Vec) -> float:
        return _dot(_sub(p, self.eye), self.fwd)

    def facing(self, point_on_face: Vec, normal: Vec) -> bool:
        view = _sub(point_on_face, self.eye) if self.perspective else self.fwd
        return _dot(view, normal) < 0


# --- geometry ----------------------------------------------------------------


def _faces(box):
    """Six faces of an axis-aligned box: (name, normal, 4 corners)."""
    (x0, y0, z0), (x1, y1, z1) = box
    return [
        ("top", (0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
        ("south", (0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
        ("north", (0, 1, 0), [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]),
        ("west", (-1, 0, 0), [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)]),
        ("east", (1, 0, 0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
    ]


def _face_grey(name: str) -> tuple[int, int, int]:
    return {
        "top": (226, 226, 226),
        "south": (172, 172, 172),
        "north": (150, 150, 150),
        "west": (140, 140, 140),
        "east": (190, 190, 190),
    }[name]


def _visible_faces(cam: Camera):
    """Faces in painter's order (far boxes first, podium before towers)."""
    order = sorted(
        BOXES,
        key=lambda b: (b[0][2] > 0.0, -cam.depth(tuple((b[0][i] + b[1][i]) / 2 for i in range(3)))),
    )
    for box in order:
        for name, normal, corners in _faces(box):
            centre = tuple(sum(c[i] for c in corners) / 4 for i in range(3))
            if cam.facing(centre, normal):
                yield box, name, normal, corners


def _storey_lines(box, name, corners):
    """Horizontal storey lines (3 m grid) on the sides of a tower."""
    if name == "top" or box is PODIUM:
        return []
    z0, z1 = box[0][2], box[1][2]
    lines = []
    z = z0 + STOREY
    while z < z1 - 1e-6:
        a = [c for c in corners if abs(c[2] - z0) < 1e-6]
        a0, a1 = a[0], a[1]
        lines.append(((a0[0], a0[1], z), (a1[0], a1[1], z)))
        z += STOREY
    return lines


def _font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _caption(img: Image.Image, bg=(30, 30, 30), fg=(235, 235, 235)) -> None:
    d = ImageDraw.Draw(img)
    d.rectangle([0, img.height - 40, img.width, img.height], fill=bg)
    d.text((16, img.height - 31), CAPTION, fill=fg, font=_font(18))


def _clip_near(cam: Camera, pts: list[Vec], near: float = 1.0) -> list[Vec]:
    """Clip a 3D polygon/polyline against the camera's near plane (perspective only)."""
    if not cam.perspective:
        return pts
    out: list[Vec] = []
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        dp, dq = cam.depth(p) - near, cam.depth(q) - near
        if dp >= 0:
            out.append(p)
        if (dp >= 0) != (dq >= 0):
            s = dp / (dp - dq)
            out.append(tuple(p[k] + (q[k] - p[k]) * s for k in range(3)))
    return out


def _ground(cam: Camera, d: ImageDraw.ImageDraw, ext: float = 40.0) -> None:
    x_min, x_max, y_min, y_max = -ext, 40 + ext, -ext, 24 + ext
    quad = _clip_near(cam, [(x_min, y_min, 0), (x_max, y_min, 0), (x_max, y_max, 0),
                            (x_min, y_max, 0)])
    d.polygon([cam.project(p) for p in quad], fill=(205, 208, 205))
    grid = (192, 195, 192)
    x = x_min
    while x <= x_max + 1e-6:  # 6 m grid, light
        seg = _clip_near(cam, [(x, y_min, 0), (x, y_max, 0)])
        if len(seg) == 2:
            d.line([cam.project(p) for p in seg], fill=grid)
        x += 6.0
    y = y_min
    while y <= y_max + 1e-6:
        seg = _clip_near(cam, [(x_min, y, 0), (x_max, y, 0)])
        if len(seg) == 2:
            d.line([cam.project(p) for p in seg], fill=grid)
        y += 6.0


# --- renderers ---------------------------------------------------------------


def draw_massing(cam: Camera, sky=(238, 241, 244), ground_ext: float = 40.0) -> Image.Image:
    img = Image.new("RGB", (W, H), sky)
    d = ImageDraw.Draw(img)
    _ground(cam, d, ground_ext)
    for box, name, _n, corners in _visible_faces(cam):
        poly = [cam.project(c) for c in corners]
        d.polygon(poly, fill=_face_grey(name), outline=(70, 70, 70))
        for a, b in _storey_lines(box, name, corners):
            d.line([cam.project(a), cam.project(b)], fill=(120, 120, 120), width=1)
        d.line(poly + [poly[0]], fill=(70, 70, 70), width=1)
    _caption(img)
    return img


def draw_sketch(cam: Camera, seed: int = 33) -> Image.Image:
    """Same massing as pencil lines: seeded jitter, overshoot, double strokes."""
    rng = random.Random(seed)
    img = Image.new("RGB", (W, H), (247, 244, 236))
    d = ImageDraw.Draw(img)

    def stroke(p, q, grey=(88, 88, 92), width=2, overshoot=14.0):
        (x0, y0), (x1, y1) = p, q
        length = math.hypot(x1 - x0, y1 - y0) or 1.0
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        for _ in range(2):
            o0, o1 = rng.uniform(0, overshoot), rng.uniform(0, overshoot)
            a = (x0 - ux * o0 + rng.gauss(0, 1.6), y0 - uy * o0 + rng.gauss(0, 1.6))
            b = (x1 + ux * o1 + rng.gauss(0, 1.6), y1 + uy * o1 + rng.gauss(0, 1.6))
            mid = ((a[0] + b[0]) / 2 + rng.gauss(0, 1.4), (a[1] + b[1]) / 2 + rng.gauss(0, 1.4))
            shade = rng.randint(-14, 18)
            g = tuple(min(255, c + shade) for c in grey)
            d.line([a, mid, b], fill=g, width=width, joint="curve")

    # a few ground lines, loose
    for i in range(-3, 7):
        y = i * 8.0
        stroke(cam.project((-30, y, 0)), cam.project((70, y, 0)), grey=(200, 198, 192), width=1)
    for box, name, _n, corners in _visible_faces(cam):
        poly = [cam.project(c) for c in corners]
        d.polygon(poly, fill=(247, 244, 236))  # occlude what is behind
        if name in ("west", "north"):  # light hatching on the shaded sides
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            for k in range(int(min(xs)), int(max(xs)), 9):
                p, q = (k, min(ys)), (k - 24, max(ys))
                if all(_inside(poly, (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
                       for t in (0.25, 0.5, 0.75)):
                    d.line([p, q], fill=(214, 212, 206), width=1)
        for a, b in _storey_lines(box, name, corners):
            stroke(cam.project(a), cam.project(b), grey=(150, 150, 152), width=1, overshoot=0)
        for i in range(4):
            stroke(poly[i], poly[(i + 1) % 4])
    _caption(img, bg=(60, 58, 54))
    return img


def _inside(poly, pt) -> bool:
    x, y = pt
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def draw_elevation() -> Image.Image:
    """South elevation, orthographic, with a window grid on every storey."""
    img = Image.new("RGB", (W, H), (238, 241, 244))
    d = ImageDraw.Draw(img)
    scale = 14.0  # px per metre
    x_off, ground_y = (W - 40 * scale) / 2, H - 150

    def P(x, z):
        return (x_off + x * scale, ground_y - z * scale)

    d.rectangle([0, ground_y, W, H - 40], fill=(205, 208, 205))
    d.line([(0, ground_y), (W, ground_y)], fill=(70, 70, 70), width=2)

    def block(x0, x1, z0, z1, fill, win_w=1.8, win_h=1.5, pitch=3.0):
        d.rectangle([P(x0, z1), P(x1, z0)], fill=fill, outline=(70, 70, 70), width=1)
        z = z0 + STOREY
        while z <= z1 + 1e-6:
            d.line([P(x0, z), P(x1, z)], fill=(110, 110, 110), width=1)
            z += STOREY
        z = z0
        while z + STOREY <= z1 + 1e-6:
            x = x0 + 1.2
            while x + win_w <= x1 - 0.6 + 1e-6:
                d.rectangle([P(x, z + 0.9 + win_h), P(x + win_w, z + 0.9)],
                            fill=(120, 150, 175), outline=(60, 80, 100))
                x += pitch
            z += STOREY

    block(0, 40, 0, 6, (190, 190, 190), win_w=2.2, win_h=1.8, pitch=3.0)
    block(4, 16, 6, 6 + 12 * STOREY, (176, 176, 176))
    block(24, 36, 6, 6 + 18 * STOREY, (166, 166, 166))
    # scale bar
    d.line([P(0, -2.2), P(10, -2.2)], fill=(40, 40, 40), width=3)
    d.text(P(0, -4.4), "10 m", fill=(40, 40, 40), font=_font(16))
    _caption(img)
    return img


# --- DXF ---------------------------------------------------------------------


def write_dxf(path: Path) -> None:
    import ezdxf

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.M
    for name, color in (("WALLS", 7), ("DOORS", 3), ("WINDOWS", 5), ("TEXT", 2), ("DIMS", 1)):
        doc.layers.add(name, color=color)
    msp = doc.modelspace()
    t = 0.2  # wall thickness
    L, Wd = 14.0, 9.0

    def wall(x0, y0, x1, y1):
        msp.add_lwpolyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], close=True,
                           dxfattribs={"layer": "WALLS"})

    # outer shell (outer and inner outline) and two partitions -> 3 rooms
    wall(0, 0, L, Wd)
    wall(t, t, L - t, Wd - t)
    wall(7.0 - t / 2, t, 7.0 + t / 2, Wd - t)           # vertical partition
    wall(7.0 + t / 2, 4.5 - t / 2, L - t, 4.5 + t / 2)  # horizontal partition, right side

    def door(x, y, width, horizontal=True):
        if horizontal:
            msp.add_line((x, y), (x + width, y), dxfattribs={"layer": "DOORS"})
            msp.add_arc((x, y), width, 0, 90, dxfattribs={"layer": "DOORS"})
            msp.add_line((x, y), (x, y + width), dxfattribs={"layer": "DOORS"})
        else:
            msp.add_line((x, y), (x, y + width), dxfattribs={"layer": "DOORS"})
            msp.add_arc((x, y), width, 270, 360, dxfattribs={"layer": "DOORS"})
            msp.add_line((x, y), (x + width, y), dxfattribs={"layer": "DOORS"})

    door(2.0, 0.0, 0.9)               # front door, left room
    door(7.0 - t / 2, 6.5, 0.9, False)  # opening between left room and right-top room
    door(10.0, 4.5 - t / 2, 0.9)       # opening between the two right rooms

    for x0 in (3.0, 10.0):  # windows in the south wall
        msp.add_line((x0, 0.0), (x0 + 1.6, 0.0), dxfattribs={"layer": "WINDOWS"})
        msp.add_line((x0, t), (x0 + 1.6, t), dxfattribs={"layer": "WINDOWS"})
    for y0 in (1.5, 6.0):  # windows in the east wall
        msp.add_line((L, y0), (L, y0 + 1.4), dxfattribs={"layer": "WINDOWS"})
        msp.add_line((L - t, y0), (L - t, y0 + 1.4), dxfattribs={"layer": "WINDOWS"})

    for label, x, y in (("LIVING", 2.6, 4.2), ("BEDROOM", 8.6, 6.5), ("STUDY", 8.9, 2.2)):
        msp.add_text(label, height=0.35, dxfattribs={"layer": "TEXT", "insert": (x, y)})
    msp.add_text("SAMPLE PLAN — SYNTHETIC", height=0.5,
                 dxfattribs={"layer": "TEXT", "insert": (0.0, -2.6)})

    msp.add_linear_dim(base=(0, -1.2), p1=(0, 0), p2=(L, 0), dimstyle="EZDXF",
                       override={"dimtxt": 0.3}, dxfattribs={"layer": "DIMS"}).render()
    msp.add_linear_dim(base=(L + 1.2, 0), p1=(L, 0), p2=(L, Wd), angle=90, dimstyle="EZDXF",
                       override={"dimtxt": 0.3}, dxfattribs={"layer": "DIMS"}).render()
    doc.saveas(path)


# --- IFC ---------------------------------------------------------------------


def write_ifc(path: Path) -> bool:
    """One storey, four walls, a slab and a door opening. False if unavailable."""
    try:
        import ifcopenshell
        import ifcopenshell.api.aggregate
        import ifcopenshell.api.context
        import ifcopenshell.api.geometry
        import ifcopenshell.api.project
        import ifcopenshell.api.root
        import ifcopenshell.api.spatial
        import ifcopenshell.api.unit
        import numpy as np
    except ImportError:
        return False

    api = ifcopenshell.api
    model = api.project.create_file(version="IFC4")
    project = api.root.create_entity(model, ifc_class="IfcProject", name="Synthetic sample")
    api.unit.assign_unit(model)
    ctx = api.context.add_context(model, context_type="Model")
    body = api.context.add_context(
        model, context_type="Model", context_identifier="Body",
        target_view="MODEL_VIEW", parent=ctx,
    )
    site = api.root.create_entity(model, ifc_class="IfcSite", name="Site")
    building = api.root.create_entity(model, ifc_class="IfcBuilding", name="Sample building")
    storey = api.root.create_entity(model, ifc_class="IfcBuildingStorey", name="Ground")
    api.aggregate.assign_object(model, relating_object=project, products=[site])
    api.aggregate.assign_object(model, relating_object=site, products=[building])
    api.aggregate.assign_object(model, relating_object=building, products=[storey])

    def place(product, x, y, z=0.0, angle=0.0):
        m = np.eye(4)
        m[0][0], m[0][1] = math.cos(angle), -math.sin(angle)
        m[1][0], m[1][1] = math.sin(angle), math.cos(angle)
        m[0][3], m[1][3], m[2][3] = x, y, z
        api.geometry.edit_object_placement(model, product=product, matrix=m)

    L, Wd, h, th = 14.0, 9.0, 3.0, 0.2
    walls = []
    for name, (x, y, ang, length) in {
        "South wall": (0.0, 0.0, 0.0, L),
        "East wall": (L, 0.0, math.pi / 2, Wd),
        "North wall": (L, Wd, math.pi, L),
        "West wall": (0.0, Wd, -math.pi / 2, Wd),
    }.items():
        wall = api.root.create_entity(model, ifc_class="IfcWall", name=name)
        rep = api.geometry.add_wall_representation(
            model, context=body, length=length, height=h, thickness=th
        )
        api.geometry.assign_representation(model, product=wall, representation=rep)
        place(wall, x, y, 0.0, ang)
        api.spatial.assign_container(model, relating_structure=storey, products=[wall])
        walls.append(wall)

    slab = api.root.create_entity(model, ifc_class="IfcSlab", name="Ground slab")
    rep = api.geometry.add_slab_representation(model, context=body, depth=0.2)
    api.geometry.assign_representation(model, product=slab, representation=rep)
    place(slab, 0.0, 0.0, -0.2)
    api.spatial.assign_container(model, relating_structure=storey, products=[slab])
    model.write(str(path))
    return True


# --- main --------------------------------------------------------------------


def _save_png(img: Image.Image, path: Path) -> None:
    clean = Image.new("RGB", img.size)
    clean.paste(img)
    clean.save(path, format="PNG", optimize=True)  # no exif/pnginfo passed -> none written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("examples/inputs"))
    args = ap.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    aerial = Camera((-70.0, -95.0, 85.0), (20.0, 12.0, 20.0), perspective=False, scale=9.0,
                    centre=(W / 2, H / 2 - 20))
    street = Camera((-50.0, -105.0, 3.0), (20.0, 12.0, 20.0), perspective=True, scale=1450.0,
                    centre=(W / 2, H / 2 + 10))
    _save_png(draw_massing(aerial), out / "massing-aerial.png")
    _save_png(draw_massing(street, ground_ext=150.0), out / "massing-street.png")
    _save_png(draw_sketch(street), out / "sketch-street.png")
    _save_png(draw_elevation(), out / "elevation-sample.png")
    write_dxf(out / "plan-sample.dxf")
    wrote = ["massing-aerial.png", "massing-street.png", "sketch-street.png",
             "elevation-sample.png", "plan-sample.dxf"]
    if write_ifc(out / "model-sample.ifc"):
        wrote.append("model-sample.ifc")
    else:
        print("ifcopenshell not available - model-sample.ifc skipped", file=sys.stderr)
    for name in wrote:
        print(f"wrote {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
