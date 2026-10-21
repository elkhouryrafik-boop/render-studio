"""What a surface is made of, resolved from what the 3D model actually says.

This is the module that turns "some triangles tagged `<auto>17`, on a layer
called `A-WALL MASONRY`" into "masonry, sRGB (0.60, 0.44, 0.36), coursing every
215mm, and we are fairly sure". Nothing here rasterizes, renders, or talks to
fal.ai — it is a lookup, so it is cheap to run and cheap to test.

Why it exists at all: every ControlNet conditioning signal available to us
carries *geometry* (canny, depth) or *object class* (seg, which is hard-locked
to ADE20K's 150 classes and has no material class in it whatsoever). None of
them can say "this wall is travertine". So material identity has to arrive in
the base image's pixels instead — the renderer paints the right colour in the
right place because it knows which triangle has which material, and the image
model refines from there. That makes this lookup the load-bearing part: get it
wrong and the render is confidently wrong.

The three signals, in the order they are trusted:

  1. **A named library material.** SketchUp's stock materials use a
     `[Bracketed Name]` convention and carry a real texture with a real-world
     tile size. `[Polished Concrete Old]` is unambiguous and self-describing.
     This is the only tier where the file tells us the answer directly.
  2. **The layer/tag name.** `A-WALL MASONRY`, `A-GLAZ`, `ASPHALT`. Offices
     following the AIA CAD Layer Guidelines produce these by convention, and
     they survive the DWG → SketchUp import that destroys material names.
     Note SketchUp tags carry no material data of their own — `Sketchup::Layer`
     has no `material` method — so this is inference from a naming convention,
     not extraction. It is inference that works.
  3. **Nothing usable.** `<auto>17` is a DWG import artifact: AutoCAD's ACI
     palette index turned into a SketchUp material, so its colour is an index
     into a drafting palette, not an appearance. `Heather_Jeans` is a stock
     cutout person. Both mean "we do not know" — and saying so is the point.

Tier drives `confidence`, and confidence drives how hard the image model is
allowed to overwrite that region. A `[Polished Concrete Old]` surface should
come back as polished concrete; an `<auto>17` surface should be reinvented.
That is what `change_strength` encodes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

# --- taxonomy ----------------------------------------------------------------


class MaterialClass(StrEnum):
    """The classes a surface can resolve to.

    The first twelve are Lumion's own material-class taxonomy, which every
    major visualization tool converged on independently and which architects
    already think in. The rest are the architectural cases Lumion folds into
    "Masonry" but which look nothing alike in a render — asphalt is not brick,
    and a render that confuses them reads as wrong immediately.
    """

    GLASS = "glass"
    WATER = "water"
    CONCRETE = "concrete"
    METAL = "metal"
    WOOD = "wood"
    MASONRY = "masonry"
    BRICK = "brick"
    CERAMIC = "ceramic"
    FABRIC = "fabric"
    PLASTIC = "plastic"
    STONE = "stone"
    GRASS = "grass"

    ASPHALT = "asphalt"
    PLASTER = "plaster"
    ROOFING = "roofing"
    GRAVEL = "gravel"
    SOIL = "soil"
    VEGETATION = "vegetation"
    PEOPLE = "people"
    VEHICLE = "vehicle"
    FURNITURE = "furniture"

    # Not a material at all: 2D drafting furniture that rode into the 3D model
    # on a DWG import. Text, hatches, grids, section markers, revision clouds,
    # level datums, xref boundaries. It has geometry, so it rasterizes, but it
    # is not part of the building and must never reach the render.
    ANNOTATION = "annotation"

    UNKNOWN = "unknown"


class Confidence(StrEnum):
    """How much the model is allowed to overwrite what we painted.

    Named rather than numeric because the tier is a fact about the *source*
    ("the file told us" / "we read a material name" / "we only know what kind
    of element it is" / "we guessed"), while the number it maps to is a tuning
    parameter that will move once the strength sweep runs.
    """

    STATED = "stated"  # tier 1: a named library material with a texture
    INFERRED = "inferred"  # tier 2: a name that states a material
    ELEMENT = "element"  # tier 3: a name that states only a building element
    UNKNOWN = "unknown"  # tier 4: import artifact, entourage, or silence


# Per-pixel strength for `fal-ai/flux-general/differential-diffusion`, whose
# change map is a genuine per-pixel multiplier (arXiv 2306.00950), not a binary
# mask. A surface we are sure about gets held close to what we painted; a
# surface we guessed at is handed to the model.
#
# These numbers are "how freely may the model repaint this", 1.0 = repaint
# freely. That is NOT how fal encodes the map — on the wire BLACK repaints and
# WHITE preserves, so the value is inverted by `change_map_pixel` below. This
# comment used to claim the wire convention was "black preserves, bright
# rewrites"; it is the other way round, which made the task return its own
# input for three sessions and made every scale and strength knob look inert.
#
# The values themselves were measured 2026-08-31 against a ~27 MB
# SketchUp test model, once the polarity fix made the map do anything at all.
# The previous ones (0.30 / 0.50 / 0.65 / 0.80) held every stated surface so
# tightly that the render came back as the flat unlit albedo: 41.9% of that
# model is one STATED ground material, so a 0.30 tier froze
# most of the frame.
#
# The band is high and narrow on purpose, and it is not a fudge. What the tier
# protects is material *identity*, and identity is carried by the colour
# differential diffusion starts from — not by suppressing the denoise. A flat
# unlit sand plane has to change a great deal to become photographed sand while
# staying sand. Relaxing every tier toward repaint by k (c' = c + k(1 - c)) and
# sweeping k at 640x480, one fixed seed:
#
#   k=0.00  materials separate, but no cast shadows, flat water, flat ground
#   k=0.30  shadows appear, water gains banks
#   k=0.55  cast shadows, water with banks, ground markings, planting
#
# So k=0.55, which is what these four numbers are.
#
# The depth scale that pairs with it is in routing.yaml and is 0.35, NOT the
# 0.60 this sweep was run at: the sweep was at 640x480, where 0.35 drifted
# and 0.60 held — and that ordering reverses at
# the app's real 1280x960, where 0.60 drifts and 0.35 holds. Read that as a warning about this comment too: four
# samples at 640x480 plus a two-sample full-resolution confirm is better than
# the two points these numbers used to rest on, and is still not converged.
# Anything tuned here must be re-checked at 1280x960 before it is believed.
_CHANGE_STRENGTH: dict[Confidence, float] = {
    Confidence.STATED: 0.68,
    Confidence.INFERRED: 0.78,
    Confidence.ELEMENT: 0.84,
    Confidence.UNKNOWN: 0.91,
}


@dataclass(frozen=True)
class MaterialSignal:
    """One resolved surface appearance, ready to paint into the albedo pass.

    `albedo` is **sRGB** in 0..1 — the number an architect sees in SketchUp's
    colour picker, and the form both the baked scene materials and the class
    defaults below arrive in.

    It is NOT glTF's `baseColorFactor`, which this docstring used to claim it
    matched. `baseColorFactor` is linear by specification, and the two are only
    the same number for 0.0 and 1.0. Handing an sRGB value straight to a
    renderer's `baseColorFactor` gamma-encodes it a second time on the way out,
    which lifts every surface toward white: measured 2026-08-31 on a
    large test model, every mid-tone colour came out lighter — each
    one exactly its own value raised to 1/2.2. That washed-out albedo was the
    base image `skp_material_render` was refining from, and no conditioning
    scale or diffusion strength could recover it. Convert with `srgb_to_linear`
    at the point of handing this to a renderer; see `cad_convert.py`.

    `tile_width_mm` is the real-world width of one texture repeat. It is the
    single most valuable number here and the one no amount of prompting can
    recover: an image model given "brick" will invent a coursing frequency, and
    getting it wrong is a complaint architects raise about every AI render tool
    on the market. `None` means we have no scale claim, not "scale doesn't
    matter".
    """

    material_class: MaterialClass
    albedo: tuple[float, float, float]
    roughness: float
    metallic: float
    confidence: Confidence
    source: str
    tile_width_mm: float | None = None
    texture_name: str | None = None

    @property
    def change_strength(self) -> float:
        """How freely the image model may repaint this surface, 0..1."""
        return _CHANGE_STRENGTH[self.confidence]


# --- class defaults ----------------------------------------------------------

# Appearance per class, for when we know *what* something is but not what it
# looks like — which is every IFC file ever exported from Revit. Autodesk's
# exporter has no texture-writing code path at all (verified against
# `Autodesk/revit-ifc` at master: no `CreateImageTexture`, no
# `CreateSurfaceStyleWithTextures`), so an IFC gives us a material *name* and
# four numbers. This table is what turns that name into something paintable.
#
# Roughness/metallic follow the glTF metallic-roughness convention. Tile widths
# are real construction dimensions, not guesses: brick is one stretcher plus a
# 10mm perpend, CMU is a standard block, and so on.
_CLASS_DEFAULTS: dict[MaterialClass, tuple[tuple[float, float, float], float, float, float | None]] = {
    #                        albedo (sRGB 0..1)      rough  metal  tile mm
    MaterialClass.GLASS:      ((0.62, 0.71, 0.73), 0.08, 0.00, None),
    MaterialClass.WATER:      ((0.18, 0.32, 0.38), 0.05, 0.00, None),
    MaterialClass.CONCRETE:   ((0.62, 0.61, 0.58), 0.75, 0.00, 1200.0),
    MaterialClass.METAL:      ((0.66, 0.67, 0.69), 0.35, 0.90, None),
    MaterialClass.WOOD:       ((0.53, 0.37, 0.22), 0.55, 0.00, 140.0),
    MaterialClass.MASONRY:    ((0.60, 0.56, 0.50), 0.80, 0.00, 400.0),
    MaterialClass.BRICK:      ((0.53, 0.31, 0.24), 0.85, 0.00, 225.0),
    MaterialClass.CERAMIC:    ((0.80, 0.78, 0.75), 0.25, 0.00, 300.0),
    MaterialClass.FABRIC:     ((0.55, 0.52, 0.48), 0.90, 0.00, None),
    MaterialClass.PLASTIC:    ((0.70, 0.70, 0.70), 0.40, 0.00, None),
    MaterialClass.STONE:      ((0.69, 0.66, 0.60), 0.60, 0.00, 600.0),
    MaterialClass.GRASS:      ((0.29, 0.42, 0.18), 0.95, 0.00, 500.0),
    MaterialClass.ASPHALT:    ((0.22, 0.22, 0.23), 0.92, 0.00, 2000.0),
    MaterialClass.PLASTER:    ((0.84, 0.82, 0.78), 0.85, 0.00, None),
    MaterialClass.ROOFING:    ((0.32, 0.30, 0.29), 0.70, 0.00, 330.0),
    MaterialClass.GRAVEL:     ((0.50, 0.47, 0.43), 0.95, 0.00, 300.0),
    MaterialClass.SOIL:       ((0.36, 0.28, 0.20), 0.95, 0.00, 800.0),
    MaterialClass.VEGETATION: ((0.24, 0.38, 0.16), 0.90, 0.00, 700.0),
    MaterialClass.PEOPLE:     ((0.55, 0.45, 0.40), 0.70, 0.00, None),
    MaterialClass.VEHICLE:    ((0.45, 0.46, 0.50), 0.30, 0.10, None),
    MaterialClass.FURNITURE:  ((0.52, 0.48, 0.44), 0.60, 0.00, None),
    MaterialClass.ANNOTATION: ((1.00, 1.00, 1.00), 1.00, 0.00, None),
    MaterialClass.UNKNOWN:    ((0.58, 0.57, 0.55), 0.60, 0.00, None),
}


# --- name matching -----------------------------------------------------------


# Keyword → class. Matched against a normalized name, longest keyword first, so
# "polished concrete" beats "concrete" and "curtain wall" does not get eaten by
# "wall". Entries cover the AIA CAD Layer Guidelines discipline prefixes 
# DWG-derived layers actually use (A- architectural, L- landscape,
# C- civil, G- general), the SketchUp stock library's bracketed names, and the
# abbreviations that survive a 31-character DWG layer-name limit ("CONCRET",
# "BUILDNG" — truncated, not typos).
_KEYWORDS: dict[str, MaterialClass] = {
    # glass / glazing
    "glaz": MaterialClass.GLASS,
    "glass": MaterialClass.GLASS,
    "curtain wall": MaterialClass.GLASS,
    "curtainwall": MaterialClass.GLASS,
    "window": MaterialClass.GLASS,
    "mullion": MaterialClass.METAL,
    "skylight": MaterialClass.GLASS,
    "زجاج": MaterialClass.GLASS,
    # water
    "water": MaterialClass.WATER,
    "pool": MaterialClass.WATER,
    "gully": MaterialClass.WATER,
    "fountain": MaterialClass.WATER,
    "مياه": MaterialClass.WATER,
    # concrete
    "concrete": MaterialClass.CONCRETE,
    "concret": MaterialClass.CONCRETE,
    "conc": MaterialClass.CONCRETE,
    "screed": MaterialClass.CONCRETE,
    "precast": MaterialClass.CONCRETE,
    "gfrc": MaterialClass.CONCRETE,
    "خرسانة": MaterialClass.CONCRETE,
    # metal
    "metal": MaterialClass.METAL,
    "steel": MaterialClass.METAL,
    "alum": MaterialClass.METAL,
    "corten": MaterialClass.METAL,
    "zinc": MaterialClass.METAL,
    "copper": MaterialClass.METAL,
    "brass": MaterialClass.METAL,
    "handrail": MaterialClass.METAL,
    "balustrade": MaterialClass.METAL,
    "railing": MaterialClass.METAL,
    "louvre": MaterialClass.METAL,
    "louver": MaterialClass.METAL,
    "fence": MaterialClass.METAL,
    "light pole": MaterialClass.METAL,
    "lightpole": MaterialClass.METAL,
    # timber
    "wood": MaterialClass.WOOD,
    "timber": MaterialClass.WOOD,
    "oak": MaterialClass.WOOD,
    "teak": MaterialClass.WOOD,
    "pine": MaterialClass.WOOD,
    "cedar": MaterialClass.WOOD,
    "decking": MaterialClass.WOOD,
    "pergola": MaterialClass.WOOD,
    "خشب": MaterialClass.WOOD,
    # masonry / brick
    "masonry": MaterialClass.MASONRY,
    "blockwork": MaterialClass.MASONRY,
    "cmu": MaterialClass.MASONRY,
    "brick": MaterialClass.BRICK,
    "طوب": MaterialClass.BRICK,
    # stone
    "stone": MaterialClass.STONE,
    "granite": MaterialClass.STONE,
    "marble": MaterialClass.STONE,
    "travertine": MaterialClass.STONE,
    "limestone": MaterialClass.STONE,
    "basalt": MaterialClass.STONE,
    "kerb": MaterialClass.STONE,
    "curb": MaterialClass.STONE,
    "حجر": MaterialClass.STONE,
    # ceramic / tile
    "ceramic": MaterialClass.CERAMIC,
    "porcelain": MaterialClass.CERAMIC,
    "tile": MaterialClass.CERAMIC,
    "sanitary": MaterialClass.CERAMIC,
    # plaster / render / paint
    "plaster": MaterialClass.PLASTER,
    "render": MaterialClass.PLASTER,
    "stucco": MaterialClass.PLASTER,
    "gypsum": MaterialClass.PLASTER,
    "drywall": MaterialClass.PLASTER,
    "paint": MaterialClass.PLASTER,
    # roofing
    "roof": MaterialClass.ROOFING,
    "shingle": MaterialClass.ROOFING,
    "membrane": MaterialClass.ROOFING,
    "سقف": MaterialClass.ROOFING,
    # ground
    "asphalt": MaterialClass.ASPHALT,
    "blacktop": MaterialClass.ASPHALT,
    "tarmac": MaterialClass.ASPHALT,
    "road": MaterialClass.ASPHALT,
    "carriageway": MaterialClass.ASPHALT,
    "parking": MaterialClass.ASPHALT,
    "carpark": MaterialClass.ASPHALT,
    "gravel": MaterialClass.GRAVEL,
    "aggregate": MaterialClass.GRAVEL,
    "earthwork": MaterialClass.SOIL,
    "terrain": MaterialClass.SOIL,
    "soil": MaterialClass.SOIL,
    "sand": MaterialClass.SOIL,
    "contour": MaterialClass.SOIL,
    "grass": MaterialClass.GRASS,
    "lawn": MaterialClass.GRASS,
    "turf": MaterialClass.GRASS,
    # planting
    "plant": MaterialClass.VEGETATION,
    "tree": MaterialClass.VEGETATION,
    "shrub": MaterialClass.VEGETATION,
    "hedge": MaterialClass.VEGETATION,
    "palm": MaterialClass.VEGETATION,
    "foliage": MaterialClass.VEGETATION,
    "in leaf": MaterialClass.VEGETATION,
    "نخيل": MaterialClass.VEGETATION,
    # entourage — recognised so it can be *excluded* from material treatment,
    # not so it can be painted. A render's people should come from the image
    # model, not from a flat-shaded SketchUp cutout baked into the base image.
    "people": MaterialClass.PEOPLE,
    "person": MaterialClass.PEOPLE,
    "figure": MaterialClass.PEOPLE,
    "car": MaterialClass.VEHICLE,
    "vehicle": MaterialClass.VEHICLE,
    "truck": MaterialClass.VEHICLE,
    # interior
    "furn": MaterialClass.FURNITURE,
    "meuble": MaterialClass.FURNITURE,
    "fabric": MaterialClass.FABRIC,
    "upholstery": MaterialClass.FABRIC,
    "carpet": MaterialClass.FABRIC,
    "curtain": MaterialClass.FABRIC,
    "plastic": MaterialClass.PLASTIC,
    "vinyl": MaterialClass.PLASTIC,
    "pvc": MaterialClass.PLASTIC,
}

# Longest first, so a specific keyword always wins over a substring of itself.
_KEYWORDS_BY_LENGTH: list[tuple[str, MaterialClass]] = sorted(
    _KEYWORDS.items(), key=lambda kv: -len(kv[0])
)


# Drafting furniture, not building fabric. A SketchUp model imported from DWG
# carries the whole 2D sheet with it — section markers, revision clouds, level
# datums, area text, hatch fills, xref boundaries, `Defpoints`. It has real
# geometry, so it rasterizes into the view like anything else, and then the
# image model faithfully renders a floating "A-REV CLOUD 260611" into an
# architectural photograph.
#
# Recognising these is worth more than it looks: it is the difference between
# an office being able to drop a raw imported model into the app and having to
# clean it first. Checked before material keywords, because `A-TEXT 200` must
# never reach a matcher that would find something in it.
_ANNOTATION_KEYWORDS: tuple[str, ...] = (
    "annotation", "anno", "text", "hatch", "grid", "symbol", "dim",
    "xref", "defpoints", "revision", "rev cloud", "marker", "legend",
    "keynote", "callout", "tag", "label", "titleblock", "title block",
    "viewport", "detl", "detail ref", "north arrow", "scale bar",
    "match line", "matchline", "section line", "proj", "projection",
    "level ", "datum", "centerline", "centre line", "guide",
)

# Building elements, named without their material. `A-WALL EXTERNAL` says
# where a surface is, not what it is made of; `g-stair` says it is a stair.
# These are real, renderable surfaces — dropping them would leave holes in the
# base image — but the material is our guess, so they resolve at a weaker
# confidence than a name that actually states a material.
#
# The mapping is the boring-but-correct default for each element in the kind of
# work this office does, not an attempt to be clever.
_ELEMENT_KEYWORDS: dict[str, MaterialClass] = {
    "wall": MaterialClass.PLASTER,
    "partition": MaterialClass.PLASTER,
    "ceiling": MaterialClass.PLASTER,
    "soffit": MaterialClass.PLASTER,
    "floor": MaterialClass.CONCRETE,
    "slab": MaterialClass.CONCRETE,
    "stair": MaterialClass.CONCRETE,
    "step": MaterialClass.CONCRETE,
    "ramp": MaterialClass.CONCRETE,
    "column": MaterialClass.CONCRETE,
    "beam": MaterialClass.CONCRETE,
    "structure": MaterialClass.CONCRETE,
    "core": MaterialClass.CONCRETE,
    "foundation": MaterialClass.CONCRETE,
    "culvert": MaterialClass.CONCRETE,
    "manhole": MaterialClass.CONCRETE,
    "kerbstone": MaterialClass.STONE,
    "building": MaterialClass.PLASTER,
    "buildng": MaterialClass.PLASTER,  # 31-char DWG truncation, not a typo
    "bldg": MaterialClass.PLASTER,
    "shed": MaterialClass.METAL,
    "door": MaterialClass.WOOD,
    "overhead": MaterialClass.CONCRETE,
    "canopy": MaterialClass.METAL,
    "lane": MaterialClass.ASPHALT,
    "sidewalk": MaterialClass.CONCRETE,
    "pavement": MaterialClass.CONCRETE,
    "site": MaterialClass.SOIL,
    "storm": MaterialClass.CONCRETE,
    "sewer": MaterialClass.CONCRETE,
    # Site utilities: small metal objects that read as street furniture.
    "hydrant": MaterialClass.METAL,
    "valve": MaterialClass.METAL,
    "camera": MaterialClass.METAL,
    "cable": MaterialClass.METAL,
    "electric": MaterialClass.METAL,
    "barrier": MaterialClass.METAL,
}

_ELEMENTS_BY_LENGTH: list[tuple[str, MaterialClass]] = sorted(
    _ELEMENT_KEYWORDS.items(), key=lambda kv: -len(kv[0])
)


# Names that carry no material claim, however confidently they are spelled.
#
# `<auto>N` is the big one: importing a DWG into SketchUp mints one material
# per source layer colour, named `<auto>`, `<auto>1`, `<auto>2`… Their RGB is
# an AutoCAD ACI drafting-palette index — pure (255,0,0), (255,255,0),
# (0,255,255) — chosen so lines are distinguishable on a black screen. Painting
# a building with them would produce a render in primary colours.
#
# `Heather_*` / `Lily_*` are SketchUp's stock cutout people, whose per-garment
# materials (`Heather_Jeans`, `Lily_Blonde`) otherwise look like real named
# materials to a keyword matcher.
_MEANINGLESS_MATERIAL = re.compile(
    r"""^(
        <auto>\d*            # DWG-import artifact
      | material\d*          # generic placeholder
      | default              # SketchUp's unpainted state
      | layer\d*(_\d+)*      # `Layer0_255_101_50` — a layer, not a material
      | color[_ ]?[a-f0-9]{6}
      | \*+                  # `*` — a wildcard that rode in on an import
    )$""",
    re.IGNORECASE | re.VERBOSE,
)

_STOCK_ENTOURAGE = re.compile(r"^(heather|lily|chris|susan|dave|nancy)[_ ]", re.IGNORECASE)

# SketchUp's stock COLOUR swatches: the "Colors" library (`[0033_NavajoWhite]`,
# `[0135_DarkSalmon]`) and the lettered grid (`[Color A03]`, `[Color_F12]`).
# A swatch is a colour choice, not a material claim — `[Color_F12]` painted on
# tens of thousands of plant-leaf cards in a large model means "make the leaves this
# green", not "the leaves are made of Color_F12". The bracket convention makes
# them look like tier-1 library materials to `is_library_material`, so they are
# refused by name here, before the tier logic can promote them. Their RGB is
# still a legitimate appearance hint (unlike ACI `<auto>N` colours) — callers
# keep passing it as `base_color`, and it is honoured whenever a real tier
# resolves the surface some other way.
_COLOR_SWATCH = re.compile(
    r"""^\[?\s*(
        colou?r[_ ]?[a-z]{0,2}\d{1,3}   # Color A03, Color_F12, Color B04
      | \d{3,4}[_ ][a-z]+               # 0033_NavajoWhite, 0135_DarkSalmon
    )\s*\]?\d*$""",
    re.IGNORECASE | re.VERBOSE,
)


def is_color_swatch(name: str | None) -> bool:
    """Whether a name is a stock colour swatch — a colour claim, not a material."""
    return bool(name) and bool(_COLOR_SWATCH.match(name.strip()))


def is_placeholder_material(name: str | None) -> bool:
    """Whether a material name carries no signal at all.

    True for DWG-import artifacts (`<auto>17`), generic placeholders
    (`Material3`, `Default`, `*`), layer-shaped names (`Layer0_255_101_50`),
    and stock cutout-people garments (`Heather_Jeans`). These names can never
    *produce* a material signal, so a caller joining materials by colour should
    not let them *block* one either: a colour shared by `Heather_Soles` and
    `[Blacktop New]` is a blacktop signal, not an ambiguity.
    """
    if not name:
        return False
    stripped = name.strip()
    return bool(_MEANINGLESS_MATERIAL.match(stripped)) or bool(_STOCK_ENTOURAGE.match(stripped))

# SketchUp's stock library wraps its names in brackets: `[Polished Concrete Old]`.
# Anything bracketed was chosen from a library rather than auto-generated, which
# is the whole reason tier 1 exists.
_BRACKETED = re.compile(r"^\[(?P<name>.+?)\]\d*$")


def _normalize(name: str) -> str:
    """Fold a name to something keyword-matchable.

    Separators become spaces so `A-WALL_MASONRY` and `A WALL MASONRY` match the
    same way. Unicode is NFKC-normalized rather than stripped — these files
    carry non-ASCII (e.g. Arabic) layer names and an ASCII-only
    matcher would silently drop them into the unknown tier.
    """
    folded = unicodedata.normalize("NFKC", name).casefold()
    return re.sub(r"[\s_\-./\\]+", " ", folded).strip()


def is_annotation(name: str | None) -> bool:
    """Whether a name marks 2D drafting furniture rather than building fabric."""
    if not name:
        return False
    haystack = _normalize(name)
    return any(_normalize(k) in haystack for k in _ANNOTATION_KEYWORDS)


def classify_name(name: str | None) -> MaterialClass:
    """Best material class for a single name, or UNKNOWN.

    Deliberately total: callers pass layer names, material names and glTF
    material names through the same door, and none of them are guaranteed to
    mean anything.

    Annotation is checked first and short-circuits — `A-TEXT 200` contains no
    material, and a matcher allowed to keep looking would eventually find one.
    """
    if not name:
        return MaterialClass.UNKNOWN
    if is_placeholder_material(name) or is_color_swatch(name):
        return MaterialClass.UNKNOWN
    haystack = _normalize(name)
    if not haystack:
        return MaterialClass.UNKNOWN
    # Materials before annotation, because the annotation vocabulary is broad
    # enough to swallow real surfaces: `A-GRID PARKING` is a parking deck, not
    # a column grid, and `grid` alone would have taken it.
    for keyword, material_class in _KEYWORDS_BY_LENGTH:
        if _normalize(keyword) in haystack:
            return material_class
    if is_annotation(name):
        return MaterialClass.ANNOTATION
    return MaterialClass.UNKNOWN


def classify_element(name: str | None) -> MaterialClass:
    """Material implied by a building-element name, or UNKNOWN.

    Only consulted after `classify_name` has failed, so `A-WALL MASONRY`
    resolves as masonry rather than as the generic wall default.
    """
    if not name or is_annotation(name):
        return MaterialClass.UNKNOWN
    if is_placeholder_material(name) or is_color_swatch(name):
        return MaterialClass.UNKNOWN
    haystack = _normalize(name)
    for keyword, material_class in _ELEMENTS_BY_LENGTH:
        if _normalize(keyword) in haystack:
            return material_class
    return MaterialClass.UNKNOWN


def is_library_material(name: str | None) -> bool:
    """Whether a material name was chosen from a library rather than generated."""
    return bool(name) and bool(_BRACKETED.match(name.strip()))


def library_material_name(name: str) -> str:
    """`[Polished Concrete Old]1` → `Polished Concrete Old`.

    SketchUp appends a disambiguating digit when a second copy of a library
    material enters the model — both files here contain `[Blacktop Old 01]` and
    `[Blacktop Old 01]1`, which are the same product at different texture
    scales. Stripping the suffix is what makes them resolve to one material.
    """
    match = _BRACKETED.match(name.strip())
    return match.group("name").strip() if match else name.strip()


# --- resolution --------------------------------------------------------------


def resolve(
    *,
    material_name: str | None = None,
    layer_name: str | None = None,
    base_color: tuple[float, float, float] | None = None,
    roughness: float | None = None,
    metallic: float | None = None,
    texture_name: str | None = None,
    texture_width_mm: float | None = None,
) -> MaterialSignal:
    """Resolve one surface to a paintable appearance and a confidence.

    Everything is optional because every source is partial: an IFC brings a
    material name and no texture, a DWG-derived SketchUp model brings a layer
    name and a meaningless material name, and a well-authored SketchUp model
    brings all of it.

    The tiers are tried in order, and the *first* one that produces a real
    class wins — a stated material is never overruled by a layer-name guess,
    because the layer is about where a surface lives and the material is about
    what it is made of. Those disagree constantly and legitimately: glass
    balustrades live on `A-STAIRS`.
    """
    # Tier 1 — the file states it. A bracketed library name, or any material
    # name that both means something and carries a texture.
    if material_name:
        stated_class = classify_name(
            library_material_name(material_name)
            if is_library_material(material_name)
            else material_name
        )
        if stated_class not in (MaterialClass.UNKNOWN, MaterialClass.ANNOTATION) and (
            is_library_material(material_name) or texture_name
        ):
            return _build(
                stated_class,
                Confidence.STATED,
                source=f"material:{material_name}",
                base_color=base_color,
                roughness=roughness,
                metallic=metallic,
                texture_name=texture_name,
                texture_width_mm=texture_width_mm,
            )

    # Tier 2 — a naming convention implies it. A meaningful material name with
    # no texture lands here too: `Translucent Glass Gray` is a real claim, but
    # without a texture it is a weaker one than `[Polished Concrete Old]`.
    for candidate, kind in ((material_name, "material"), (layer_name, "layer")):
        if not candidate:
            continue
        inferred = classify_name(candidate)
        if inferred not in (MaterialClass.UNKNOWN, MaterialClass.ANNOTATION):
            return _build(
                inferred,
                Confidence.INFERRED,
                source=f"{kind}:{candidate}",
                base_color=base_color,
                roughness=roughness,
                metallic=metallic,
                texture_name=texture_name,
                texture_width_mm=texture_width_mm,
            )

    # Annotation, checked before the element tier so `A-HATCH HORIZONTAL
    # ACCESS` is not read as an access ramp. Painted pure white and given the
    # ANNOTATION class so the renderer can drop it outright.
    for candidate, kind in ((material_name, "material"), (layer_name, "layer")):
        if candidate and classify_name(candidate) is MaterialClass.ANNOTATION:
            return _build(
                MaterialClass.ANNOTATION,
                Confidence.STATED,
                source=f"{kind}:{candidate}",
                base_color=None,
                roughness=None,
                metallic=None,
                texture_name=None,
                texture_width_mm=None,
            )

    # Tier 3 — the name states an element, not a material. `A-WALL EXTERNAL`
    # is a real surface that must be painted something; plaster is the honest
    # default and the higher change strength says so.
    for candidate, kind in ((material_name, "material"), (layer_name, "layer")):
        if not candidate:
            continue
        element = classify_element(candidate)
        if element is not MaterialClass.UNKNOWN:
            return _build(
                element,
                Confidence.ELEMENT,
                source=f"{kind}-element:{candidate}",
                base_color=base_color,
                roughness=roughness,
                metallic=metallic,
                texture_name=texture_name,
                texture_width_mm=texture_width_mm,
            )

    # Tier 4 — nothing usable. Note we do NOT fall back to `base_color` here:
    # for the `<auto>N` case that colour is an ACI drafting index, and painting
    # it would be worse than admitting ignorance, because a confident wrong
    # colour is what the image model would then faithfully preserve.
    return _build(
        MaterialClass.UNKNOWN,
        Confidence.UNKNOWN,
        source=f"unresolved:{material_name or layer_name or '?'}",
        base_color=None,
        roughness=roughness,
        metallic=metallic,
        texture_name=None,
        texture_width_mm=None,
    )


def _build(
    material_class: MaterialClass,
    confidence: Confidence,
    *,
    source: str,
    base_color: tuple[float, float, float] | None,
    roughness: float | None,
    metallic: float | None,
    texture_name: str | None,
    texture_width_mm: float | None,
) -> MaterialSignal:
    """Fill the gaps in a resolved class from the class defaults.

    The model's own values win where it has them — a SketchUp material that
    says its concrete is (0.75, 0.74, 0.72) knows better than our generic
    concrete. The defaults exist for IFC, which has no appearance data at all.
    """
    default_albedo, default_rough, default_metal, default_tile = _CLASS_DEFAULTS[material_class]
    return MaterialSignal(
        material_class=material_class,
        albedo=base_color if base_color is not None else default_albedo,
        roughness=roughness if roughness is not None else default_rough,
        metallic=metallic if metallic is not None else default_metal,
        confidence=confidence,
        source=source,
        tile_width_mm=texture_width_mm if texture_width_mm is not None else default_tile,
        texture_name=texture_name,
    )


def change_map_pixel(change_strength: float) -> int:
    """`change_strength` (1.0 = repaint freely) -> a change-map pixel, 0..255.

    fal's `change_map_image_url` is BLACK = repaint, WHITE = preserve, so this
    inverts. It exists as a named function, rather than as arithmetic inline in
    the renderer, because getting it backwards is invisible: the render still
    succeeds, still costs money, and returns a plausible image that happens to
    be its own input. It shipped that way from 2026-08-07 to 2026-08-31 and
    took three sessions and two rounds of scale-tuning to catch.
    """
    return round((1.0 - change_strength) * 255)


def srgb_to_linear(c: tuple[float, float, float]) -> tuple[float, float, float]:
    """sRGB 0..1 -> linear 0..1, the IEC 61966-2-1 transfer function.

    `MaterialSignal.albedo` is sRGB. Every renderer parameter named
    `baseColorFactor` is linear, per the glTF specification. This is the
    conversion between them, and skipping it is what painted the sand
    site as near-white (see the note on `MaterialSignal`).

    Not a 2.2 power: the real curve is linear below 0.04045, and using the pure
    power law instead crushes dark materials — asphalt at 0.12 differs by a
    tenth of a stop, which is visible on a large paved area.
    """
    return tuple(  # type: ignore[return-value]
        x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
        for x in c
    )


# Surfaces the base image should NOT assert an appearance for. Flat-shaded
# SketchUp cutout people and low-poly cars baked into a base image survive
# refinement as flat-shaded cutout people and low-poly cars — "entourage
# cloning" and CG-looking figures are two of the complaints architects raise
# about AI renders. Better to leave these regions to the image model, which is
# genuinely good at them.
ENTOURAGE_CLASSES = frozenset(
    {MaterialClass.PEOPLE, MaterialClass.VEHICLE}
)


def is_entourage(signal: MaterialSignal) -> bool:
    """Whether this surface should be handed to the model rather than painted."""
    return signal.material_class in ENTOURAGE_CLASSES


def is_renderable(signal: MaterialSignal) -> bool:
    """Whether this surface belongs in the render at all.

    Drafting furniture does not. It is the one category where the right answer
    is to delete the geometry rather than paint it — a revision cloud has no
    correct appearance.
    """
    return signal.material_class is not MaterialClass.ANNOTATION
