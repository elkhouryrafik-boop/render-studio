"""Material resolution — the lookup the whole albedo pass rests on.

The names exercised here are modelled on common SketchUp/CAD naming
conventions; all names are invented. That matters
for a keyword matcher: the failure mode is not "the regex is wrong", it is "the
regex is right about names nobody writes".
"""

from __future__ import annotations

import pytest

from app.services.materials import (
    Confidence,
    MaterialClass,
    classify_name,
    is_color_swatch,
    is_entourage,
    is_library_material,
    is_placeholder_material,
    is_renderable,
    library_material_name,
    resolve,
)

# --- name classification -----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # AIA CAD Layer Guidelines, as they arrive after a DWG import.
        ("A-WALL MASONRY", MaterialClass.MASONRY),
        ("A-WALL EXTERNAL", MaterialClass.UNKNOWN),  # a location, not a material
        ("A-GLASS", MaterialClass.GLASS),
        ("A-GLAZ", MaterialClass.GLASS),
        ("A-CONCRETE", MaterialClass.CONCRETE),
        ("A-DOOR", MaterialClass.UNKNOWN),
        ("A-PLANT", MaterialClass.VEGETATION),
        ("A-PEOPLE", MaterialClass.PEOPLE),
        ("A-CARS", MaterialClass.VEHICLE),
        ("A-CAR NEW 1", MaterialClass.VEHICLE),
        ("A-GRID PARKING", MaterialClass.ASPHALT),
        ("A-Road New", MaterialClass.ASPHALT),
        # DWG's 31-character layer names, truncated by the authoring tool.
        ("CONCRET", MaterialClass.CONCRETE),
        ("ASPHALT", MaterialClass.ASPHALT),
        ("MILLING ASPHALT", MaterialClass.ASPHALT),
        ("KERBSTONE", MaterialClass.STONE),
        ("FENCE", MaterialClass.METAL),
        ("G-EARTHWORK", MaterialClass.SOIL),
        ("Terrain source", MaterialClass.SOIL),
        ("g-carpark", MaterialClass.ASPHALT),
        ("Plant in Leaf", MaterialClass.VEGETATION),
        ("g-furniture", MaterialClass.FURNITURE),
        ("POOL UP", MaterialClass.WATER),
    ],
)
def test_layer_names_classify(name: str, expected: MaterialClass) -> None:
    assert classify_name(name) is expected


def test_non_ascii_layer_names_are_not_dropped() -> None:
    """Layer tags may be non-ASCII; an ASCII-only matcher would lose them."""
    glass = "0 زجاج"  # the Arabic word for glass
    assert classify_name(glass) is MaterialClass.GLASS


def test_longer_keyword_wins_over_a_substring_of_itself() -> None:
    """`curtain wall` is glazing; `curtain` alone is fabric. Order decides."""
    assert classify_name("A-CURTAIN WALL") is MaterialClass.GLASS
    assert classify_name("Curtain") is MaterialClass.FABRIC


@pytest.mark.parametrize(
    "name",
    ["<auto>", "<auto>17", "Default", "Layer0_255_101_50", "Material3", "color_ff00ff"],
)
def test_import_artifacts_are_refused(name: str) -> None:
    """A DWG import mints one `<auto>N` material per source layer COLOUR.

    Their RGB is an AutoCAD ACI drafting index — pure red, pure magenta — so a
    matcher that took them seriously would paint buildings in primaries.
    """
    assert classify_name(name) is MaterialClass.UNKNOWN


@pytest.mark.parametrize("name", ["Heather_Jeans", "Lily_Blonde", "Heather_Skin"])
def test_stock_cutout_people_are_refused(name: str) -> None:
    """SketchUp's stock figures ship per-garment materials that read as real."""
    assert classify_name(name) is MaterialClass.UNKNOWN


def test_classification_is_total() -> None:
    assert classify_name(None) is MaterialClass.UNKNOWN
    assert classify_name("") is MaterialClass.UNKNOWN
    assert classify_name("   ") is MaterialClass.UNKNOWN


# --- library names -----------------------------------------------------------


def test_bracketed_names_are_library_materials() -> None:
    assert is_library_material("[Polished Concrete Old]")
    assert not is_library_material("<auto>4")
    assert not is_library_material("Translucent Glass Gray")


def test_duplicate_suffix_is_stripped() -> None:
    """`[Blacktop Old 01]` and `[Blacktop Old 01]1` are one product.

    Both appear in `site_model.skp` at different texture scales; without
    stripping they resolve as two unrelated materials.
    """
    assert library_material_name("[Blacktop Old 01]1") == "Blacktop Old 01"
    assert library_material_name("[Polished Concrete Old]") == "Polished Concrete Old"


# --- resolution tiers --------------------------------------------------------


def test_library_material_with_texture_is_stated() -> None:
    signal = resolve(
        material_name="[Polished Concrete Old]",
        layer_name="A-FLOOR",
        texture_name="Polished Concrete Old.jpg",
        texture_width_mm=609.6,
    )
    assert signal.material_class is MaterialClass.CONCRETE
    assert signal.confidence is Confidence.STATED
    assert signal.tile_width_mm == pytest.approx(609.6)
    assert signal.source == "material:[Polished Concrete Old]"


def test_layer_name_carries_a_model_whose_materials_are_junk() -> None:
    """The DWG-derived case: the material says nothing, the layer says asphalt."""
    signal = resolve(material_name="<auto>17", layer_name="MILLING ASPHALT")
    assert signal.material_class is MaterialClass.ASPHALT
    assert signal.confidence is Confidence.INFERRED
    assert signal.source == "layer:MILLING ASPHALT"


def test_a_stated_material_outranks_the_layer_it_sits_on() -> None:
    """Glass balustrades live on `A-STAIRS`. The material is the truth."""
    signal = resolve(
        material_name="[Translucent Glass Gray]",
        layer_name="A-STAIRS",
        texture_name="glass.png",
    )
    assert signal.material_class is MaterialClass.GLASS


def test_meaningful_material_without_texture_is_only_inferred() -> None:
    signal = resolve(material_name="Translucent Glass Gray", layer_name="A-GLAZ")
    assert signal.material_class is MaterialClass.GLASS
    assert signal.confidence is Confidence.INFERRED


def test_nothing_usable_resolves_to_unknown() -> None:
    signal = resolve(material_name="<auto>4", layer_name="Layer0")
    assert signal.material_class is MaterialClass.UNKNOWN
    assert signal.confidence is Confidence.UNKNOWN


def test_aci_import_colour_is_never_painted() -> None:
    """`<auto>3` is pure magenta because AutoCAD needed a distinguishable line.

    Carrying that into the base image would hand the model a confident wrong
    answer, which it then preserves. Admitting ignorance is strictly better.
    """
    signal = resolve(
        material_name="<auto>3",
        layer_name="Layer0",
        base_color=(1.0, 0.0, 1.0),
    )
    assert signal.confidence is Confidence.UNKNOWN
    assert signal.albedo != (1.0, 0.0, 1.0)


def test_model_supplied_appearance_beats_the_class_default() -> None:
    """A SketchUp material knows its own concrete better than our generic one."""
    signal = resolve(
        material_name="[Polished Concrete New]",
        texture_name="Polished Concrete New.jpg",
        base_color=(0.976, 0.976, 0.980),
    )
    assert signal.albedo == pytest.approx((0.976, 0.976, 0.980))


def test_ifc_style_name_only_input_still_paints() -> None:
    """Revit's IFC exporter writes no textures at all — name plus four numbers.

    Verified against `Autodesk/revit-ifc` at master: there is no
    `CreateImageTexture` and no `CreateSurfaceStyleWithTextures` anywhere in
    the exporter. So the class defaults are the entire appearance for IFC.
    """
    signal = resolve(material_name="Concrete, Cast-in-Place")
    assert signal.material_class is MaterialClass.CONCRETE
    assert signal.tile_width_mm is not None
    assert signal.albedo is not None


# --- confidence → change strength -------------------------------------------


def test_change_strength_rises_as_confidence_falls() -> None:
    """The change map is a per-pixel strength, so this ordering IS the feature.

    A surface the file described should come back as described; a surface we
    guessed at should be reinvented by the model.
    """
    stated = resolve(material_name="[Grass Light Green]", texture_name="g.jpg")
    inferred = resolve(layer_name="ASPHALT")
    unknown = resolve(material_name="<auto>9")

    assert stated.change_strength < inferred.change_strength < unknown.change_strength
    assert 0.0 <= stated.change_strength <= 1.0
    assert 0.0 <= unknown.change_strength <= 1.0


# --- entourage ---------------------------------------------------------------


def test_people_and_cars_are_left_to_the_image_model() -> None:
    """Baking flat-shaded cutouts into the base image preserves them as cutouts."""
    assert is_entourage(resolve(layer_name="A-PEOPLE"))
    assert is_entourage(resolve(layer_name="A-CARS"))
    assert not is_entourage(resolve(layer_name="A-WALL MASONRY"))


# --- annotation: 2D drafting furniture that rode in on a DWG import ----------


@pytest.mark.parametrize(
    "name",
    [
        "A-TEXT 200", "G-ANNO-TEXT", "A-AREA TEXT", "A-Hatch", "custom hatch",
        "A-HATCH HORIZONTAL ACCESS", "GRID", "SYMBOL", "A-XREF", "Defpoints",
        "A-REV CLOUD 260611", "C-MARKER_Standard", "A-PROJ", "L-PROJ",
        "projection", "A-DETL", "Level 41", "g-ramp-annotation", "g-text",
    ],
)
def test_drafting_furniture_is_recognised(name: str) -> None:
    """These layers carry real geometry and belong in no render.

    Left unhandled, the image model faithfully renders a floating revision
    cloud into an architectural photograph. Recognising them is what lets an
    office drop a raw imported model in without cleaning it first.
    """
    assert classify_name(name) is MaterialClass.ANNOTATION
    assert not is_renderable(resolve(layer_name=name))


def test_a_material_keyword_outranks_the_annotation_vocabulary() -> None:
    """`A-GRID PARKING` is a parking deck; `grid` alone would have taken it."""
    assert classify_name("A-GRID PARKING") is MaterialClass.ASPHALT
    assert is_renderable(resolve(layer_name="A-GRID PARKING"))


# --- element tier: the name says what it is, not what it is made of ----------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("A-WALL EXTERNAL", MaterialClass.PLASTER),
        ("WALL", MaterialClass.PLASTER),
        ("A-FLOOR", MaterialClass.CONCRETE),
        ("A-STAIRS", MaterialClass.CONCRETE),
        ("g-stair", MaterialClass.CONCRETE),
        ("g-ramp", MaterialClass.CONCRETE),
        ("A-Columns New01", MaterialClass.CONCRETE),
        ("g-core", MaterialClass.CONCRETE),
        ("g-structure", MaterialClass.CONCRETE),
        ("BUILDNG", MaterialClass.PLASTER),
        ("EXS_BLDG", MaterialClass.PLASTER),
        ("SHED", MaterialClass.METAL),
        ("DOOR", MaterialClass.WOOD),
        ("BOX CULVERT", MaterialClass.CONCRETE),
        ("Sewerage Manhole", MaterialClass.CONCRETE),
        ("FIRE HYDRANT", MaterialClass.METAL),
        ("Stop Valve", MaterialClass.METAL),
        ("g-lanes", MaterialClass.ASPHALT),
        ("SITE", MaterialClass.SOIL),
    ],
)
def test_element_names_resolve_at_the_weaker_tier(
    name: str, expected: MaterialClass
) -> None:
    signal = resolve(layer_name=name)
    assert signal.material_class is expected
    assert signal.confidence is Confidence.ELEMENT


def test_a_stated_material_outranks_the_element_default() -> None:
    """`A-WALL MASONRY` is masonry, not the generic wall default of plaster."""
    assert resolve(layer_name="A-WALL MASONRY").material_class is MaterialClass.MASONRY
    assert resolve(layer_name="A-WALL EXTERNAL").material_class is MaterialClass.PLASTER


def test_element_tier_sits_between_inferred_and_unknown() -> None:
    """An element guess deserves more freedom than a material claim, less than none."""
    material = resolve(layer_name="A-CONCRETE")
    element = resolve(layer_name="A-FLOOR")
    unknown = resolve(layer_name="Layer0")
    assert material.change_strength < element.change_strength < unknown.change_strength


def test_every_class_has_appearance_defaults() -> None:
    """A class with no default would paint `None` into the albedo pass."""
    for material_class in MaterialClass:
        signal = resolve(material_name=material_class.value)
        assert len(signal.albedo) == 3
        assert 0.0 <= signal.roughness <= 1.0
        assert 0.0 <= signal.metallic <= 1.0


# --- colour swatches: a colour choice, not a material claim -------------------


@pytest.mark.parametrize(
    "name",
    [
        # The lettered grid, exactly as `large_model.skp` carries them.
        "[Color_F12]", "[Color B04]", "[Color A03]", "[Color F08]",
        "[Color E03]", "[Color C06]", "[Color D08]", "[Color A01]",
        # The numbered "Colors" library.
        "[0033_NavajoWhite]", "[0135_DarkSalmon]",
        # Unbracketed and duplicate-suffixed forms.
        "Color_F12", "[Color B04]1",
    ],
)
def test_stock_colour_swatches_are_not_material_claims(name: str) -> None:
    """`[Color_F12]` on 46,588 leaf cards means "this green", not "made of F12".

    The bracket convention makes swatches look like tier-1 library materials,
    which would grant a paint colour STATED confidence — the model would then
    preserve a flat swatch green as if the file had described a material.
    """
    assert is_color_swatch(name)
    assert classify_name(name) is MaterialClass.UNKNOWN


def test_a_swatch_never_reaches_the_stated_tier() -> None:
    signal = resolve(material_name="[Color_F12]", layer_name="Plant in Leaf")
    assert signal.confidence is not Confidence.STATED
    # The layer still rescues the class, and the swatch RGB is honoured as the
    # appearance — the colour is real, only the *confidence* was inflated.
    assert signal.material_class is MaterialClass.VEGETATION
    assert signal.confidence is Confidence.INFERRED


def test_a_swatch_with_no_other_signal_is_unknown() -> None:
    signal = resolve(material_name="[Color B04]", layer_name="Layer0")
    assert signal.material_class is MaterialClass.UNKNOWN
    assert signal.confidence is Confidence.UNKNOWN


@pytest.mark.parametrize(
    "name",
    ["[Colorado Rustic Brick]", "[Color Wheel Mosaic]", "Colour Bond Steel 5"],
)
def test_real_materials_that_merely_start_with_color_survive(name: str) -> None:
    """The swatch pattern must not swallow real products named around colour."""
    assert not is_color_swatch(name)


# --- placeholder names: can neither produce nor block a signal ----------------


@pytest.mark.parametrize(
    "name",
    ["<auto>", "<auto>41", "Material3", "Default", "*", "Heather_Soles", "Lily_Blonde"],
)
def test_placeholder_names_are_recognised(name: str) -> None:
    """A colour→name join must not let `Heather_Soles` veto `[Blacktop New]`.

    openskp keys baked materials by RGB, so every material sharing a colour is
    a join candidate. Names that could never produce a signal should not count
    toward ambiguity — the one real name left is the signal.
    """
    assert is_placeholder_material(name)
    assert classify_name(name) is MaterialClass.UNKNOWN


@pytest.mark.parametrize("name", ["[Blacktop New]", "Translucent Glass Gray", "ASPHALT"])
def test_real_names_are_not_placeholders(name: str) -> None:
    assert not is_placeholder_material(name)
