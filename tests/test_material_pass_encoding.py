"""The two encodings that made `skp_material_render` return its own input.

Both bugs were invisible in every way that matters operationally: the job
succeeded, the cost was billed, an image came back, and the image looked like
a plausible-if-washed-out render. Nothing failed, so nothing was caught, and
three sessions of scale-tuning were spent on knobs that could not do anything.
The point of these tests is that the next such inversion fails in CI instead.

Measurements referenced here are from 2026-08-31 against a ~27 MB
SketchUp test model.
"""
from __future__ import annotations

import pytest

from app.services.materials import (
    _CHANGE_STRENGTH,
    Confidence,
    change_map_pixel,
    srgb_to_linear,
)

# --- change map polarity -----------------------------------------------------

def test_free_repaint_is_black_and_full_preservation_is_white():
    """fal's convention, established by paid probe, not by reading a doc page.

    An all-white change map returned the input almost unchanged (MAE 2.45 of
    255); an all-black one was the only configuration that rendered at all
    (MAE 14.77). The app had this backwards, which is why it asked fal to
    preserve 73% of every frame and got the albedo handed straight back.
    """
    assert change_map_pixel(1.0) == 0      # repaint freely -> black
    assert change_map_pixel(0.0) == 255    # hold exactly   -> white


def test_more_freedom_is_always_darker():
    strengths = [0.0, 0.1, 0.25, 0.5, 0.68, 0.8, 0.91, 1.0]
    pixels = [change_map_pixel(s) for s in strengths]
    assert pixels == sorted(pixels, reverse=True), pixels


@pytest.mark.parametrize("confidence", list(Confidence))
def test_every_confidence_tier_encodes_in_range(confidence: Confidence):
    px = change_map_pixel(_CHANGE_STRENGTH[confidence])
    assert 0 <= px <= 255


def test_a_surface_the_file_states_is_held_harder_than_one_we_guessed():
    """The tier ordering is the whole point of the map — if it ever inverts,
    the pipeline is confidently repainting what the architect specified and
    preserving what it invented."""
    stated = change_map_pixel(_CHANGE_STRENGTH[Confidence.STATED])
    inferred = change_map_pixel(_CHANGE_STRENGTH[Confidence.INFERRED])
    element = change_map_pixel(_CHANGE_STRENGTH[Confidence.ELEMENT])
    unknown = change_map_pixel(_CHANGE_STRENGTH[Confidence.UNKNOWN])
    assert stated > inferred > element > unknown
    # And every real surface is held harder than empty background.
    assert unknown > change_map_pixel(1.0)


# --- albedo colour space -----------------------------------------------------

def test_srgb_to_linear_endpoints_are_exact():
    assert srgb_to_linear((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
    r, g, b = srgb_to_linear((1.0, 1.0, 1.0))
    assert r == pytest.approx(1.0) and g == pytest.approx(1.0) and b == pytest.approx(1.0)


def test_srgb_to_linear_darkens_midtones():
    """The direction that matters. Passing sRGB to a linear `baseColorFactor`
    lifted the whole albedo pass toward white; converting first puts it back.
    """
    mid, = {round(v, 4) for v in srgb_to_linear((0.5, 0.5, 0.5))}
    assert 0.2 < mid < 0.25          # 0.5 sRGB is ~0.214 linear, not 0.5


def test_srgb_to_linear_uses_the_linear_segment_near_black():
    """Below 0.04045 the real curve is a line, not a power law. Asphalt and
    other dark materials live down here, and the pure 2.2 power crushes them.
    """
    v, _, _ = srgb_to_linear((0.02, 0.02, 0.02))
    assert v == pytest.approx(0.02 / 12.92)
    assert v > 0.02 ** 2.4           # the power law would go lower


def test_round_trip_reproduces_the_real_site_materials():
    """The regression, in the units it was measured in.

    A sand-coloured material is rgb(0.76, 0.72, 0.62) in the file and
    covers 41.9% of the test model. Painted without conversion it
    came out (225, 220, 206) — near-white. Converted, the renderer's own sRGB
    encode returns the original number.
    """
    def encode(linear: float) -> float:      # the renderer's output transform
        return (
            linear * 12.92 if linear <= 0.0031308
            else 1.055 * linear ** (1 / 2.4) - 0.055
        )

    for srgb in (0.76, 0.72, 0.62, 0.52, 0.25, 0.12):
        lin = srgb_to_linear((srgb, srgb, srgb))[0]
        assert encode(lin) == pytest.approx(srgb, abs=1e-6)
        # ...and the unconverted path is what shifted it, materially:
        assert encode(srgb) > srgb + 0.05
