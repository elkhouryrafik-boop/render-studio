# Sample inputs

Everything in `inputs/` is **synthetic**: an invented three-volume massing (a 40 x 24 x 6 m podium and two
towers of 12 and 18 storeys on a 3 m grid) and an invented 14 x 9 m three-room plan. No real site, client
or project. Regenerate them (deterministic for the images and the DXF) with:

```bash
uv run python examples/make_sample.py --out examples/inputs
```

| File | What it is | Try it with | Expected with `RENDER_ENGINE=fake` |
|---|---|---|---|
| `massing-aerial.png` | axonometric massing view, 1536x1024 | `massing_to_render`, `aerial_view_render`, `variation` | a labelled placeholder image, same size class, no fal call |
| `massing-street.png` | two-point-perspective massing view | `massing_to_render`, `inpaint`, `relight` | labelled placeholder |
| `sketch-street.png` | the street view as a jittered pencil sketch | `sketch_to_render` | labelled placeholder |
| `elevation-sample.png` | orthographic elevation with a window grid | `elevation_section_render` | labelled placeholder |
| `plan-sample.dxf` | plan; layers `WALLS DOORS WINDOWS TEXT DIMS` | upload as CAD, pick layers, confirm, then `floorplan_prettify` | the confirm step rasterises locally (real); the render is a labelled placeholder |
| `model-sample.ifc` | one storey ("Ground"), 4 walls, a slab | upload as CAD, confirm with the `iso` camera, then `massing_to_render` | clay view + depth rendered locally (real); the render is a labelled placeholder |

Each image carries a caption strip ("Synthetic massing - render-studio sample"); the DXF carries the text
"SAMPLE PLAN - SYNTHETIC". The IFC project is named "Synthetic sample". Image metadata is stripped.
`tests/test_examples.py` loads every file through the asset / CAD API.
