# Task status

Status of the 14 tasks in `routing.yaml`: 10 on, 2 on but API-only, 2 experimental. Prices in `routing.yaml` carry a "priced on" date and drift; check fal.ai for current rates.

| Task | Status | Notes |
|---|---|---|
| `massing_to_render` | on | Re-verified on final code. |
| `variation` | on | Re-verified on final code. |
| `inpaint` | on | Re-verified on final code. |
| `upscale` | on | Re-verified on final code. |
| `masterplan_prettify` | on | Launch-ready; not re-checked on final code. |
| `floorplan_prettify` | on | Launch-ready; not re-checked on final code. |
| `interior_to_render` | on | Launch-ready; not re-checked on final code. |
| `aerial_view_render` | on, with caveat | Proportion drift is possible; check the massing. |
| `sketch_to_render` | on, with caveat | The proven engine is weaker on sketches. |
| `elevation_section_render` | on, with caveat | Roofs can be invented. |
| `relight` | on (API-only) | No UI intent; call `POST /api/renders`. |
| `edit_with_reference` | on (API-only) | No UI intent; call `POST /api/renders`. |
| `entourage` | experimental | Partial results; API-only. Hidden and refused unless `ENABLE_EXPERIMENTAL_TASKS=1`. |
| `skp_material_render` | experimental | Fixed but not proven (one model, one camera). Hidden and refused unless `ENABLE_EXPERIMENTAL_TASKS=1`; the `material` intent also needs SKP passes. |
