import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  CAD_CAMERAS,
  CadPendingError,
  SKP_CAMERAS,
  api,
  describeError,
  type Asset,
  type CadBakePlan,
  type CadLayer,
  type CadMaterial,
  type SkpCamera,
} from "./api";
import { Empty, Field, Note, OptionGroup, Section } from "./components";

/** Seconds as something a person reads at a glance: "45s", "3 min", "16 min". */
function formatDuration(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)}s`;
  return `${Math.round(seconds / 60)} min`;
}

type CadFormat = "dxf" | "ifc" | "skp";

const ACCEPTED = ".dxf, .ifc and .skp";

/** Unknown extensions are refused rather than guessed at. A `.png` dropped on
 * this zone used to be parsed as a DXF and come back as an empty drawing. */
function formatOf(name: string): CadFormat | null {
  const lower = name.toLowerCase();
  if (lower.endsWith(".dxf")) return "dxf";
  if (lower.endsWith(".ifc")) return "ifc";
  if (lower.endsWith(".skp")) return "skp";
  return null;
}

const CAMERA_LABEL: Record<SkpCamera, string> = {
  iso: "Isometric",
  front: "Front elevation",
  top: "Plan, from above",
  "buildings-iso": "Buildings only, isometric",
};

// Baking a large SKP file takes minutes, not seconds (proportional to
// file size). Polling every 3s is cheap and keeps the progress line honest
// without hammering the server; it eases off from there so a file that is
// going to take a quarter of an hour is not asked about 300 times.
const BAKE_POLL_MS = 3000;
const BAKE_POLL_MAX_MS = 15000;
const BAKE_POLL_LIMIT = 150;

// Anything past the first figure takes minutes rather than seconds, so it is
// worth saying so before the upload starts instead of after. Past the second,
// the converter will not finish at all.
const LARGE_FILE_MB = 25;
const MAX_FILE_MB = 250;

const mbOf = (file: File) => file.size / (1024 * 1024);

/**
 * CAD file in, a normal reference image out.
 *
 * fal.ai reads pixels only, so a CAD file is never itself attachable: it is
 * uploaded, previewed and adjusted here, and only `confirm` turns it into the
 * PNG that becomes an ordinary reference. Nothing before that step touches
 * fal.ai or spends a credit, which is the whole point of having a preview at
 * all: a bad rasterization costs nothing to redo here, and everything to redo
 * after a render.
 *
 * DXF shows layer checkboxes and a line-weight control; IFC and SKP show
 * storeys/layers and camera presets instead (a 3D model has no single "the"
 * view). SKP's confirm also returns a materially-correct albedo pass, a
 * per-pixel confidence map, and a finish schedule: the resolved-materials
 * table shown after a successful confirm, before the panel hands off to the
 * render form.
 */
export function CadPanel({
  onConfirmed,
  onCancel,
}: {
  onConfirmed: (
    asset: Asset,
    depthAssetId?: number,
    albedoAssetId?: number,
    changeAssetId?: number,
  ) => void;
  onCancel: () => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const layersLabelId = useId();

  const [cadAsset, setCadAsset] = useState<Asset | null>(null);
  const [format, setFormat] = useState<CadFormat>("dxf");
  const [layers, setLayers] = useState<CadLayer[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [lineWeight, setLineWeight] = useState(1);
  const [camera, setCamera] = useState<SkpCamera>("iso");

  const [uploading, setUploading] = useState(false);
  const [baking, setBaking] = useState<{
    stage: string;
    elapsedSeconds: number;
    estimatedSeconds: number;
  } | null>(null);
  // What this file will cost in TIME, asked before anyone commits to it. A
  // 16-minute wait with no number on screen reads as a hang.
  const [bakePlan, setBakePlan] = useState<CadBakePlan | null>(null);
  const [converting, setConverting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [oversize, setOversize] = useState<{ file: File; fmt: CadFormat; mb: number } | null>(
    null,
  );

  // Every checkbox used to refire a full server-side rasterization. The set the
  // preview is actually drawn from lags the checkboxes by a moment.
  const [previewOf, setPreviewOf] = useState<string[]>([]);
  const [preview, setPreview] = useState<"loading" | "ready" | "error">("loading");
  const [previewAttempt, setPreviewAttempt] = useState(0);

  // Set only for SKP: confirm has already produced everything, and this
  // holds the panel open on the finish schedule until the architect
  // acknowledges it, instead of attaching and closing immediately the way
  // DXF/IFC do.
  const [confirmed, setConfirmed] = useState<{
    asset: Asset;
    depthAssetId?: number;
    albedoAssetId?: number;
    changeAssetId?: number;
    materials: CadMaterial[];
  } | null>(null);

  // The bake poll used to reschedule itself with a bare setTimeout. Cancelling
  // mid-bake unmounted the panel and left the chain running against
  // /api/cad/{id}/layers forever.
  const pollTimer = useRef<number | null>(null);
  const polls = useRef(0);
  const mounted = useRef(true);

  const stopPolling = () => {
    if (pollTimer.current !== null) {
      window.clearTimeout(pollTimer.current);
      pollTimer.current = null;
    }
    polls.current = 0;
  };

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      if (pollTimer.current !== null) window.clearTimeout(pollTimer.current);
      pollTimer.current = null;
    };
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => setPreviewOf(Array.from(selected)), 400);
    return () => window.clearTimeout(timer);
  }, [selected]);

  const is3d = format === "ifc" || format === "skp";

  const previewSrc = useMemo(() => {
    if (!cadAsset || previewOf.length === 0) return "";
    return api.cadPreviewUrl(cadAsset.id, previewOf, is3d ? camera : undefined);
  }, [cadAsset, previewOf, is3d, camera]);

  useEffect(() => setPreview("loading"), [previewSrc, previewAttempt]);

  const reset = () => {
    stopPolling();
    setCadAsset(null);
    setFormat("dxf");
    setLayers([]);
    setSelected(new Set());
    setPreviewOf([]);
    setLineWeight(1);
    setCamera("iso");
    setError(null);
    setWarnings([]);
    setBaking(null);
    setConfirmed(null);
    setOversize(null);
  };

  const cancel = () => {
    stopPolling();
    onCancel();
  };

  // Fetch the layer manifest; for SKP this is also what discovers whether the
  // file's background bake is still running. `CadPendingError` is not a
  // failure: it schedules the next poll and returns quietly.
  const loadLayers = async (assetId: number) => {
    try {
      const manifest = await api.cadLayers(assetId);
      if (!mounted.current) return;
      stopPolling();
      setBaking(null);
      setLayers(manifest);
      const visible = manifest.filter((l) => l.visible).map((l) => l.name);
      setSelected(new Set(visible));
      setPreviewOf(visible);
    } catch (e) {
      if (!mounted.current) return;
      if (e instanceof CadPendingError) {
        setBaking({
          stage: e.stage,
          elapsedSeconds: e.elapsedSeconds,
          estimatedSeconds: e.estimatedSeconds,
        });
        polls.current += 1;
        if (polls.current > BAKE_POLL_LIMIT) {
          stopPolling();
          setBaking(null);
          setError(
            `This file has been baking for ${Math.round(
              e.elapsedSeconds / 60,
            )} minutes and this window has stopped waiting for it. The bake may still finish on the server: open the same file again in a few minutes, or export fewer storeys.`,
          );
          return;
        }
        const wait = Math.min(BAKE_POLL_MS * 1.15 ** polls.current, BAKE_POLL_MAX_MS);
        pollTimer.current = window.setTimeout(() => {
          pollTimer.current = null;
          void loadLayers(assetId);
        }, wait);
        return;
      }
      stopPolling();
      setError(describeError(e));
    }
  };

  const upload = async (file: File, fmt: CadFormat) => {
    setUploading(true);
    setError(null);
    setWarnings([]);
    try {
      const asset = await api.cadUpload(file);
      if (!mounted.current) return;
      setCadAsset(asset);
      setFormat(fmt);
      // Advisory only: if this endpoint fails the conversion still works, so
      // it must never block the upload it is describing.
      api
        .cadBakePlan(asset.id)
        .then((plan) => mounted.current && setBakePlan(plan))
        .catch(() => undefined);
      await loadLayers(asset.id);
    } catch (e) {
      if (mounted.current) setError(describeError(e));
    } finally {
      if (mounted.current) setUploading(false);
    }
  };

  const pickFile = (file: File) => {
    setError(null);
    setWarnings([]);
    setOversize(null);
    const fmt = formatOf(file.name);
    if (!fmt) {
      setError(`${file.name} is not a drawing this can read. It takes ${ACCEPTED} files.`);
      return;
    }
    const mb = mbOf(file);
    if (mb > MAX_FILE_MB) {
      setError(
        `${file.name} is ${mb.toFixed(0)} MB. The converter takes files up to ${MAX_FILE_MB} MB: export the storeys you need and try that instead.`,
      );
      return;
    }
    if (mb > LARGE_FILE_MB) {
      setOversize({ file, fmt, mb });
      return;
    }
    void upload(file, fmt);
  };

  const toggleLayer = (name: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const confirm = async () => {
    if (!cadAsset || selected.size === 0) return;
    setConverting(true);
    setError(null);
    try {
      const result = await api.cadConfirm(cadAsset.id, {
        layers: Array.from(selected),
        line_weight_scale: lineWeight,
        camera,
      });
      if (!mounted.current) return;
      setWarnings(result.warnings);
      if (format === "skp") {
        // Hold the panel open on the finish schedule rather than attaching
        // and closing immediately: this table is the actual point of an SKP
        // conversion, not a side effect of it.
        setConfirmed({
          asset: result.asset,
          depthAssetId: result.depth_asset?.id,
          albedoAssetId: result.albedo_asset?.id,
          changeAssetId: result.change_asset?.id,
          materials: result.materials,
        });
      } else {
        // Clear up first: `onConfirmed` hands the asset to the render form,
        // which closes this panel, so anything set afterwards lands nowhere.
        reset();
        onConfirmed(result.asset, result.depth_asset?.id);
      }
    } catch (e) {
      if (mounted.current) setError(describeError(e));
    } finally {
      if (mounted.current) setConverting(false);
    }
  };

  /* ------------------------------------------------------ finish schedule */

  if (confirmed) {
    return (
      <div className="cad-panel">
        <Section
          title="Finish schedule"
          note={`${confirmed.materials.length} resolved surface${
            confirmed.materials.length === 1 ? "" : "s"
          }, by share of the view`}
        />
        {confirmed.materials.length === 0 ? (
          <Empty title="No materials resolved on this camera">
            Nothing visible from here carries a named material. The render will invent every
            surface, so describe the finishes in the prompt.
          </Empty>
        ) : (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>
                    <span className="sr-only">Colour</span>
                  </th>
                  <th>Material</th>
                  <th>Confidence</th>
                  <th className="num">Share</th>
                  <th className="num">Tile</th>
                </tr>
              </thead>
              <tbody>
                {confirmed.materials.map((m, i) => (
                  <tr key={`${m.material_class}-${i}`}>
                    <td>
                      <span
                        className="cad-material-swatch"
                        style={{
                          background: `rgb(${Math.round(m.albedo[0] * 255)}, ${Math.round(
                            m.albedo[1] * 255,
                          )}, ${Math.round(m.albedo[2] * 255)})`,
                        }}
                      />
                    </td>
                    <td>{m.material_class}</td>
                    <td>
                      <span className={`cad-confidence cad-confidence-${m.confidence}`}>
                        {m.confidence}
                      </span>
                    </td>
                    <td className="num">{(m.share_of_visible_pixels * 100).toFixed(1)}%</td>
                    <td className="num">
                      {m.tile_width_mm ? `${Math.round(m.tile_width_mm)} mm` : "not stated"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="caption">
          Each row is a claim read from the file itself, not guessed from the image. "stated"
          means a named library material with a texture; "unknown" means the render is free to
          reinvent that surface. Attaching this view also carries the depth, albedo and
          confidence-map passes along, so the Material render can
          use them.
        </p>
        {warnings.map((w, i) => (
          <Note key={i} kind="warn" live>
            {w}
          </Note>
        ))}
        <div className="actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => {
              const done = confirmed;
              reset();
              onConfirmed(
                done.asset,
                done.depthAssetId,
                done.albedoAssetId,
                done.changeAssetId,
              );
            }}
          >
            Use this view
          </button>
          <button type="button" className="btn" onClick={cancel}>
            Cancel
          </button>
        </div>
      </div>
    );
  }

  /* -------------------------------------------------------------- pick up */

  if (!cadAsset) {
    return (
      <div className="cad-panel">
        <button
          type="button"
          className="dropzone"
          onClick={() => input.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const file = e.dataTransfer.files[0];
            if (file) pickFile(file);
          }}
        >
          {uploading ? (
            <>
              <span className="spin" />
              Uploading
            </>
          ) : (
            <span>Drop a DXF, IFC or SKP file here, or click to choose one</span>
          )}
        </button>
        <input
          ref={input}
          type="file"
          accept=".dxf,.ifc,.skp"
          className="sr-only"
          aria-hidden="true"
          tabIndex={-1}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) pickFile(file);
            e.target.value = "";
          }}
        />

        {oversize && (
          <Note
            kind="warn"
            live
            actions={
              <div className="actions">
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  onClick={() => {
                    const chosen = oversize;
                    setOversize(null);
                    void upload(chosen.file, chosen.fmt);
                  }}
                >
                  Upload it anyway
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => setOversize(null)}
                >
                  Choose another file
                </button>
              </div>
            }
          >
            {oversize.file.name} is {oversize.mb.toFixed(0)} MB. A file this size takes several
            minutes to parse and bake the first time, with nothing to look at while it runs.
            Every conversion after the first reuses the cache.
          </Note>
        )}

        {error && (
          <Note kind="error" live>
            {error}
          </Note>
        )}

        <div className="actions">
          <button type="button" className="btn btn-sm" onClick={cancel}>
            Cancel
          </button>
        </div>
      </div>
    );
  }

  /* ---------------------------------------------------------------- baking */

  if (baking) {
    return (
      <div className="cad-panel">
        <Section title="Baking" note={baking.stage.replace(/_/g, " ")} />
        <p className="caption" role="status">
          Parsing and baking this file: {baking.stage.replace(/_/g, " ")},{" "}
          {formatDuration(baking.elapsedSeconds)} so far
          {baking.estimatedSeconds > 0 && (
            <> of about {formatDuration(baking.estimatedSeconds)}</>
          )}
          . <strong>This happens only once for this file.</strong> The result is
          cached, so every conversion of it after this one — another camera, a
          different layer selection, another view — takes seconds. You can leave
          this page; it keeps going in the background.
        </p>
        <div className="actions">
          <button type="button" className="btn btn-sm" onClick={cancel}>
            Cancel
          </button>
        </div>
      </div>
    );
  }

  /* --------------------------------------------------------------- adjust */

  const isIfc = format === "ifc";
  const isSkp = format === "skp";
  const cameras: SkpCamera[] = isSkp ? [...SKP_CAMERAS] : [...CAD_CAMERAS];

  return (
    <div className="cad-panel">
      <Section title="Preview" note="free, nothing is billed until you use this" />

      {/* Said before the wait, not during it. A SketchUp model is parsed once
          and cached by content hash; on a ~27 MB test model that first
          parse is 16 minutes, and someone who has not been told will assume the
          app has hung and kill it. Only shown when there is actually a wait. */}
      {bakePlan && !bakePlan.cached && (
        <p className="caption cad-bake-notice" role="status">
          <strong>
            First conversion of this file: about {formatDuration(bakePlan.estimated_seconds)}.
          </strong>{" "}
          That happens only once — the result is cached, so every conversion of
          the same file after this takes seconds. You can leave this page; it
          keeps going in the background.
        </p>
      )}

      {selected.size === 0 ? (
        <Empty title={`Select at least one ${is3d ? "storey or layer" : "layer"}`}>
          The preview draws whatever is ticked below. Nothing is rendered until you
          choose Use this view.
        </Empty>
      ) : (
        <>
          <div className="well cad-preview">
            {preview !== "error" && previewSrc && (
              <img
                key={previewAttempt}
                src={previewSrc}
                alt="The layers you selected, drawn as they will be sent"
                onLoad={() => setPreview("ready")}
                onError={() => setPreview("error")}
              />
            )}
          </div>
          {preview === "loading" && (
            <p className="caption" role="status">
              Drawing the preview.
            </p>
          )}
          {preview === "error" && (
            <Note
              kind="warn"
              live
              actions={
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => setPreviewAttempt((n) => n + 1)}
                >
                  Draw it again
                </button>
              }
            >
              The preview did not draw. The file may be too heavy for this camera, or the
              server may be restarting.
            </Note>
          )}
        </>
      )}

      {is3d && (
        <OptionGroup
          label="Camera"
          value={camera}
          options={cameras.map((c) => ({ value: c, label: CAMERA_LABEL[c] }))}
          onPick={setCamera}
        />
      )}
      {isSkp && camera === "buildings-iso" && (
        <p className="caption">Frames on the built mass rather than the whole site.</p>
      )}

      <div className="field">
        <span className="label" id={layersLabelId}>
          {is3d ? "Storeys and layers" : "Layers"}
        </span>
        <div className="cad-layer-list" role="group" aria-labelledby={layersLabelId}>
          {layers.map((layer) => (
            <label key={layer.name} className="cad-layer-row">
              <input
                type="checkbox"
                checked={selected.has(layer.name)}
                onChange={() => toggleLayer(layer.name)}
              />
              {layer.color && (
                <span className="cad-layer-swatch" style={{ background: layer.color }} />
              )}
              {layer.name}
            </label>
          ))}
        </div>
      </div>

      {!is3d && (
        <Field label="Line weight" hint={`${lineWeight.toFixed(2)} times the drawn weight`}>
          {(id) => (
            <input
              id={id}
              type="range"
              min={0.25}
              max={3}
              step={0.25}
              value={lineWeight}
              onChange={(e) => setLineWeight(Number(e.target.value))}
            />
          )}
        </Field>
      )}

      {isIfc && (
        <p className="caption">
          Confirming also produces a true depth pass, so Hold and Exact follow the
          drawn depth instead of estimating it.
        </p>
      )}
      {isSkp && (
        <p className="caption">
          Confirming resolves every surface's material from the file itself (layer names,
          library materials, real-world tile widths) and produces a true depth pass, a
          materially-correct base image, and a per-surface confidence map alongside the usual
          view.
        </p>
      )}

      {warnings.map((w, i) => (
        <Note key={i} kind="warn" live>
          {w}
        </Note>
      ))}
      {error && (
        <Note kind="error" live>
          {error}
        </Note>
      )}

      <div className="actions">
        <button
          type="button"
          className="btn btn-primary"
          disabled={selected.size === 0 || converting}
          onClick={confirm}
        >
          {converting && <span className="spin" />}
          Use this view
        </button>
        <button type="button" className="btn" onClick={reset}>
          Try a different file
        </button>
        <button type="button" className="btn" onClick={cancel}>
          Cancel
        </button>
      </div>
    </div>
  );
}
