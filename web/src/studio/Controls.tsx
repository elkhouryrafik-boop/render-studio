import { useId } from "react";
import type { ChangeEvent, KeyboardEvent } from "react";
import type { Asset, Intent, Intents } from "../api";
import { Dropzone, IMAGE_ACCEPT } from "../components";
import { RadioGroup } from "./Radio";
import type { Base, DepthChip } from "./types";
import { money, sheet } from "./util";

export const LINE_MAX = 240;

/** Every numbered note: the number, then the name, in the one label treatment. */
function NoteHead({ no, id, children }: { no: number; id: string; children: string }) {
  return (
    <h2 className="note-head" id={id}>
      <span className="note-no" aria-hidden="true">
        {no}
      </span>
      {children}
    </h2>
  );
}

/* ------------------------------------------------------------- 1 View */

export function ViewNote({
  base,
  thumbUrl,
  depthUrl,
  chip,
  uploading,
  over,
  hasMaterials,
  onOpenFile,
  onCad,
  onToggleDepth,
}: {
  base: Base | null;
  thumbUrl: string | null;
  depthUrl: string | null;
  chip: DepthChip | null;
  uploading: boolean;
  over: boolean;
  hasMaterials: boolean;
  onOpenFile: () => void;
  onCad: () => void;
  onToggleDepth: () => void;
}) {
  const id = useId();
  const showingDepth = Boolean(chip?.showing && depthUrl);
  const links = (
    <div className="note-links">
      {base && (
        <button type="button" className="btn btn-ghost btn-sm" onClick={onOpenFile} disabled={uploading}>
          Replace the view
        </button>
      )}
      <button type="button" className="btn btn-ghost btn-sm" onClick={onCad}>
        Convert a DXF, IFC or SKP
      </button>
    </div>
  );
  return (
    <section className="note" aria-labelledby={id}>
      <NoteHead no={1} id={id}>
        View
      </NoteHead>

      {!base || !thumbUrl ? (
        <Dropzone onOpen={onOpenFile} busy={uploading} over={over} />
      ) : (
        <div className="view">
          <div className="well view-well">
            <img
              src={showingDepth ? depthUrl! : thumbUrl}
              alt={showingDepth ? "The depth pass of the view" : "The view"}
            />
            {showingDepth && <span className="well-tag">Depth</span>}
          </div>

          <div className="view-meta">
            {base.kind === "sheet" && (
              <p className="view-based">
                Based on sheet {sheet(base.job.id)}
                {base.job.outputs.length > 1 ? `, image ${base.index + 1}` : ""}
              </p>
            )}
            {hasMaterials && <p className="view-based">Materials from SketchUp attached</p>}
            {links}
            {chip && (
              <div className="chip-row">
                <span className="state-chip">{chip.label}</span>
                {chip.canShow && (
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    aria-pressed={chip.showing}
                    onClick={onToggleDepth}
                    disabled={chip.loading}
                  >
                    {chip.loading ? "Reading depth" : chip.showing ? "Hide depth" : "Show depth"}
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {(!base || !thumbUrl) && links}
    </section>
  );
}

/* ------------------------------------------------------ 2 What to do */

export function IntentNote({
  cards,
  intent,
  viewKind,
  refs,
  uploadingRef,
  hasBase,
  markedRegion,
  onIntent,
  onView,
  onAddRef,
  onRemoveRef,
  onMark,
}: {
  cards: Intent[];
  intent: Intent | null;
  viewKind: string | null;
  refs: Asset[];
  uploadingRef: boolean;
  hasBase: boolean;
  markedRegion: boolean;
  onIntent: (key: string) => void;
  onView: (key: string) => void;
  onAddRef: (files: FileList) => void;
  onRemoveRef: (id: number) => void;
  onMark: () => void;
}) {
  const id = useId();
  const refInput = useId();
  return (
    <section className="note" aria-labelledby={id}>
      <NoteHead no={2} id={id}>
        What to do
      </NoteHead>

      <RadioGroup
        label="What to do"
        className="intent-cards"
        itemClassName="intent-card"
        value={intent?.key ?? null}
        onPick={onIntent}
        items={cards.map((c) => ({
          value: c.key,
          hint: c.hint,
          label: (
            <>
              <span className="intent-label">{c.label}</span>
              <span className="intent-hint">{c.hint}</span>
            </>
          ),
        }))}
      />

      {intent && intent.views.length > 0 && (
        <div className="sub sub-inline">
          <span className="label">This view is a:</span>
          <RadioGroup
            label="This view is a"
            className="chips"
            itemClassName="chip"
            value={viewKind}
            onPick={onView}
            items={intent.views.map((v) => ({ value: v.key, label: v.label }))}
          />
        </div>
      )}

      {intent && intent.max_references > 0 && (
        <div className="sub">
          <span className="label">Reference images (optional)</span>
          <div className="ref-strip">
            {refs.map((r, i) => (
              <div className="ref" key={r.id}>
                <img className="well ref-thumb" src={r.url} alt={`Reference ${i + 1}`} />
                <button
                  type="button"
                  className="btn btn-sm btn-danger"
                  onClick={() => onRemoveRef(r.id)}
                  aria-label={`Remove reference ${i + 1}`}
                >
                  Remove
                </button>
              </div>
            ))}
            {refs.length < intent.max_references && (
              <>
                <label className="btn btn-sm" htmlFor={refInput}>
                  {uploadingRef ? "Uploading" : refs.length ? "Add another" : "Add a reference"}
                </label>
                <input
                  id={refInput}
                  type="file"
                  className="sr-only"
                  accept={IMAGE_ACCEPT}
                  multiple
                  onChange={(e: ChangeEvent<HTMLInputElement>) => {
                    if (e.target.files?.length) onAddRef(e.target.files);
                    e.target.value = "";
                  }}
                />
              </>
            )}
          </div>
        </div>
      )}

      {intent?.needs_mask && (
        <div className="sub sub-row">
          <span className="state-chip">{markedRegion ? "Region marked" : "No region marked"}</span>
          <button type="button" className="btn btn-sm" onClick={onMark} disabled={!hasBase}>
            {markedRegion ? "Mark it again" : "Mark the region"}
          </button>
        </div>
      )}
    </section>
  );
}

/* --------------------------------------------------------- 3 Geometry */

export function GeometryNote({
  catalogue,
  value,
  exactOk,
  drawing,
  onChange,
}: {
  catalogue: Intents;
  value: number;
  exactOk: boolean;
  /** The view is a drawing (sketch, plan, master plan, elevation): no slider. */
  drawing: boolean;
  onChange: (v: number) => void;
}) {
  const id = useId();
  const ticks = useId();
  const snaps = catalogue.geometry_hold.snaps;
  const top = Math.max(...snaps.map((s) => s.value));
  const below = snaps.filter((s) => s.value < top).reduce((m, s) => Math.max(m, s.value), 0);
  const max = exactOk ? top : below;
  const nearest = snaps.reduce((a, b) => (Math.abs(b.value - value) < Math.abs(a.value - value) ? b : a));

  const set = (raw: number) => {
    // A pull toward the three named positions, so Hold is easy to land on and
    // the in-between is still there for someone who wants it.
    const near = snaps.find((s) => Math.abs(s.value - raw) <= 6);
    onChange(Math.min(near ? near.value : raw, max));
  };

  if (drawing) {
    return (
      <section className="note" aria-labelledby={id}>
        <NoteHead no={3} id={id}>
          Geometry
        </NoteHead>
        <p className="hint">Drawings keep their tuned line hold.</p>
      </section>
    );
  }

  const shown = Math.min(value, max);
  const onSnap = snaps.find((s) => s.value === shown);

  // Arrow keys and Page keys step to the neighbouring named position (the
  // native +/-1 would be pulled straight back to where it started); Home and
  // End go to the ends of what is allowed.
  const ordered = snaps.map((s) => s.value).sort((a, b) => a - b);
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    const up = ["ArrowRight", "ArrowUp", "PageUp"].includes(e.key);
    const down = ["ArrowLeft", "ArrowDown", "PageDown"].includes(e.key);
    if (!up && !down && e.key !== "Home" && e.key !== "End") return;
    e.preventDefault();
    if (e.key === "Home") return onChange(ordered[0] ?? 0);
    if (e.key === "End") return onChange(max);
    const next = up
      ? ordered.find((v) => v > shown && v <= max)
      : [...ordered].reverse().find((v) => v < shown);
    if (next != null) onChange(next);
  };

  return (
    <section className="note" aria-labelledby={id}>
      <NoteHead no={3} id={id}>
        Geometry
      </NoteHead>
      <div className="geo">
        <input
          type="range"
          min={0}
          max={top}
          step={1}
          value={shown}
          list={ticks}
          aria-label="How closely to hold the drawing"
          aria-valuetext={onSnap ? onSnap.label : `Between snaps, nearest ${nearest.label}`}
          aria-describedby={`${id}-hint`}
          onChange={(e) => set(Number(e.target.value))}
          onKeyDown={onKey}
        />
        <datalist id={ticks}>
          {snaps.map((s) => (
            <option key={s.key} value={s.value} label={s.label} />
          ))}
        </datalist>
        {/* Plain text under the track, not controls: the slider is the one way in. */}
        <div className="geo-labels" aria-hidden="true">
          {snaps.map((s, i) => {
            const locked = s.value > max;
            const place = i === 0 ? "is-first" : i === snaps.length - 1 ? "is-last" : "is-mid";
            return (
              <span
                key={s.key}
                className={`geo-label ${place} ${nearest.key === s.key ? "is-on" : ""} ${locked ? "is-locked" : ""}`}
                onClick={() => !locked && set(s.value)}
              >
                {s.label}
              </span>
            );
          })}
        </div>
        <p className="hint" id={`${id}-hint`} aria-live="polite">
          {nearest.hint}
          {!exactOk && " Edges only: install the depth estimator to use Exact."}
        </p>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------ 4 Look */

export function LookNote({
  catalogue,
  style,
  line,
  required,
  label,
  placeholder,
  onStyle,
  onLine,
}: {
  catalogue: Intents;
  style: string;
  line: string;
  required: boolean;
  label: string;
  placeholder: string;
  onStyle: (key: string) => void;
  onLine: (text: string) => void;
}) {
  const id = useId();
  const field = useId();
  return (
    <section className="note" aria-labelledby={id}>
      <NoteHead no={4} id={id}>
        Look
      </NoteHead>
      <RadioGroup
        label="Look"
        className="chips"
        itemClassName="chip"
        value={style}
        onPick={onStyle}
        items={catalogue.styles.map((s) => ({ value: s.key, label: s.label }))}
      />
      <div className="field">
        <label className="label" htmlFor={field}>
          {label}
          {required ? "" : " (optional)"}
        </label>
        <input
          id={field}
          type="text"
          value={line}
          maxLength={LINE_MAX}
          placeholder={placeholder}
          required={required}
          autoComplete="off"
          onChange={(e) => onLine(e.target.value)}
        />
        {line.length >= LINE_MAX - 40 && (
          <span className="caption" aria-live="polite">
            {line.length} of {LINE_MAX}
          </span>
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------ footer */

export function Footer({
  count,
  countFixed,
  maxCount,
  sizeKey,
  sizes,
  estimate,
  overBy,
  busy,
  reason,
  showReason,
  onCount,
  onSize,
}: {
  count: number;
  /** The count is not a choice for this intent (a region is one image). */
  countFixed: boolean;
  maxCount: number;
  sizeKey: string;
  sizes: Intents["sizes"];
  estimate: number | null;
  overBy: number | null;
  busy: boolean;
  /** Why Render is not available yet; null when it is. */
  reason: string | null;
  /** True once Render has been pressed while the form was not ready. */
  showReason: boolean;
  onCount: (n: number) => void;
  onSize: (key: string) => void;
}) {
  const sizeId = useId();
  const countId = useId();
  const over = overBy != null && overBy > 0;
  return (
    <div className={`footer ${over ? "is-over" : ""}`}>
      {over && (
        <p className="footer-over" role="status">
          Over the cap by {money(overBy)}. Choose 1 image or wait for next month.
        </p>
      )}
      <div className="footer-row">
        <div className="cost-line">
          {!countFixed && (
            <>
            <label className="sr-only" htmlFor={countId}>
              Number of images
            </label>
            <select
              id={countId}
              className="size-select"
              value={count === 1 ? 1 : maxCount}
              disabled={maxCount <= 1}
              onChange={(e) => onCount(Number(e.target.value))}
            >
              <option value={1}>1 image</option>
              {maxCount > 1 && <option value={maxCount}>{maxCount} images</option>}
            </select>
            <span aria-hidden="true">·</span>
            </>
          )}
          <label className="sr-only" htmlFor={sizeId}>
            Image size
          </label>
          <select id={sizeId} className="size-select" value={sizeKey} onChange={(e) => onSize(e.target.value)}>
            {sizes.map((s) => (
              <option key={s.key} value={s.key}>
                {s.label}
              </option>
            ))}
          </select>
          <span aria-hidden="true">·</span>
          <span className="cost" aria-label={estimate == null ? "Cost not yet known" : `Cost ${money(estimate)}`}>
            {estimate == null ? "$ …" : money(estimate)}
          </span>
        </div>
        <button
          type="submit"
          className="btn btn-primary"
          disabled={busy || over}
          aria-disabled={reason != null || undefined}
          title={over ? "Over the cap" : (reason ?? undefined)}
          aria-describedby={reason && showReason && !over ? "render-reason" : undefined}
        >
          {busy && <span className="spin" aria-hidden="true" />}
          {over ? "Over the cap" : "Render"}
        </button>
      </div>
      {reason && showReason && !over && (
        <p className="caption render-reason" id="render-reason" role="status">
          {reason}
        </p>
      )}
    </div>
  );
}
