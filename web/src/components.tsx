import { useEffect, useId, useRef, useState } from "react";
import type { ReactNode } from "react";

/** A section heading: the name, an optional note, then a rule to the edge. */
export function Section({ title, note }: { title: string; note?: string }) {
  return (
    <div className="section">
      <h2>{title}</h2>
      {note && <span className="note">{note}</span>}
    </div>
  );
}

/**
 * A message.
 *
 * `live` marks it as an aria-live region. Errors and arrivals get it; static
 * explanatory text does not, or a screen reader announces the furniture.
 */
export function Note({
  children,
  kind = "plain",
  live = false,
  actions,
}: {
  children: ReactNode;
  kind?: "plain" | "error" | "good" | "blue" | "warn";
  live?: boolean;
  actions?: ReactNode;
}) {
  const cls = kind === "plain" ? "" : `is-${kind}`;
  return (
    <div
      className={`note-box ${cls}`}
      role={live ? "status" : undefined}
      aria-live={live ? "polite" : undefined}
    >
      <span className="note-text">{children}</span>
      {actions}
    </div>
  );
}

/** Title-block rows: label left, value right in mono. */
export function Spec({
  rows,
  total,
  over,
}: {
  rows: [string, string][];
  total?: [string, string];
  over?: boolean;
}) {
  return (
    <div className="spec">
      {rows.map(([k, v]) => (
        <div className="spec-row" key={k}>
          <span className="spec-key">{k}</span>
          <span className="spec-val">{v}</span>
        </div>
      ))}
      {total && (
        <div className={`spec-row is-total ${over ? "is-over" : ""}`}>
          <span className="spec-key">{total[0]}</span>
          <span className="spec-val">{total[1]}</span>
        </div>
      )}
    </div>
  );
}

/** A labelled control. The label is a real one, bound to the control's id. */
export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: (id: string) => ReactNode;
}) {
  const id = useId();
  return (
    <div className="field">
      <label className="label" htmlFor={id}>
        {label}
      </label>
      {children(id)}
      {hint && <span className="hint">{hint}</span>}
    </div>
  );
}

/**
 * A single-choice list.
 *
 * Radios are the right affordance and this is now actually one: a radiogroup
 * with aria-checked, so a screen reader says which option is selected instead
 * of reading out a row of anonymous buttons.
 */
export function OptionGroup<T extends string>({
  label,
  value,
  options,
  onPick,
}: {
  label: string;
  value: T;
  options: { value: T; label: string; disabled?: string }[];
  onPick: (value: T) => void;
}) {
  // Arrow keys skip anything disabled, so keyboard navigation cannot land on
  // an option that does nothing.
  const pickable = options.filter((o) => !o.disabled);
  return (
    <div className="options" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          aria-disabled={o.disabled ? true : undefined}
          // Kept focusable rather than `disabled` so a screen reader still
          // reaches it and reads WHY it cannot be chosen. An option that is
          // simply absent from the tab order is an option nobody is told about.
          tabIndex={o.value === value || (!value && !o.disabled) ? 0 : -1}
          className={o.disabled ? "is-unavailable" : undefined}
          title={o.disabled || undefined}
          onClick={() => {
            if (o.disabled) return;
            onPick(o.value);
          }}
          onKeyDown={(e) => {
            if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
            e.preventDefault();
            if (pickable.length === 0) return;
            const i = pickable.findIndex((x) => x.value === value);
            const next =
              e.key === "ArrowDown"
                ? pickable[(i + 1) % pickable.length]
                : pickable[(i - 1 + pickable.length) % pickable.length];
            if (next) onPick(next.value);
          }}
        >
          {o.label}
          {o.disabled && <span className="option-tag">in development</span>}
        </button>
      ))}
    </div>
  );
}

export function Empty({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="empty">
      <strong>{title}</strong>
      {children && <span>{children}</span>}
    </div>
  );
}

const IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"];
export const IMAGE_ACCEPT = IMAGE_TYPES.join(",");

/** Drag-and-drop bypasses the file input's `accept`, so it is checked here. */
export function pickImages(files: FileList | File[]): { ok: File[]; rejected: string[] } {
  const ok: File[] = [];
  const rejected: string[] = [];
  for (const f of Array.from(files)) {
    if (IMAGE_TYPES.includes(f.type)) ok.push(f);
    else rejected.push(f.name);
  }
  return { ok, rejected };
}

/**
 * The empty view. A single button: click it for the file dialog. Dragging a
 * file onto the page and pasting are handled by the page itself, so this only
 * has to say that they work.
 */
export function Dropzone({
  onOpen,
  busy,
  over,
}: {
  onOpen: () => void;
  busy?: boolean;
  over?: boolean;
}) {
  return (
    <button
      type="button"
      className={`dropzone ${over ? "is-over" : ""}`}
      onClick={onOpen}
      disabled={busy}
    >
      {busy ? (
        <>
          <span className="spin" aria-hidden="true" />
          Uploading the view
        </>
      ) : (
        <span className="dropzone-text">
          <strong>Drop a view here</strong>
          <span>or paste one, or click to choose a PNG, JPEG or WebP</span>
        </span>
      )}
    </button>
  );
}

/* ------------------------------------------------------------ mask canvas */

/**
 * Paint the area to change.
 *
 * The exported mask is written at the source image's **natural** pixel size,
 * not the size it happens to be displayed at. `flux-pro/v1/fill` requires the
 * mask and the image to match exactly and fails otherwise, and the browser
 * scaling the image to fit a column is exactly how that mismatch happens.
 *
 * White is the area to repaint, black is left alone.
 */
export function MaskCanvas({
  src,
  title,
  onExport,
  onCancel,
}: {
  src: string;
  title?: string;
  onExport: (blob: Blob) => void;
  onCancel: () => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const image = useRef<HTMLImageElement>(null);
  const drawing = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);
  const [brush, setBrush] = useState(48);
  const [ready, setReady] = useState(false);
  const [touched, setTouched] = useState(false);
  const brushId = useId();

  // Esc leaves the mask tool, the same way it leaves the lightbox.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  // `ready` stays true across a src change, so keying only on it could size the
  // canvas from the previous image's natural dimensions.
  useEffect(() => setReady(false), [src]);

  useEffect(() => {
    const img = image.current;
    const cv = canvas.current;
    if (!img || !cv || !ready) return;
    cv.width = img.naturalWidth;
    cv.height = img.naturalHeight;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cv.width, cv.height);
    setTouched(false);
  }, [ready, src]);

  const at = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const cv = canvas.current;
    if (!cv) return null;
    const rect = cv.getBoundingClientRect();
    // Displayed size differs from natural size; scale the pointer into image
    // pixels or the mark lands somewhere else entirely.
    return {
      x: (e.clientX - rect.left) * (cv.width / rect.width),
      y: (e.clientY - rect.top) * (cv.height / rect.height),
      scale: cv.width / rect.width,
    };
  };

  const paint = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const cv = canvas.current;
    if (!cv || !drawing.current) return;
    const p = at(e);
    const ctx = cv.getContext("2d");
    if (!p || !ctx) return;
    const r = (brush * p.scale) / 2;

    // A stroke is a line, not a row of dots. One arc per pointer event leaves
    // gaps on a fast drag, and flux-pro/v1/fill inpaints those gaps as
    // speckles across the repaired area.
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = r * 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (last.current) {
      ctx.beginPath();
      ctx.moveTo(last.current.x, last.current.y);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
    last.current = { x: p.x, y: p.y };
    setTouched(true);
  };

  const stop = () => {
    drawing.current = false;
    last.current = null;
  };

  const clear = () => {
    const cv = canvas.current;
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx) return;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cv.width, cv.height);
    setTouched(false);
  };

  return (
    <div className="stack">
      <Section title="Mark the region" note={title} />

      <div className="mask-stage">
        <img
          ref={image}
          src={src}
          alt="The sheet being marked"
          onLoad={() => setReady(true)}
          draggable={false}
        />
        <canvas
          ref={canvas}
          onPointerDown={(e) => {
            drawing.current = true;
            last.current = null;
            e.currentTarget.setPointerCapture(e.pointerId);
            paint(e);
          }}
          onPointerMove={paint}
          onPointerUp={stop}
          // A cancelled touch pointer used to leave the canvas painting on
          // every subsequent hover, forever.
          onPointerCancel={stop}
          onLostPointerCapture={stop}
        />
      </div>

      <div className="row">
        <div className="field">
          <label className="label" htmlFor={brushId}>
            Brush · {brush}px
          </label>
          <input
            id={brushId}
            type="range"
            min={8}
            max={200}
            value={brush}
            onChange={(e) => setBrush(Number(e.target.value))}
          />
        </div>
        <div className="shrink actions">
          <button type="button" className="btn" onClick={clear} disabled={!touched}>
            Clear
          </button>
          <button type="button" className="btn" onClick={onCancel}>
            Cancel
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!touched}
            onClick={() =>
              canvas.current?.toBlob((b) => {
                if (b) onExport(b);
              }, "image/png")
            }
          >
            Done marking
          </button>
        </div>
      </div>

      <p className="caption">
        Paint what should change. Everything you leave unpainted is kept exactly as it is.
      </p>
    </div>
  );
}
