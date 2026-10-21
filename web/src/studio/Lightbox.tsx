import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent } from "react";
import type { Job } from "../api";
import { downloadName, imageUrl, sheet } from "./util";

/**
 * A render at full size, with the view laid under it. The divider is the
 * comparison: left of it is the drawing, right of it is the render, and the
 * architect drags it across the edge they are checking.
 */
export function Lightbox({
  job,
  index,
  sourceUrl,
  onIndex,
  onClose,
}: {
  job: Job;
  index: number;
  sourceUrl: string | null;
  onIndex: (i: number) => void;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const [split, setSplit] = useState(50);
  const url = imageUrl(job, index);
  const total = job.outputs.filter((o) => o.url).length;

  // A modal dialog keeps focus inside itself; closing it hands focus back to
  // whatever opened it, which a re-rendered tile would otherwise lose. Opened
  // by a key (1 to 4) nothing has focus, so it goes to the tile for the image
  // that was last on show.
  const last = useRef(index);
  useEffect(() => {
    last.current = index;
  }, [index]);
  useEffect(() => {
    const el = dialog.current;
    const opener = document.activeElement as HTMLElement | null;
    if (el && !el.open) el.showModal();
    return () => {
      el?.close();
      if (opener && opener !== document.body && opener.isConnected) opener.focus();
      else
        document
          .querySelector<HTMLElement>(`.tile-figure[aria-keyshortcuts="${last.current + 1}"]`)
          ?.focus();
    };
  }, []);

  // Keys 1-4 move between the images of the sheet while it is open.
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      const n = Number(e.key);
      if (n >= 1 && n <= total && imageUrl(job, n - 1)) onIndex(n - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [job, total, onIndex]);

  const moveTo = (clientX: number) => {
    const box = stage.current?.getBoundingClientRect();
    if (!box || box.width === 0) return;
    setSplit(Math.round(Math.min(100, Math.max(0, ((clientX - box.left) / box.width) * 100))));
  };

  const drag = (e: PointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    moveTo(e.clientX);
  };

  const nudge = (e: KeyboardEvent) => {
    if (e.key === "ArrowLeft") setSplit((s) => Math.max(0, s - 5));
    else if (e.key === "ArrowRight") setSplit((s) => Math.min(100, s + 5));
    else return;
    e.preventDefault();
  };

  return (
    <dialog
      ref={dialog}
      className="lightbox"
      aria-label={`Sheet ${sheet(job.id)}, image ${index + 1}`}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        if (e.target === dialog.current) onClose();
      }}
    >
      <div className="lb-bar">
        <span className="label">
          Sheet {sheet(job.id)} · Image {index + 1} of {total}
        </span>
        <span className="lb-tools">
          {url && (
            <a className="btn btn-sm" href={url} download={downloadName(job, index)}>
              Save
            </a>
          )}
          <button type="button" className="btn btn-sm" onClick={onClose}>
            Close
          </button>
        </span>
      </div>

      {url && (
        <div
          className="lb-stage well"
          ref={stage}
          // The one dynamic value: where the divider is. Position and size
          // stay in the stylesheet; only this number comes from the pointer.
          style={{ "--split": `${split}%` } as React.CSSProperties}
        >
          <img className="lb-render" src={url} alt={`Image ${index + 1} of the sheet`} draggable={false} />
          {sourceUrl && (
            <>
              <img className="lb-source" src={sourceUrl} alt="" draggable={false} />
              <span className="well-tag lb-tag-left">The view</span>
              <span className="well-tag lb-tag-right">The render</span>
              <div
                className="lb-handle"
                role="slider"
                tabIndex={0}
                aria-label="Compare the view with the render"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={split}
                aria-valuetext={`${split} percent view`}
                onPointerDown={drag}
                onPointerMove={(e) => e.buttons === 1 && moveTo(e.clientX)}
                onKeyDown={nudge}
              >
                <span className="lb-grip" aria-hidden="true" />
              </div>
            </>
          )}
        </div>
      )}
      <p className="caption">
        {sourceUrl ? "Drag the divider to compare with the view. " : ""}Keys 1 to {Math.max(total, 1)} change image. Esc closes.
      </p>
    </dialog>
  );
}
