import { useEffect, useRef, useState } from "react";
import type { Intents, Job } from "../api";
import { Note } from "../components";
import { Pic } from "./Pic";
import {
  PENDING,
  ageSeconds,
  clock,
  downloadName,
  firstImage,
  imageUrl,
  jobTitle,
  initials,
  isMine,
  sheet,
  statusWord,
} from "./util";
import type { StatusWord } from "./util";

const STATUS_CLASS: Record<StatusWord, string> = {
  queued: "queued",
  rendering: "in_progress",
  done: "completed",
  failed: "failed",
};

/** The one pill for a job's standing, always a word as well as a colour. */
export function StatusPill({ status }: { status: string }) {
  const word = statusWord(status);
  return <span className={`status ${STATUS_CLASS[word]}`}>{word}</span>;
}

/** Whose sheet it is, only when it is not the signed-in person's own. */
function Owner({ job, me }: { job: Job; me: string }) {
  if (isMine(job, me)) return null;
  const who = job.username.split("@")[0];
  return (
    <span className="owner-tag" title={who}>
      <span className="sr-only">By {who}, </span>
      <span aria-hidden="true">{initials(job.username)}</span>
    </span>
  );
}

/* --------------------------------------------------------------- tiles */

interface TileActions {
  onUse: (job: Job, index: number) => void;
  onRegion: (job: Job, index: number) => void;
  onEnlarge: (job: Job, index: number) => void;
  onOpen: (index: number) => void;
}

/** How long a press must last before it turns into "hold to compare". */
const HOLD_MS = 220;

function Tile({
  job,
  index,
  url,
  sourceUrl,
  actions,
  busy,
}: {
  job: Job;
  index: number;
  url: string;
  sourceUrl: string | null;
  actions: TileActions;
  busy: boolean;
}) {
  const [comparing, setComparing] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  const held = useRef(false);
  const n = index + 1;

  useEffect(() => () => window.clearTimeout(timer.current), []);

  const start = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    held.current = false;
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      held.current = true;
      setComparing(true);
    }, HOLD_MS);
  };
  const end = () => {
    window.clearTimeout(timer.current);
    if (held.current) setComparing(false);
  };

  const showSource = comparing && sourceUrl;

  return (
    <article className="tile" aria-label={`Image ${n}`}>
      <button
        type="button"
        className="well tile-figure"
        aria-label={`Image ${n}. Open full size. Hold to compare with the view.`}
        aria-keyshortcuts={String(n)}
        onContextMenu={(e) => e.preventDefault()}
        onPointerDown={start}
        onPointerUp={end}
        onPointerLeave={end}
        onPointerCancel={end}
        onClick={() => {
          if (held.current) {
            held.current = false;
            return;
          }
          actions.onOpen(index);
        }}
        onKeyDown={(e) => {
          if (e.key.toLowerCase() === "c" && !e.repeat) setComparing(true);
        }}
        onKeyUp={(e) => {
          if (e.key.toLowerCase() === "c") setComparing(false);
        }}
        onBlur={() => setComparing(false)}
      >
        <Pic src={showSource ? sourceUrl : url} alt="" />
        <span className="tile-no" aria-hidden="true">
          {n}
        </span>
        {showSource && <span className="well-tag">The view</span>}
      </button>
      <span className="sr-only" role="status">
        {showSource ? `Image ${n}: showing the view. Release to return to the render.` : ""}
      </span>

      <div className="tile-actions">
        <button type="button" className="btn btn-sm" aria-label={`Use as base, image ${n}`} onClick={() => actions.onUse(job, index)}>
          Use as base
        </button>
        <button type="button" className="btn btn-sm" aria-label={`Change a region, image ${n}`} onClick={() => actions.onRegion(job, index)}>
          Change a region
        </button>
        <button
          type="button"
          className="btn btn-sm"
          aria-label={`Enlarge, image ${n}`}
          disabled={busy}
          onClick={() => actions.onEnlarge(job, index)}
        >
          Enlarge
        </button>
        <a className="btn btn-sm" aria-label={`Save, image ${n}`} href={url} download={downloadName(job, index)}>
          Save
        </a>
      </div>
    </article>
  );
}

function PendingSlot({ n, job, now }: { n: number; job: Job; now: number }) {
  return (
    <div className="well tile-wait" role="status" aria-label={`Image ${n} is ${statusWord(job.status)}`}>
      <span className="tile-no" aria-hidden="true">
        {n}
      </span>
      <span className="wait-word">{statusWord(job.status)}</span>
      <span className="wait-clock">{clock(ageSeconds(job.created_at, now))}</span>
    </div>
  );
}

/* ---------------------------------------------------------- the sheet */

export function SheetPanel({
  job,
  me,
  catalogue,
  count,
  now,
  sourceUrl,
  enlargeBusy,
  actions,
  onRetry,
  onOpenSheet,
}: {
  job: Job;
  me: string;
  catalogue: Intents | null;
  /** How many images were asked for, when the sheet does not say. */
  count: number;
  now: number;
  sourceUrl: string | null;
  enlargeBusy: boolean;
  actions: TileActions;
  onRetry: (job: Job) => void;
  onOpenSheet: (id: number) => void;
}) {
  const word = statusWord(job.status);
  const outputs = job.outputs.filter((o) => o.url);
  const asked = job.intent === "enlarge" ? 1 : (job.num_images ?? count);
  const slots = word === "done" ? Math.max(outputs.length, 1) : Math.max(asked, 1);

  return (
    <section className="sheet-panel" aria-labelledby="sheet-title">
      <div className="sheet-head">
        <h2 className="sheet-title" id="sheet-title">
          Sheet {sheet(job.id)} · {jobTitle(job, catalogue)}
          {word === "done" ? ` · ${outputs.length} ${outputs.length === 1 ? "image" : "images"}` : ""}
        </h2>
        <StatusPill status={job.status} />
        <Owner job={job} me={me} />
        {job.parent_job_id != null && (
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => onOpenSheet(job.parent_job_id!)}>
            Based on sheet {sheet(job.parent_job_id)}
          </button>
        )}
      </div>

      {word === "failed" ? (
        <div className="sheet-failed" role="alert">
          <p>{job.error ?? "This render did not finish."}</p>
          <button type="button" className="btn btn-primary" onClick={() => onRetry(job)}>
            Retry
          </button>
        </div>
      ) : (
        <div className={`grid ${slots === 1 ? "is-single" : ""}`}>
          {Array.from({ length: slots }, (_, i) => {
            const url = word === "done" ? imageUrl(job, i) : null;
            return url ? (
              <Tile
                key={i}
                job={job}
                index={i}
                url={url}
                sourceUrl={sourceUrl}
                actions={actions}
                busy={enlargeBusy}
              />
            ) : (
              <PendingSlot key={i} n={i + 1} job={job} now={now} />
            );
          })}
        </div>
      )}
    </section>
  );
}

/** Before anything has been rendered: the steps left, set like sheet notes. Step 1 goes once a view is in. */
export function EmptySheet({ hasView = false }: { hasView?: boolean }) {
  return (
    <section className="sheet-panel is-empty" aria-label="Results">
      <ol className="steps">
        {!hasView && (
          <li>
            <span className="note-no" aria-hidden="true">
              1
            </span>
            Drop a view.
          </li>
        )}
        <li>
          <span className="note-no" aria-hidden="true">
            2
          </span>
          Say what to do.
        </li>
        <li>
          <span className="note-no" aria-hidden="true">
            3
          </span>
          Render.
        </li>
      </ol>
    </section>
  );
}

/* -------------------------------------------------------------- queue */

export function Queue({
  jobs,
  me,
  catalogue,
  now,
  onRetry,
  onOpenSheet,
}: {
  jobs: Job[];
  me: string;
  catalogue: Intents | null;
  now: number;
  onRetry: (job: Job) => void;
  onOpenSheet: (id: number) => void;
}) {
  if (!jobs.length) return null;
  return (
    <section className="queue" aria-labelledby="queue-title">
      <h2 className="label" id="queue-title">
        Queue
      </h2>
      <ul className="queue-list">
        {jobs.map((j) => {
          const word = statusWord(j.status);
          return (
            <li className="queue-row" key={j.id}>
              <button type="button" className="sheet-no" onClick={() => onOpenSheet(j.id)}>
                {sheet(j.id)}
              </button>
              <span className="queue-what">{jobTitle(j, catalogue)}</span>
              <StatusPill status={j.status} />
              <Owner job={j} me={me} />
              {PENDING.has(j.status) && <span className="queue-clock">{clock(ageSeconds(j.created_at, now))}</span>}
              {word === "failed" && (
                <>
                  <span className="queue-error">{j.error ?? "This render did not finish."}</span>
                  <button type="button" className="btn btn-sm" onClick={() => onRetry(j)}>
                    Retry
                  </button>
                </>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

/* ----------------------------------------------------- earlier sheets */

export function Earlier({
  jobs,
  me,
  currentId,
  catalogue,
  canLoadMore,
  onOpenSheet,
  onMore,
}: {
  jobs: Job[];
  me: string;
  currentId: number | null;
  catalogue: Intents | null;
  canLoadMore: boolean;
  onOpenSheet: (id: number) => void;
  onMore: () => void;
}) {
  if (!jobs.length) return null;
  return (
    <section className="earlier" aria-labelledby="earlier-title">
      <h2 className="label" id="earlier-title">
        Earlier sheets
      </h2>
      <ul className="strip">
        {jobs.map((j) => (
          <li key={j.id}>
            <button
              type="button"
              className="well strip-thumb"
              aria-current={j.id === currentId ? "true" : undefined}
              aria-label={`Sheet ${sheet(j.id)}, ${jobTitle(j, catalogue)}${isMine(j, me) ? "" : `, by ${j.username.split("@")[0]}`}`}
              onClick={() => onOpenSheet(j.id)}
            >
              <Pic src={firstImage(j) ?? ""} alt="" />
              <span className="strip-no">{sheet(j.id)}</span>
              {!isMine(j, me) && (
                <span className="owner-tag is-strip" aria-hidden="true">
                  {initials(j.username)}
                </span>
              )}
            </button>
          </li>
        ))}
      </ul>
      {canLoadMore && (
        <button type="button" className="btn btn-sm" onClick={onMore}>
          Load more
        </button>
      )}
    </section>
  );
}

/* ------------------------------------------------------------ notices */

export function ReadyNote({ ids, onOpen }: { ids: number[]; onOpen: (id: number) => void }) {
  const id = ids[ids.length - 1];
  if (id === undefined) return null;
  return (
    <Note
      kind="good"
      live
      actions={
        <button type="button" className="btn btn-sm" onClick={() => onOpen(id)}>
          Open it
        </button>
      }
    >
      {ids.length === 1 ? `Sheet ${sheet(id)} is ready.` : `${ids.length} sheets are ready, the latest is ${sheet(id)}.`}
    </Note>
  );
}
