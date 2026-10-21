import { useCallback, useEffect, useRef, useState } from "react";
import { api, describeError, type ActivityJob, type ActivityUser } from "./api";
import { Empty, Note, Section } from "./components";
import { RadioGroup } from "./studio/Radio";
import { initials } from "./studio/util";

// The first page is re-read this often while the tab is on screen.
const REFRESH_MS = 30_000;
const PAGE = 100;

const STATUSES: { value: string; label: string }[] = [
  { value: "", label: "All" },
  { value: "reserved", label: "Reserved" },
  { value: "queued", label: "Queued" },
  { value: "in_progress", label: "Running" },
  { value: "completed", label: "Completed" },
  { value: "failed", label: "Failed" },
  { value: "refused", label: "Refused" },
  { value: "cancelled", label: "Cancelled" },
];

/** "4 min ago". The exact time goes in a title attribute beside it. */
export function relativeTime(iso: string, now: number = Date.now()): string {
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return iso;
  const s = Math.max(0, Math.round((now - then) / 1000));
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} ${h === 1 ? "hour" : "hours"} ago`;
  const d = Math.round(h / 24);
  if (d < 30) return `${d} ${d === 1 ? "day" : "days"} ago`;
  return new Date(then).toISOString().slice(0, 10);
}

const usd = (n: number) => `$${n.toFixed(2)}`;

function matches(job: ActivityJob, needle: string): boolean {
  if (!needle) return true;
  const hay = [job.line, job.view, job.style, job.task, job.username, job.user_name, job.intent]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return hay.includes(needle.toLowerCase());
}

/**
 * Everything every user has generated: who, when, what they asked for, what it
 * cost and how it ended. Administrators only; the route enforces that too.
 */
export function Activity() {
  const [jobs, setJobs] = useState<ActivityJob[]>([]);
  const [users, setUsers] = useState<ActivityUser[]>([]);
  const [monthTotal, setMonthTotal] = useState(0);
  const [nextBefore, setNextBefore] = useState<number | null>(null);
  const [userId, setUserId] = useState<number | null>(null);
  const [status, setStatus] = useState("");
  const [text, setText] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());

  // A response for filters the administrator has since changed must not land.
  const generation = useRef(0);
  const filters = useRef({ userId, status });
  filters.current = { userId, status };

  const query = (before_id?: number) => ({
    limit: PAGE,
    before_id,
    user_id: filters.current.userId ?? undefined,
    status: filters.current.status || undefined,
  });

  // The first page for new filters replaces everything on screen.
  const loadFirst = useCallback(() => {
    const mine = ++generation.current;
    setJobs([]);
    setNextBefore(null);
    setLoaded(false);
    api
      .adminActivity(query())
      .then((r) => {
        if (mine !== generation.current) return;
        setJobs(r.jobs);
        setUsers(r.users);
        setMonthTotal(r.month_total_usd);
        setNextBefore(r.next_before_id);
        setNow(Date.now());
        setError(null);
      })
      .catch((e) => {
        if (mine === generation.current) setError(describeError(e));
      })
      .finally(() => {
        if (mine === generation.current) setLoaded(true);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // The periodic read merges the first page into what is already shown, so
  // pages added with "Load more" stay put and nextBefore is left alone.
  const refresh = useCallback(() => {
    const mine = generation.current;
    api
      .adminActivity(query())
      .then((r) => {
        if (mine !== generation.current) return;
        setJobs((cur) => {
          // Inside the new page's window the server is the truth: a row that
          // is no longer returned has left the filter (or been removed), so it
          // goes. Rows older than the window are kept as they were.
          const oldest = r.jobs.length ? Math.min(...r.jobs.map((j) => j.id)) : 0;
          const kept = cur.filter((j) => j.id < oldest);
          return [...r.jobs, ...kept].sort((x, y) => y.id - x.id);
        });
        setUsers(r.users);
        setMonthTotal(r.month_total_usd);
        setNow(Date.now());
        setError(null);
      })
      .catch((e) => {
        if (mine === generation.current) setError(describeError(e));
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    loadFirst();
  }, [loadFirst, userId, status]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") refresh();
    }, REFRESH_MS);
    const onVisible = () => {
      if (document.visibilityState === "visible") refresh();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [refresh]);

  const loadMore = () => {
    if (nextBefore === null) return;
    const mine = generation.current;
    setLoadingMore(true);
    api
      .adminActivity(query(nextBefore))
      .then((r) => {
        if (mine !== generation.current) return;
        setJobs((cur) => {
          const seen = new Set(cur.map((j) => j.id));
          return [...cur, ...r.jobs.filter((j) => !seen.has(j.id))];
        });
        setNextBefore(r.next_before_id);
        setError(null);
      })
      .catch((e) => {
        if (mine === generation.current) setError(describeError(e));
      })
      .finally(() => setLoadingMore(false));
  };

  const shown = jobs.filter((j) => matches(j, text));
  const monthJobs = users.reduce((n, u) => n + u.jobs_month, 0);

  return (
    <div className="stack">
      <section className="panel">
        <Section title="This month" note="office" />
        <p>
          This month: <strong>{usd(monthTotal)}</strong> across {monthJobs}{" "}
          {monthJobs === 1 ? "render" : "renders"}
        </p>
        <RadioGroup
          label="Show one person"
          value={userId === null ? "all" : String(userId)}
          items={[
            { value: "all", label: "Everyone" },
            ...users
              .filter((u) => u.user_id !== null)
              .map((u) => ({
                value: String(u.user_id),
                label: `${u.name || u.email} · ${u.jobs_month} · ${usd(u.spent_month_usd)}`,
              })),
          ]}
          onPick={(v) => setUserId(v === "all" ? null : Number(v))}
          className="chips activity-chips"
          itemClassName="chip"
        />
      </section>

      <section className="panel">
        <Section title="Activity" note={`${shown.length} shown`} />
        <div className="activity-filters">
          <label>
            <span className="caption">Status</span>
            <select value={status} onChange={(e) => setStatus(e.target.value)}>
              {STATUSES.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span className="caption">Search the loaded rows</span>
            <input
              type="search"
              value={text}
              placeholder="prompt, view, style, task, person"
              onChange={(e) => setText(e.target.value)}
            />
          </label>
        </div>

        {error && (
          <Note kind="error" live>
            {error}
          </Note>
        )}
        {!loaded && !error && <p className="caption">Reading activity</p>}

        {loaded && shown.length === 0 && !error && (
          <Empty title="Nothing here">No renders match these filters.</Empty>
        )}

        <ul className="activity-list">
          {shown.map((j) => (
            <JobRow key={j.id} job={j} now={now} />
          ))}
        </ul>

        {nextBefore !== null && (
          <button type="button" className="btn" disabled={loadingMore} onClick={loadMore}>
            {loadingMore ? "Loading" : "Load more"}
          </button>
        )}
      </section>
    </div>
  );
}

function JobRow({ job, now }: { job: ActivityJob; now: number }) {
  const who = job.user_name || job.username;
  const failed = Boolean(job.error || job.timed_out_at);
  const images = job.outputs.filter((o) => o.url);
  const what = job.intent || job.task;
  return (
    <li className="activity-row">
      <div className="activity-when" title={job.created_at}>
        {relativeTime(job.created_at, now)}
      </div>
      <div className="activity-who">
        <span className="owner-tag" aria-hidden="true">
          {initials(job.username)}
        </span>
        <span>
          {who}
          <small className="caption activity-email">{job.username}</small>
        </span>
      </div>
      <div className="activity-what">
        <div>
          <strong>{what}</strong>
          {job.view ? ` · ${job.view}` : ""}
          {job.style ? ` · ${job.style}` : ""}
          {job.geometry_hold != null ? ` · Hold ${job.geometry_hold}` : ""}
        </div>
        {job.line && <p className="activity-line">{job.line}</p>}
        {failed && (
          <p className="activity-error">
            {job.error ||
              (job.timed_out_at ? `Timed out at ${job.timed_out_at}.` : "This render did not finish.")}
          </p>
        )}
      </div>
      <div className="activity-meta">
        <span className={`status ${job.status}`}>{job.status.replace("_", " ")}</span>
        <span className="caption">
          est {usd(job.est_usd)}
          {job.actual_usd !== null ? `, actual ${usd(job.actual_usd)}` : ""}
        </span>
        {job.depth_source && <span className="chip activity-depth">{job.depth_source}</span>}
      </div>
      {images.length > 0 && (
        <div className="activity-thumbs">
          {images.map((o) => (
            <a
              key={o.index}
              href={o.url!}
              target="_blank"
              rel="noreferrer"
              aria-label={`Open image ${o.index + 1} of render ${job.id} at full size`}
            >
              <img src={o.url!} alt="" loading="lazy" />
            </a>
          ))}
        </div>
      )}
    </li>
  );
}
