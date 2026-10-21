import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { Job } from "../api";
import { PENDING, isMine } from "./util";

/** While one of the signed-in user's own renders is in flight, look often; otherwise now and then. */
const POLL_ACTIVE_MS = 2500;
const POLL_IDLE_MS = 6000;
export const PAGE = 60;

/**
 * The office's renders, newest first, kept fresh: every 2.5 s while one of the
 * signed-in person's own is in flight, every 6 s otherwise.
 *
 * The poll callback lives in a ref: listing `jobs` in the effect's deps would
 * rebuild the timer on every response and drift the cadence to 6s + latency.
 * `offline` flips only after two failed reads in a row, so one dropped request
 * does not put a line on the screen.
 */
export function useJobs(me: string) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [limit, setLimit] = useState(PAGE);
  const [loaded, setLoaded] = useState(false);
  const [offline, setOffline] = useState(false);
  const seq = useRef(0);
  const misses = useRef(0);

  const load = useCallback(async () => {
    const mine = ++seq.current;
    try {
      const list = await api.renders(limit);
      if (mine !== seq.current) return;
      setJobs(list);
      misses.current = 0;
      setOffline(false);
    } catch {
      if (mine !== seq.current) return;
      misses.current += 1;
      if (misses.current >= 2) setOffline(true);
    } finally {
      if (mine === seq.current) setLoaded(true);
    }
  }, [limit]);

  useEffect(() => {
    void load();
  }, [load]);

  const loadRef = useRef(load);
  useEffect(() => {
    loadRef.current = load;
  }, [load]);

  const anyPending = jobs.some((j) => PENDING.has(j.status));
  const minePending = jobs.some((j) => PENDING.has(j.status) && isMine(j, me));
  useEffect(() => {
    const timer = window.setInterval(
      () => {
        // Asking the engine for news is only worth it while something is in flight.
        (anyPending ? api.refresh().catch(() => undefined) : Promise.resolve()).then(() => loadRef.current());
      },
      minePending ? POLL_ACTIVE_MS : POLL_IDLE_MS,
    );
    return () => window.clearInterval(timer);
  }, [anyPending, minePending]);

  return {
    jobs,
    loaded,
    offline,
    anyPending,
    reload: load,
    canLoadMore: jobs.length >= limit,
    loadMore: () => setLimit((n) => n + PAGE),
  };
}
