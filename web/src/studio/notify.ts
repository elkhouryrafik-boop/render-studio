import { useEffect, useRef, useState } from "react";
import type { Intents, Job } from "../api";
import { PENDING, jobTitle, sheet } from "./util";

/**
 * Permission is asked once, on the first Render click: the one moment when a
 * person has just said they will be waiting. Asking on page load is how a
 * browser learns to block a site.
 */
export function askToNotifyOnce() {
  try {
    if ("Notification" in window && Notification.permission === "default") {
      void Notification.requestPermission();
    }
  } catch {
    /* A browser without the API simply never notifies. */
  }
}

export function notifyReady(id: number, what: string) {
  try {
    if ("Notification" in window && Notification.permission === "granted") {
      new Notification(`Sheet ${sheet(id)} is ready`, { body: what });
    }
  } catch {
    /* Some mobile browsers throw on the constructor. The in-app note remains. */
  }
}

/** "(1 ready) Render" in the tab while finished sheets are waiting to be seen. */
export function useTabBadge(ready: number) {
  useEffect(() => {
    const original = document.title;
    return () => {
      document.title = original;
    };
  }, []);
  useEffect(() => {
    if (ready > 0) document.title = `(${ready} ready) Render`;
    else document.title = "Render Studio";
  }, [ready]);
}

/**
 * Sheets that finished while the person was looking elsewhere: another sheet
 * in front of them, or another tab. Each one gets a note on the page, a badge
 * in the tab title and, if permission was granted, a system notification.
 * Looking at the sheet is what clears it.
 */
export function useReady(
  jobs: Job[],
  currentId: number | null,
  catalogue: Intents | null,
): [number[], React.Dispatch<React.SetStateAction<number[]>>] {
  const [ready, setReady] = useState<number[]>([]);
  const seen = useRef<Map<number, string>>(new Map());
  const current = useRef<number | null>(null);

  useEffect(() => {
    current.current = currentId;
  }, [currentId]);

  useEffect(() => {
    for (const j of jobs) {
      const before = seen.current.get(j.id);
      if (before && PENDING.has(before) && j.status === "completed") {
        if (j.id !== current.current || document.hidden) {
          setReady((r) => (r.includes(j.id) ? r : [...r, j.id]));
          notifyReady(j.id, jobTitle(j, catalogue));
        }
      }
      seen.current.set(j.id, j.status);
    }
  }, [jobs, catalogue]);

  useEffect(() => {
    const clear = () => {
      if (!document.hidden && current.current != null) {
        setReady((r) => r.filter((x) => x !== current.current));
      }
    };
    clear();
    document.addEventListener("visibilitychange", clear);
    return () => document.removeEventListener("visibilitychange", clear);
  }, [currentId]);

  useTabBadge(ready.length);
  return [ready, setReady];
}
