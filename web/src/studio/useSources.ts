import { useEffect, useState } from "react";
import { api } from "../api";
import type { Job } from "../api";
import { statusWord } from "./util";

/**
 * The drawing each finished sheet was made from, looked up once per sheet.
 * Hold-to-compare and the lightbox both need it, and the answer never changes.
 */
export function useSources(current: Job | null): Record<number, string | null> {
  const [sources, setSources] = useState<Record<number, string | null>>({});

  useEffect(() => {
    if (!current || statusWord(current.status) !== "done" || current.id in sources) return;
    let live = true;
    api
      .renderInputs(current.id)
      .then((list) => {
        // A mask or a depth pass is an input too, but never the drawing.
        const src = list.find((a) => a.kind !== "mask" && a.kind !== "control") ?? list[0];
        if (live) setSources((s) => ({ ...s, [current.id]: src?.url ?? null }));
      })
      .catch(() => live && setSources((s) => ({ ...s, [current.id]: null })));
    return () => {
      live = false;
    };
  }, [current, sources]);

  return sources;
}
