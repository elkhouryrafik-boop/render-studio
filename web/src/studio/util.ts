import type { Intents, Job } from "../api";

/** Jobs the server has not finished with yet. */
export const PENDING = new Set(["reserved", "queued", "in_progress"]);

export const sheet = (id: number) => String(id).padStart(4, "0");

/**
 * The four status words a person ever sees. The internal values (`reserved`,
 * `in_progress`, ...) are bookkeeping and never reach the screen.
 */
export type StatusWord = "queued" | "rendering" | "done" | "failed";

export function statusWord(status: string): StatusWord {
  if (status === "completed") return "done";
  if (status === "failed" || status === "refused" || status === "cancelled") return "failed";
  if (status === "in_progress") return "rendering";
  return "queued";
}

/** The URL of one image of a render, or null when that image is not on disk. */
export function imageUrl(job: Job, index: number): string | null {
  return job.outputs.find((o) => o.index === index)?.url ?? null;
}

export function firstImage(job: Job): string | null {
  return job.outputs.find((o) => o.url)?.url ?? job.output_url;
}

/** Saved files get a name and an extension; the image route has neither. */
export function downloadName(job: Job, index: number): string {
  const suffix = index > 0 ? `-${index + 1}` : "";
  return `render-${sheet(job.id)}${suffix}.png`;
}

/** The server stamps UTC; some older rows carry no zone marker. */
export function ageSeconds(createdAt: string, now: number): number {
  const stamp = /[zZ]|[+-]\d\d:?\d\d$/.test(createdAt) ? createdAt : `${createdAt.replace(" ", "T")}Z`;
  const then = Date.parse(stamp);
  return Number.isFinite(then) ? Math.max(0, Math.floor((now - then) / 1000)) : 0;
}

export function clock(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function money(usd: number): string {
  return `$${usd.toFixed(2)}`;
}

/** "Make it real · Hold · Daylight", from what the job recorded. */
export function jobTitle(job: Job, catalogue: Intents | null): string {
  if (!job.intent) return "Render";
  const intent = catalogue?.intents.find((i) => i.key === job.intent);
  const parts = [intent?.label ?? "Render"];
  if (typeof job.geometry_hold === "number" && catalogue) {
    const snaps = catalogue.geometry_hold.snaps;
    const nearest = snaps.reduce((a, b) =>
      Math.abs(b.value - job.geometry_hold!) < Math.abs(a.value - job.geometry_hold!) ? b : a,
    );
    parts.push(nearest.label);
  }
  const style = catalogue?.styles.find((s) => s.key === job.style);
  if (style) parts.push(style.label);
  return parts.join(" · ");
}

/** Reduced motion is respected for the one smooth scroll we do. */
export function stillMotion(): boolean {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

/** Only these views are perspective; the others are drawings with a tuned line hold. */
const PERSPECTIVE_VIEWS = new Set(["model", "interior", "aerial"]);
export const isDrawingView = (view: string | null): boolean => view != null && !PERSPECTIVE_VIEWS.has(view);

/** Whose sheet it is: the render's username is the account's email. */
export const isMine = (job: Job, me: string): boolean => job.username.toLowerCase() === me.toLowerCase();

/** "layla.haddad@office" becomes "LH"; "omar@office" becomes "OM". */
export function initials(username: string): string {
  const local = username.split("@")[0] ?? username;
  const parts = local.split(/[^A-Za-z0-9]+/).filter(Boolean);
  const letters = parts.length > 1 ? parts.slice(0, 2).map((p) => p[0]).join("") : (parts[0] ?? "?").slice(0, 2);
  return letters.toUpperCase();
}

export const MATCH_KEY = "match";
const MATCH_WIDTH = 1280;
const MATCH_CAP = 2048;

/**
 * The output size that follows the view: 1280 wide, the view's own proportions,
 * the height rounded to a multiple of 16 and both sides capped at 2048. A size
 * that disagrees with the view's aspect leaks black bars into the render.
 */
export function matchSize(natural: { w: number; h: number } | null): { width: number; height: number } | null {
  if (!natural || natural.w <= 0 || natural.h <= 0) return null;
  let width = MATCH_WIDTH;
  let height = Math.round((MATCH_WIDTH * natural.h) / natural.w / 16) * 16;
  if (height > MATCH_CAP) {
    width = Math.max(16, Math.round((width * MATCH_CAP) / height / 16) * 16);
    height = MATCH_CAP;
  }
  return { width: Math.min(width, MATCH_CAP), height: Math.max(16, height) };
}
