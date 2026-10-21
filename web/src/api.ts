/**
 * The one place that talks to the API.
 *
 * Every call goes through `request()`, so the two behaviours that must never
 * vary — sending the session cookie, and turning a non-2xx into a readable
 * error — are written once.
 */

export type ImageRole =
  | "geometry"
  | "style"
  | "material"
  | "context"
  | "entourage"
  | "mask";

export interface User {
  id: number;
  email: string;
  name: string;
  status: "pending" | "active" | "suspended";
  is_admin: boolean;
  budget_monthly_usd: number | null;
  created_at?: string | null;
  approved_at?: string | null;
  /** Only on `/auth/me`: "fake" means placeholder images and nothing sent out. */
  engine?: "fal" | "fake";
}

export interface Asset {
  id: number;
  kind: string;
  width: number | null;
  height: number | null;
  created_at: string;
  url: string;
}

export interface CadLayer {
  name: string;
  color: string | null;
  visible: boolean;
}

export interface CadMaterial {
  material_class: string;
  confidence: string;
  albedo: [number, number, number];
  share_of_visible_pixels: number;
  source: string;
  tile_width_mm: number | null;
}

export interface CadConfirmResult {
  asset: Asset;
  // True depth pass — IFC and SKP. Sent as control_assets={"depth": id} on
  // the render so the depth conditioning reads real geometry instead of
  // guessing.
  depth_asset: Asset | null;
  // SKP only: the pre-textured material base and its per-pixel confidence
  // map. Attach albedo as the GEOMETRY reference and send
  // control_assets={"depth": depth_asset.id, "change_map": change_asset.id}
  // on the skp_material_render task.
  albedo_asset: Asset | null;
  change_asset: Asset | null;
  camera: string | null;
  warnings: string[];
  materials: CadMaterial[];
}

export const CAD_CAMERAS = ["iso", "front", "top"] as const;
export type CadCamera = (typeof CAD_CAMERAS)[number];

// SKP's own camera vocabulary is a superset of the shared one: "buildings-iso"
// frames on the built mass rather than the whole site slab (a dolly-in
// distinct from anything IFC offers, since IFC has no equivalent "subject").
export const SKP_CAMERAS = [...CAD_CAMERAS, "buildings-iso"] as const;
export type SkpCamera = (typeof SKP_CAMERAS)[number];

/** A background SKP bake is still running — not a failure. `stage` and
 * `elapsedSeconds` are enough to show a progress message; poll the same
 * call again in a few seconds. */
export class CadPendingError extends Error {
  stage: string;
  elapsedSeconds: number;
  // Whole-bake estimate in seconds, 0 if unknown. Lets the progress panel say
  // "about 16 minutes" and show what is left, instead of counting up forever.
  estimatedSeconds: number;
  constructor(
    stage: string,
    elapsedSeconds: number,
    message: string,
    estimatedSeconds = 0,
  ) {
    super(message);
    this.name = "CadPendingError";
    this.stage = stage;
    this.elapsedSeconds = elapsedSeconds;
    this.estimatedSeconds = estimatedSeconds;
  }
}

export interface CadBakePlan {
  cached: boolean;
  estimated_seconds: number;
  message: string;
}

/** One image a render produced. `url` is null when its file is no longer on disk. */
export interface JobOutput {
  index: number;
  url: string | null;
  seed: number | null;
}

export interface Job {
  id: number;
  task: string;
  model_id: string;
  engine: string | null;
  status: string;
  est_usd: number;
  username: string;
  project_id: number | null;
  parent_job_id: number | null;
  root_job_id: number | null;
  created_at: string;
  error: string | null;
  /** Image 0, kept for older readers. Use `outputs` for the full set. */
  output_url: string | null;
  output_seed: number | null;
  outputs: JobOutput[];
  /** Set when this render is a retry of a failed one. */
  retry_of_job_id: number | null;
  /** Where the depth pass came from: "cad", "estimated" or "none". */
  depth_source?: DepthSource | null;
  /** What was asked for, read from the job's params; null on older renders. */
  intent?: string | null;
  view?: string | null;
  style?: string | null;
  line?: string | null;
  geometry_hold?: number | null;
  /** How many images this render makes; absent on older rows. */
  num_images?: number | null;
}

export type DepthSource = "cad" | "estimated" | "none";

export interface IntentView {
  key: string;
  label: string;
}

export interface Intent {
  key: string;
  label: string;
  hint: string;
  /** False for actions on a result (Enlarge); never shown as a card. */
  card: boolean;
  slider: boolean;
  needs_mask: boolean;
  needs_line: boolean;
  max_references: number;
  /** "cad_materials" means: only after a CAD bake attached albedo and change map. */
  requires: string | null;
  views: IntentView[];
  default_view: string | null;
}

export interface Snap {
  key: string;
  value: number;
  label: string;
  hint: string;
}

export interface SizeOption {
  key: string;
  label: string;
  width: number;
  height: number;
}

export interface Intents {
  intents: Intent[];
  styles: { key: string; label: string }[];
  default_style: string;
  geometry_hold: { default: number; snaps: Snap[] };
  sizes: SizeOption[];
  depth_available: boolean;
  num_images_default: number;
  num_images_max: number;
}

/** The fields every intent-driven call carries. */
export interface IntentFields {
  intent: string;
  view?: string | null;
  style?: string | null;
  line?: string | null;
  geometry_hold?: number | null;
}

export interface DepthPreview {
  asset: Asset;
  depth_source: DepthSource;
  seconds: number;
}

export interface Budget {
  month_total_usd: number;
  month_user_usd: number;
  monthly_cap_usd: number;
  per_user_cap_usd: number;
  kill_switch: boolean;
  warning: string | null;
}

/** The runtime switches an administrator can flip. */
export interface AdminControls {
  kill_switch: boolean;
  monthly_cap_usd: number;
}

export interface Estimate {
  est_usd: number;
  over_per_job_cap: boolean;
  model_id: string;
}

export interface SubmitResult {
  job_id: number;
  est_usd: number;
  duplicate: boolean;
  message: string;
  depth_source?: DepthSource | null;
  /** Plain sentences about what was adjusted (for example Exact capped at Hold). */
  notes?: string[] | string | null;
}

/** An API failure carrying the server's own wording, which is written for humans. */
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

/**
 * The server is not reachable, which is a different thing from the server
 * saying no. On a LAN app this is the commonest failure of all, and it used to
 * surface as the least informative message in the app ("Something went wrong")
 * because `fetch` rejects with a bare TypeError that no `instanceof ApiError`
 * check ever caught.
 */
export class NetworkError extends Error {
  constructor(message = "Cannot reach the render server. It may be restarting.") {
    super(message);
    this.name = "NetworkError";
  }
}

type UnauthorizedHandler = () => void;
let onUnauthorized: UnauthorizedHandler | null = null;

/**
 * Called once for any 401 from any endpoint. The session is an httpOnly cookie
 * that can expire, or an admin can suspend an account mid-session; without
 * this the user stays inside the signed-in shell while every single action
 * returns a red "Not authenticated" box.
 */
export function setUnauthorizedHandler(handler: UnauthorizedHandler | null) {
  onUnauthorized = handler;
}

/** Nothing here should ever hang forever with the Render button stuck busy. */
const TIMEOUT_MS = 120_000;

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  return send<T>(path, init);
}

async function send<T>(path: string, init: RequestInit): Promise<T> {
  const timer = new AbortController();
  const bail = setTimeout(() => timer.abort(), TIMEOUT_MS);
  let response: Response;
  try {
    response = await fetch(path, {
      // The session is an httpOnly cookie. Same-origin credentials are what send
      // it; nothing in this file ever handles a token directly.
      credentials: "same-origin",
      signal: timer.signal,
      ...init,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new NetworkError("The render server took too long to answer.");
    }
    throw new NetworkError();
  } finally {
    clearTimeout(bail);
  }

  if (response.status === 401) {
    // `me` is how the app asks "am I signed in?", so its 401 is the normal
    // signed-out case rather than a session that just died under the user.
    if (!path.endsWith("/auth/me")) onUnauthorized?.();
    throw new ApiError(401, "Your session ended. Sign in again.");
  }

  if (response.status === 204) return undefined as T;

  const text = await response.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }

  // 202 is "ok" by HTTP's own rules but means something specific here: a
  // background SKP bake has not reached this file yet. Only the CAD routes
  // ever send it, so this check is safe to apply globally rather than per
  // call site.
  if (response.status === 202) {
    const detail =
      body && typeof body === "object" && "detail" in body
        ? (body as { detail: Record<string, unknown> }).detail
        : {};
    throw new CadPendingError(
      typeof detail.stage === "string" ? detail.stage : "starting",
      typeof detail.elapsed_seconds === "number" ? detail.elapsed_seconds : 0,
      typeof detail.message === "string" ? detail.message : "Still baking.",
      typeof detail.estimated_seconds === "number" ? detail.estimated_seconds : 0,
    );
  }

  if (response.status === 429 && !path.startsWith("/api/auth/")) {
    const wait = Number(response.headers.get("Retry-After"));
    throw new ApiError(
      429,
      Number.isFinite(wait) && wait > 0
        ? `Too many renders in a row. Wait ${Math.ceil(wait)} ${Math.ceil(wait) === 1 ? "second" : "seconds"} and try again.`
        : "Too many renders in a row. Wait a moment and try again.",
    );
  }

  if (!response.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body
        ? plainDetail((body as { detail: unknown }).detail)
        : null;
    throw new ApiError(response.status, detail ?? `Request failed (${response.status}).`);
  }

  // A 200 of HTML on an /api path means the SPA catch-all answered instead of
  // a real route, which is what a frontend/backend path drift looks like. It
  // used to JSON-parse to null and be returned as a typed value, so the drift
  // failed silently somewhere far away from its cause.
  if (text && body === null && path.startsWith("/api/")) {
    throw new ApiError(
      response.status,
      `The server did not recognise ${path}. The web bundle and the API are out of step; rebuild the front end.`,
    );
  }
  return body as T;
}

/**
 * The sentence to show a person for any thrown value. Every message the API
 * itself produces is already written for humans, so those pass straight
 * through; everything else gets something better than "Something went wrong."
 */
/**
 * The sentence in an error response's `detail`. The server sends a string
 * everywhere it can; a pydantic list (an answer from before the 422 handler,
 * or one it did not catch) is reduced to its first message rather than shown
 * as "[object Object]". Anything else is no sentence at all.
 */
function plainDetail(detail: unknown): string | null {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const first = detail.find(
      (item): item is { msg: string } =>
        !!item && typeof item === "object" && typeof (item as { msg?: unknown }).msg === "string",
    );
    if (first) return first.msg;
  }
  return null;
}

export function describeError(e: unknown): string {
  if (e instanceof ApiError || e instanceof NetworkError) return e.message;
  if (e instanceof CadPendingError) return e.message;
  if (e instanceof Error && e.message) return e.message;
  return "Something went wrong. Try again, and tell the office admin if it keeps happening.";
}

function json<T>(path: string, method: string, payload?: unknown): Promise<T> {
  return request<T>(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
}

/** A job as the administrator's Activity view returns it. */
export interface ActivityJob extends Job {
  user_id: number | null;
  user_name: string | null;
  actual_usd: number | null;
  updated_at: string;
  timed_out_at: string | null;
}

/** One person's totals; user_id is null for jobs from before accounts existed. */
export interface ActivityUser {
  user_id: number | null;
  email: string;
  name: string;
  status: string;
  jobs_total: number;
  jobs_month: number;
  spent_month_usd: number;
  last_active: string | null;
}

export interface ActivityOut {
  jobs: ActivityJob[];
  users: ActivityUser[];
  /** Send as before_id for the next page; null on the last page. */
  next_before_id: number | null;
  month_total_usd: number;
}

export const api = {
  // --- auth
  me: () => request<User>("/api/auth/me"),
  checkAccessCode: (access_code: string) =>
    json<void>("/api/auth/access-code", "POST", { access_code }),
  signup: (body: {
    access_code: string;
    email: string;
    name: string;
    password: string;
  }) => json<{ needs_approval: boolean; message: string }>("/api/auth/signup", "POST", body),
  login: (email: string, password: string) =>
    json<User>("/api/auth/login", "POST", { email, password }),
  /** Public. Only a standing is returned, never anything else about the account. */
  approval: (email: string) =>
    request<{ status: "pending" | "active" | "unknown" }>(
      `/api/auth/approval?email=${encodeURIComponent(email)}`,
    ),
  logout: () => json<void>("/api/auth/logout", "POST"),
  changePassword: (current_password: string, new_password: string) =>
    json<void>("/api/auth/password", "POST", { current_password, new_password }),

  // --- catalogue
  intents: () => request<Intents>("/api/intents"),
  estimate: (
    body: Partial<IntentFields> & {
      task?: string;
      engine?: string | null;
      width: number;
      height: number;
      num_images: number;
    },
  ) => json<Estimate>("/api/estimate", "POST", body),
  budget: () => request<Budget>("/api/budget"),

  // --- assets
  upload: async (file: File): Promise<Asset> => {
    const form = new FormData();
    form.append("file", file);
    return request<Asset>("/api/assets", { method: "POST", body: form });
  },
  uploadMask: async (blob: Blob, sourceAssetId: number): Promise<Asset> => {
    const form = new FormData();
    form.append("file", new File([blob], "mask.png", { type: "image/png" }));
    form.append("source_asset_id", String(sourceAssetId));
    return request<Asset>("/api/assets/mask", { method: "POST", body: form });
  },

  // --- CAD (DXF/SKP → image)
  //
  // fal.ai reads pixels only, so a DXF or SKP is stored, previewed and
  // adjusted here before it ever becomes a reference — `cadConfirm` is the
  // step that turns it into a normal `upload` asset the rest of the app
  // already knows how to attach.
  cadUpload: async (file: File): Promise<Asset> => {
    const form = new FormData();
    form.append("file", file);
    return request<Asset>("/api/cad", { method: "POST", body: form });
  },
  cadLayers: (assetId: number) => request<CadLayer[]>(`/api/cad/${assetId}/layers`),
  // How long this file will take to convert, asked BEFORE committing to it.
  cadBakePlan: (assetId: number) =>
    request<CadBakePlan>(`/api/cad/${assetId}/bake-plan`),
  // Not a fetch — the preview (SVG for DXF, PNG for IFC) is meant to sit
  // directly in an <img src>, the same way Asset.url already does. The
  // browser sends the session cookie with an image request same-origin, so no
  // token handling belongs here.
  cadPreviewUrl: (assetId: number, layers?: string[], camera?: SkpCamera) => {
    const params = new URLSearchParams();
    if (layers && layers.length) params.set("layers", layers.join(","));
    if (camera) params.set("camera", camera);
    const qs = params.toString();
    return qs ? `/api/cad/${assetId}/preview?${qs}` : `/api/cad/${assetId}/preview`;
  },
  cadConfirm: (
    assetId: number,
    body: {
      layers?: string[] | null;
      line_weight_scale?: number;
      camera?: SkpCamera;
      project_id?: number | null;
    },
  ) => json<CadConfirmResult>(`/api/cad/${assetId}/confirm`, "POST", body),

  // --- renders
  submit: (
    body: IntentFields & {
      prompt?: string;
      references: { asset_id: number; role: ImageRole }[];
      width: number;
      height: number;
      num_images: number;
      mask_asset_id?: number | null;
      control_assets?: Record<string, number> | null;
    },
  ) => json<SubmitResult>("/api/renders", "POST", body),
  continueFrom: (
    jobId: number,
    body: IntentFields & {
      prompt?: string;
      extra_references: { asset_id: number; role: ImageRole }[];
      width: number;
      height: number;
      num_images: number;
      mask_asset_id?: number | null;
      /** Which image of the parent render to continue from; 0 is the first. */
      output_index?: number;
    },
  ) => json<SubmitResult>(`/api/renders/${jobId}/continue`, "POST", body),
  /** Estimate the depth pass for a view; the result is an overlay image. */
  depthPreview: (assetId: number) =>
    json<DepthPreview>(`/api/assets/${assetId}/depth`, "POST"),
  renders: (limit = 60) => request<Job[]>(`/api/renders?limit=${limit}`),
  render: (id: number) => request<Job>(`/api/renders/${id}`),
  renderInputs: (id: number) => request<Asset[]>(`/api/renders/${id}/inputs`),
  /** Send a failed render again. Only its owner or an administrator may. */
  retry: (id: number) => json<SubmitResult>(`/api/renders/${id}/retry`, "POST"),
  asReference: (id: number, outputIndex = 0) =>
    json<Asset>(`/api/renders/${id}/as-reference?output_index=${outputIndex}`, "POST"),
  refresh: () => json<Record<string, string>>("/api/renders/refresh", "POST"),

  // --- admin
  adminControls: () => request<AdminControls>("/api/admin/controls"),
  setKillSwitch: (on: boolean) =>
    json<AdminControls>("/api/admin/controls/kill-switch", "PUT", { on }),
  adminActivity: (params: {
    limit?: number;
    before_id?: number;
    user_id?: number;
    status?: string;
  }) => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
    }
    const qs = q.toString();
    return request<ActivityOut>(`/api/admin/activity${qs ? `?${qs}` : ""}`);
  },
  adminUsers: () => request<User[]>("/api/admin/users"),
  adminPending: () => request<User[]>("/api/admin/users/pending"),
  approve: (id: number) => json<User>(`/api/admin/users/${id}/approve`, "POST"),
  suspend: (id: number) => json<User>(`/api/admin/users/${id}/suspend`, "POST"),
  reinstate: (id: number) => json<User>(`/api/admin/users/${id}/reinstate`, "POST"),
  setBudget: (id: number, budget_monthly_usd: number | null) =>
    json<User>(`/api/admin/users/${id}/budget`, "PUT", { budget_monthly_usd }),
  grantAdmin: (id: number) => json<User>(`/api/admin/users/${id}/admin`, "POST"),
  revokeAdmin: (id: number) => json<User>(`/api/admin/users/${id}/admin`, "DELETE"),
  spend: () =>
    request<{
      month_total_usd: number;
      monthly_cap_usd: number;
      kill_switch: boolean;
      per_user: {
        user_id: number;
        email: string;
        name: string;
        status: string;
        spent_usd: number;
        cap_usd: number;
      }[];
    }>("/api/admin/spend"),
};
