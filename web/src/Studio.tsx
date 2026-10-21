import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import {
  api,
  describeError,
  type Asset,
  type Budget,
  type Intents,
  type Job,
} from "./api";
import { IMAGE_ACCEPT, MaskCanvas, Note, pickImages } from "./components";
import { CadDialog } from "./studio/CadDialog";
import { Footer, GeometryNote, IntentNote, LookNote, ViewNote } from "./studio/Controls";
import { Lightbox } from "./studio/Lightbox";
import { askToNotifyOnce, useReady } from "./studio/notify";
import { Earlier, EmptySheet, Queue, ReadyNote, SheetPanel } from "./studio/Results";
import type { Base } from "./studio/types";
import { useDepth } from "./studio/useDepth";
import { useJobs } from "./studio/useJobs";
import { useSources } from "./studio/useSources";
import { usePageDrop } from "./studio/usePageDrop";
import { MATCH_KEY, imageUrl, isDrawingView, isMine, matchSize, statusWord, stillMotion } from "./studio/util";

type Message = { kind: "error" | "good" | "warn" | "blue"; text: string };

const DAY_MS = 24 * 60 * 60 * 1000;
const NARROW = "(max-width: 899px)";

/** Typing in a field must never be read as a shortcut. */
function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  return el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName);
}

/**
 * The studio: one screen, four numbered notes on the left, the sheet on the
 * right. A view goes in (dropped, pasted or chosen), the person says what to do,
 * and four images come back to be judged against the drawing.
 */
export function Studio({ onSpent, budget, me }: { onSpent: () => void; budget: Budget | null; me: string }) {
  const { jobs, loaded, offline, anyPending, reload, canLoadMore, loadMore } = useJobs(me);

  const [catalogue, setCatalogue] = useState<Intents | null>(null);
  const [catalogueFailed, setCatalogueFailed] = useState<string | null>(null);

  // The setup.
  const [base, setBase] = useState<Base | null>(null);
  const [intentKey, setIntentKey] = useState<string | null>(null);
  const [viewKind, setViewKind] = useState<string | null>(null);
  const [hold, setHold] = useState(50);
  const [style, setStyle] = useState("daylight");
  const [line, setLine] = useState("");
  const [count, setCount] = useState(4);
  const [sizeKey, setSizeKey] = useState(MATCH_KEY);
  const [natural, setNatural] = useState<{ url: string; w: number; h: number } | null>(null);
  const [refs, setRefs] = useState<Asset[]>([]);
  const [maskAsset, setMaskAsset] = useState<Asset | null>(null);
  const [marking, setMarking] = useState(false);

  // The page.
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadingRef, setUploadingRef] = useState(false);
  const [enlarging, setEnlarging] = useState(false);
  const [cadOpen, setCadOpen] = useState(false);
  const [message, setMessage] = useState<Message | null>(null);
  const [estimate, setEstimate] = useState<{ usd: number; overJob: boolean } | null>(null);
  const [currentId, setCurrentId] = useState<number | null>(null);
  const [extra, setExtra] = useState<Job | null>(null);
  const [lightbox, setLightbox] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [tried, setTried] = useState(false);
  const [expected, setExpected] = useState<Record<number, number>>({});

  const fileInput = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const assetOfSheet = useRef<Map<string, Asset>>(new Map());

  /* ------------------------------------------------------------ catalogue */

  const loadCatalogue = useCallback(() => {
    setCatalogueFailed(null);
    api
      .intents()
      .then((c) => {
        setCatalogue(c);
        setIntentKey((k) => k ?? c.intents.find((i) => i.card)?.key ?? null);
        setViewKind((v) => v ?? c.intents.find((i) => i.default_view)?.default_view ?? null);
        setHold(c.geometry_hold.default);
        setStyle(c.default_style);
        setCount(c.num_images_default);
      })
      .catch((e) => setCatalogueFailed(describeError(e)));
  }, []);
  useEffect(loadCatalogue, [loadCatalogue]);

  const intent = catalogue?.intents.find((i) => i.key === intentKey) ?? null;
  // A region is one image: the count is not offered and not sent.
  const imageCount = intent?.needs_mask ? 1 : count;
  const hasMaterials = base?.kind === "asset" && Boolean(base.albedoAssetId && base.changeAssetId);
  const cards = useMemo(
    () =>
      (catalogue?.intents ?? []).filter(
        (i) => i.card && (i.requires !== "cad_materials" || hasMaterials),
      ),
    [catalogue, hasMaterials],
  );

  // An intent the view cannot do any more is dropped back to the first card.
  useEffect(() => {
    if (!catalogue || !intentKey) return;
    if (!cards.some((c) => c.key === intentKey)) setIntentKey(cards[0]?.key ?? null);
  }, [catalogue, cards, intentKey]);

  /* ----------------------------------------------------------------- jobs */

  const byId = useMemo(() => new Map(jobs.map((j) => [j.id, j])), [jobs]);
  const current = (currentId != null ? (byId.get(currentId) ?? (extra?.id === currentId ? extra : null)) : null) ?? null;

  // Open on the person's own newest finished sheet, then their newest of any
  // kind; only a person with none of their own opens on the office's newest.
  useEffect(() => {
    if (currentId != null || !loaded) return;
    const done = (j: Job) => statusWord(j.status) === "done";
    const mine = jobs.filter((j) => isMine(j, me));
    const first = mine.find(done) ?? mine[0] ?? jobs.find(done) ?? jobs[0];
    if (first) setCurrentId(first.id);
  }, [currentId, loaded, jobs, me]);

  const openSheet = useCallback((id: number) => {
    setCurrentId(id);
    setReady((r) => r.filter((x) => x !== id));
    setLightbox(null);
  }, []);

  // A sheet older than the page of results is fetched on its own.
  useEffect(() => {
    if (currentId == null || byId.has(currentId) || extra?.id === currentId) return;
    let live = true;
    api
      .render(currentId)
      .then((j) => live && setExtra(j))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [currentId, byId, extra]);

  const [ready, setReady] = useReady(jobs, currentId, catalogue);

  useEffect(() => {
    if (!anyPending) return;
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [anyPending]);

  const sources = useSources(current);

  const queueJobs = useMemo(() => {
    const retried = new Set(jobs.map((j) => j.retry_of_job_id).filter((x): x is number => x != null));
    return jobs.filter((j) => {
      if (j.id === currentId) return false;
      const word = statusWord(j.status);
      if (word === "queued" || word === "rendering") return true;
      return word === "failed" && !retried.has(j.id) && now - Date.parse(j.created_at) < DAY_MS;
    });
  }, [jobs, currentId, now]);

  const earlier = useMemo(
    () => jobs.filter((j) => j.id !== currentId && statusWord(j.status) === "done" && j.outputs.some((o) => o.url)),
    [jobs, currentId],
  );

  /* ---------------------------------------------------------------- views */

  const thumbUrl = base ? (base.kind === "asset" ? base.asset.url : imageUrl(base.job, base.index)) : null;

  // The view's own size: an uploaded asset says it; otherwise the image is measured.
  const knownW = base?.kind === "asset" ? base.asset.width : null;
  const knownH = base?.kind === "asset" ? base.asset.height : null;
  useEffect(() => {
    if (!thumbUrl || (knownW && knownH)) return;
    let live = true;
    const img = new Image();
    img.onload = () => live && setNatural({ url: thumbUrl, w: img.naturalWidth, h: img.naturalHeight });
    img.src = thumbUrl;
    return () => {
      live = false;
    };
  }, [thumbUrl, knownW, knownH]);
  const viewDims =
    knownW && knownH ? { w: knownW, h: knownH } : natural && natural.url === thumbUrl ? natural : null;
  const matched = useMemo(() => matchSize(viewDims), [viewDims?.w, viewDims?.h]);

  /** "Match the view" first when there is a view, then the fixed sizes. */
  const sizes = useMemo(() => {
    const fixed = catalogue?.sizes ?? [];
    return matched
      ? [
          { key: MATCH_KEY, label: `Match the view \u00b7 ${matched.width} \u00d7 ${matched.height}`, ...matched },
          ...fixed,
        ]
      : fixed;
  }, [catalogue, matched]);
  const size = sizes.find((x) => x.key === sizeKey) ?? sizes[0] ?? null;

  /** A sheet's image becomes an asset the first time something needs one. */
  const assetFor = useCallback(async (b: Base): Promise<Asset> => {
    if (b.kind === "asset") return b.asset;
    const key = `${b.job.id}:${b.index}`;
    const cached = assetOfSheet.current.get(key);
    if (cached) return cached;
    const asset = await api.asReference(b.job.id, b.index);
    assetOfSheet.current.set(key, asset);
    return asset;
  }, []);

  const resetForView = () => {
    setSizeKey(MATCH_KEY);
    setMaskAsset(null);
    setMarking(false);
    depth.reset();
    setMessage(null);
  };

  const takeView = useCallback(
    async (file: File) => {
      setUploading(true);
      setMessage(null);
      try {
        const asset = await api.upload(file);
        resetForView();
        setBase({ kind: "asset", asset });
      } catch (e) {
        setMessage({ kind: "error", text: describeError(e) });
      } finally {
        setUploading(false);
      }
    },
    [],
  );

  const takeFiles = useCallback(
    (files: FileList | File[]) => {
      const { ok, rejected } = pickImages(files);
      const first = ok[0];
      if (!first) {
        setMessage({
          kind: "error",
          text: rejected.length
            ? "That is not a picture. Drop a PNG, JPEG or WebP, or use Convert for a DXF, IFC or SKP."
            : "Nothing to use there. Drop a PNG, JPEG or WebP.",
        });
        return;
      }
      void takeView(first);
    },
    [takeView],
  );

  const dragging = usePageDrop(takeFiles);

  // Keys 1 to 4 open that image of the sheet.
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || isTyping(e.target) || cadOpen || lightbox != null) return;
      const n = Number(e.key);
      if (n >= 1 && n <= 4 && current && imageUrl(current, n - 1)) {
        e.preventDefault();
        setLightbox(n - 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [current, cadOpen, lightbox]);

  /* ---------------------------------------------------------------- depth */

  const depth = useDepth(base, assetFor, catalogue?.depth_available ?? false, (text) =>
    setMessage({ kind: "warn", text }),
  );
  const { chip, exactOk } = depth;

  /* ------------------------------------------------------------- estimate */

  useEffect(() => {
    if (!intentKey || !size) return;
    let live = true;
    api
      .estimate({
        intent: intentKey,
        view: intent?.views.length ? viewKind : undefined,
        width: size.width,
        height: size.height,
        num_images: imageCount,
      })
      .then((r) => live && setEstimate({ usd: r.est_usd, overJob: r.over_per_job_cap }))
      .catch(() => live && setEstimate(null));
    return () => {
      live = false;
    };
  }, [intentKey, intent, viewKind, size, imageCount]);

  const remaining = budget ? Math.max(0, budget.per_user_cap_usd - budget.month_user_usd) : null;
  const overBy = estimate && remaining != null ? Math.round((estimate.usd - remaining) * 100) / 100 : null;

  /* --------------------------------------------------------------- submit */

  const reason: string | null = !catalogue
    ? "The options are still loading."
    : !base
      ? "The view is missing. Drop a view first."
      : uploading
        ? "The view is still uploading."
        : !intent
          ? "Choose what to do."
          : intent.needs_mask && !maskAsset
            ? "Mark the region first."
            : intent.needs_line && !line.trim()
              ? intent.needs_mask
                ? "Say what the marked part becomes."
                : "Say what to change."
              : estimate?.overJob
                ? "One render may not cost this much. Choose 1 image."
                : null;

  const drawing = Boolean(intent?.slider && isDrawingView(viewKind));
  const sendsHold = Boolean(intent?.slider && !drawing);

  // The sentence goes away as soon as there is nothing left to say.
  useEffect(() => {
    if (!reason) setTried(false);
  }, [reason]);

  const refresh = async (id: number, n: number) => {
    setExpected((m) => ({ ...m, [id]: n }));
    await reload();
    openSheet(id);
    onSpent();
  };

  const submit = async () => {
    if (busy) return;
    if (reason) {
      setTried(true);
      return;
    }
    if (!catalogue || !intent || !base || !size) return;
    setBusy(true);
    setMessage(null);
    askToNotifyOnce();
    try {
      const fields = {
        intent: intent.key,
        view: intent.views.length ? viewKind : undefined,
        style,
        line: line.trim() || undefined,
        geometry_hold: sendsHold ? hold : undefined,
        width: size.width,
        height: size.height,
        num_images: imageCount,
        mask_asset_id: intent.needs_mask ? maskAsset?.id : undefined,
      };
      const styleRefs = intent.max_references ? refs.map((r) => ({ asset_id: r.id, role: "style" as const })) : [];

      let result;
      if (base.kind === "sheet") {
        result = await api.continueFrom(base.job.id, {
          ...fields,
          extra_references: styleRefs,
          output_index: base.index,
        });
      } else {
        // A SketchUp bake hands over a materials base and a change map; the
        // render reads those in place of the clay view.
        let geometry = base.asset.id;
        let control: Record<string, number> | null = null;
        if (intent.key === "material" && base.albedoAssetId && base.changeAssetId) {
          geometry = base.albedoAssetId;
          control = { change_map: base.changeAssetId };
          if (base.depthAssetId) control.depth = base.depthAssetId;
        } else if (sendsHold && base.depthAssetId) {
          control = { depth: base.depthAssetId };
        }
        result = await api.submit({
          ...fields,
          references: [{ asset_id: geometry, role: "geometry" }, ...styleRefs],
          control_assets: control,
        });
      }

      if (result.depth_source && sendsHold) depth.observe(result.depth_source);
      if (result.duplicate) {
        setMessage({
          kind: "blue",
          text: "This matches a sheet you already sent, so nothing extra was charged.",
        });
      } else {
        const notes = [result.notes ?? []].flat().filter(Boolean).join(" ");
        if (notes) setMessage({ kind: "warn", text: notes });
      }
      setMaskAsset(null);
      await refresh(result.job_id, imageCount);
    } catch (e) {
      setMessage({ kind: "error", text: describeError(e) });
    } finally {
      setBusy(false);
    }
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submit();
  };

  // Enter renders from anywhere in the controls when the form is valid, including
  // the slider and the radio cards, which a browser does not submit on its own.
  const onFormKey = (e: KeyboardEvent<HTMLFormElement>) => {
    const el = e.target as HTMLElement;
    if (e.key !== "Enter" || e.shiftKey) return;
    if (el.getAttribute("role") === "radio" || (el as HTMLInputElement).type === "range") {
      e.preventDefault();
      formRef.current?.requestSubmit();
    }
  };

  const retry = async (job: Job) => {
    setMessage(null);
    try {
      const result = await api.retry(job.id);
      await refresh(result.job_id, expected[job.id] ?? 4);
    } catch (e) {
      setMessage({ kind: "error", text: describeError(e) });
    }
  };

  /* -------------------------------------------------------- tile actions */

  const toControls = () => {
    if (window.matchMedia(NARROW).matches) window.scrollTo({ top: 0, behavior: stillMotion() ? "auto" : "smooth" });
  };

  const useAsBase = (job: Job, index: number, nextIntent = "change") => {
    resetForView();
    setBase({ kind: "sheet", job, index });
    if (nextIntent !== intentKey) setLine("");
    setIntentKey(nextIntent);
    if (nextIntent === "region") {
      setMarking(true);
    } else {
      toControls();
    }
  };

  const enlarge = async (job: Job, index: number) => {
    if (!size) return;
    setEnlarging(true);
    setMessage(null);
    try {
      const result = await api.continueFrom(job.id, {
        intent: "enlarge",
        extra_references: [],
        width: size.width,
        height: size.height,
        num_images: 1,
        output_index: index,
      });
      await refresh(result.job_id, 1);
    } catch (e) {
      setMessage({ kind: "error", text: describeError(e) });
    } finally {
      setEnlarging(false);
    }
  };

  const chooseIntent = (key: string) => {
    if (key !== intentKey) setLine("");
    setIntentKey(key);
    setMarking(key === "region" && base != null && !maskAsset);
    if (key === "region" && base) resultsRef.current?.scrollIntoView({ block: "start", behavior: stillMotion() ? "auto" : "smooth" });
  };

  const exportMask = async (blob: Blob) => {
    if (!base) return;
    try {
      const asset = await assetFor(base);
      setMaskAsset(await api.uploadMask(blob, asset.id));
      setMarking(false);
    } catch (e) {
      setMessage({ kind: "error", text: describeError(e) });
    }
  };

  const addRefs = async (files: FileList) => {
    const { ok } = pickImages(files);
    if (!ok.length || !intent) return;
    setUploadingRef(true);
    try {
      const room = intent.max_references - refs.length;
      const added: Asset[] = [];
      for (const f of ok.slice(0, room)) added.push(await api.upload(f));
      setRefs((r) => [...r, ...added]);
    } catch (e) {
      setMessage({ kind: "error", text: describeError(e) });
    } finally {
      setUploadingRef(false);
    }
  };

  const regionBase = marking && base && thumbUrl;
  const currentCount = current ? (expected[current.id] ?? 4) : 4;

  return (
    <div className="studio">
      {offline && (
        <p className="offline" role="status">
          Not reaching the server; retrying.
        </p>
      )}

      {catalogueFailed && (
        <Note
          kind="error"
          live
          actions={
            <button type="button" className="btn btn-sm" onClick={loadCatalogue}>
              Try again
            </button>
          }
        >
          {catalogueFailed}
        </Note>
      )}

      <div className="studio-grid">
        <form className="controls" ref={formRef} onSubmit={onSubmit} onKeyDown={onFormKey} aria-label="Render setup" noValidate>
          <ViewNote
            base={base}
            thumbUrl={thumbUrl}
            depthUrl={depth.overlayUrl}
            chip={chip}
            uploading={uploading}
            over={dragging}
            hasMaterials={hasMaterials}
            onOpenFile={() => fileInput.current?.click()}
            onCad={() => setCadOpen(true)}
            onToggleDepth={() => void depth.toggle()}
          />

          {catalogue && (
            <>
              <IntentNote
                cards={cards}
                intent={intent}
                viewKind={viewKind}
                refs={refs}
                uploadingRef={uploadingRef}
                hasBase={base != null}
                markedRegion={maskAsset != null}
                onIntent={chooseIntent}
                onView={setViewKind}
                onAddRef={(f) => void addRefs(f)}
                onRemoveRef={(id) => setRefs((r) => r.filter((x) => x.id !== id))}
                onMark={() => {
                  setMaskAsset(null);
                  setMarking(true);
                  resultsRef.current?.scrollIntoView({ block: "start", behavior: stillMotion() ? "auto" : "smooth" });
                }}
              />
              {intent?.slider && (
                <GeometryNote
                  catalogue={catalogue}
                  value={hold}
                  exactOk={exactOk}
                  drawing={drawing}
                  onChange={setHold}
                />
              )}
              <LookNote
                catalogue={catalogue}
                style={style}
                line={line}
                required={Boolean(intent?.needs_line)}
                label={intent?.needs_mask ? "What the marked part becomes" : "In your words"}
                placeholder={
                  intent?.needs_mask
                    ? "what the marked part becomes"
                    : intent?.needs_line
                      ? "what changes: light, materials, season"
                      : "materials, context, time of day"
                }
                onStyle={setStyle}
                onLine={setLine}
              />
            </>
          )}

          <div className="controls-foot">
            {message && (
              <Note kind={message.kind} live>
                {message.text}
              </Note>
            )}
            {catalogue && size && (
              <Footer
                count={imageCount}
                countFixed={Boolean(intent?.needs_mask)}
                maxCount={catalogue.num_images_max}
                sizeKey={size.key}
                sizes={sizes}
                estimate={estimate?.usd ?? null}
                overBy={overBy}
                busy={busy}
                reason={reason}
                showReason={tried}
                onCount={setCount}
                onSize={setSizeKey}
              />
            )}
          </div>
        </form>

        <div className="results" ref={resultsRef}>
          <ReadyNote ids={ready} onOpen={openSheet} />

          {regionBase ? (
            <section className="mask-panel" aria-label="Mark the region">
              <MaskCanvas
                src={thumbUrl}
                title="Paint over the part that is wrong"
                onExport={(b) => void exportMask(b)}
                onCancel={() => setMarking(false)}
              />
            </section>
          ) : current ? (
            <SheetPanel
              job={current}
              me={me}
              catalogue={catalogue}
              count={currentCount}
              now={now}
              sourceUrl={sources[current.id] ?? null}
              enlargeBusy={enlarging}
              actions={{
                onUse: (j, i) => useAsBase(j, i, "change"),
                onRegion: (j, i) => useAsBase(j, i, "region"),
                onEnlarge: (j, i) => void enlarge(j, i),
                onOpen: setLightbox,
              }}
              onRetry={(j) => void retry(j)}
              onOpenSheet={openSheet}
            />
          ) : (
            <EmptySheet hasView={base != null} />
          )}

          <Queue jobs={queueJobs} me={me} catalogue={catalogue} now={now} onRetry={(j) => void retry(j)} onOpenSheet={openSheet} />
          <Earlier
            jobs={earlier}
            me={me}
            currentId={currentId}
            catalogue={catalogue}
            canLoadMore={canLoadMore}
            onOpenSheet={openSheet}
            onMore={loadMore}
          />
        </div>
      </div>

      {lightbox != null && current && (
        <Lightbox
          job={current}
          index={lightbox}
          sourceUrl={sources[current.id] ?? null}
          onIndex={setLightbox}
          onClose={() => setLightbox(null)}
        />
      )}

      {cadOpen && (
        <CadDialog
          onClose={() => setCadOpen(false)}
          onConfirmed={(asset, depthAssetId, albedoAssetId, changeAssetId) => {
            resetForView();
            setBase({ kind: "asset", asset, depthAssetId, albedoAssetId, changeAssetId });
            if (albedoAssetId && changeAssetId) setIntentKey("material");
            setCadOpen(false);
          }}
        />
      )}

      <input
        ref={fileInput}
        type="file"
        className="sr-only"
        accept={IMAGE_ACCEPT}
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          if (e.target.files?.length) takeFiles(e.target.files);
          e.target.value = "";
        }}
      />

      {dragging && (
        <div className="drop-overlay" aria-hidden="true">
          <span>Drop the view</span>
        </div>
      )}
    </div>
  );
}
