import { useCallback, useState } from "react";
import { api, describeError } from "../api";
import type { Asset, DepthSource } from "../api";
import type { Base, DepthChip } from "./types";

const LABEL: Record<DepthSource, string> = {
  cad: "Depth: from CAD",
  estimated: "Depth: estimated",
  none: "Edges only",
};

/**
 * What the depth pass for the current view is, stated honestly.
 *
 * Before a render the source is a prediction (CAD if a bake attached one,
 * estimated if the server has a depth model, otherwise edges only). Once the
 * server has said what it used, that wins. The overlay is asked for once per
 * view and kept.
 */
export function useDepth(
  base: Base | null,
  assetFor: (b: Base) => Promise<Asset>,
  modelInstalled: boolean,
  onProblem: (sentence: string) => void,
) {
  const [observed, setObserved] = useState<DepthSource | null>(null);
  const [estimatedUrl, setEstimatedUrl] = useState<string | null>(null);
  const [showing, setShowing] = useState(false);
  const [loading, setLoading] = useState(false);

  const predicted: DepthSource =
    base?.kind === "asset" && base.depthAssetId ? "cad" : modelInstalled ? "estimated" : "none";
  const depthSource = observed ?? predicted;
  const cadUrl = base?.kind === "asset" && base.depthAssetId ? `/api/assets/${base.depthAssetId}/image` : null;
  const overlayUrl = cadUrl ?? estimatedUrl;

  // A previous render used as the base has no depth asset on this side; the
  // server estimates one when the job is submitted, so until a job says what
  // it used the chip says so rather than claiming a source.
  const atRender = !observed && base?.kind === "sheet" && depthSource === "estimated";
  const chip: DepthChip | null = base
    ? {
        label: atRender ? "Depth: estimated at render" : LABEL[depthSource],
        canShow: depthSource !== "none",
        showing,
        loading,
      }
    : null;

  const toggle = async () => {
    if (!base) return;
    if (showing) return setShowing(false);
    if (overlayUrl) return setShowing(true);
    setLoading(true);
    try {
      const result = await api.depthPreview((await assetFor(base)).id);
      setEstimatedUrl(result.asset.url);
      setObserved(result.depth_source);
      setShowing(true);
    } catch (e) {
      // The preview failed; what the server can do at render time is unchanged.
      onProblem(describeError(e));
    } finally {
      setLoading(false);
    }
  };

  const reset = useCallback(() => {
    setObserved(null);
    setEstimatedUrl(null);
    setShowing(false);
  }, []);

  return {
    depthSource,
    chip,
    exactOk: depthSource !== "none",
    overlayUrl,
    toggle,
    observe: setObserved,
    reset,
  };
}
