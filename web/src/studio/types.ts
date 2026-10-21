import type { Asset, Job } from "../api";

/**
 * The view a render starts from. Either a picture the person brought (and,
 * after a CAD conversion, the depth and material passes that came with it), or
 * one image of an earlier sheet.
 */
export type Base =
  | {
      kind: "asset";
      asset: Asset;
      depthAssetId?: number;
      albedoAssetId?: number;
      changeAssetId?: number;
    }
  | { kind: "sheet"; job: Job; index: number };

export interface DepthChip {
  label: string;
  /** A depth pass exists or can be made, so the overlay can be shown. */
  canShow: boolean;
  showing: boolean;
  loading: boolean;
}
