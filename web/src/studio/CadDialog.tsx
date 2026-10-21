import { useEffect, useRef } from "react";
import type { Asset } from "../api";
import { CadPanel } from "../Cad";

/**
 * The CAD converter, opened over the studio. A native dialog gives the focus
 * trap and Esc for free, and keeps the conversion from living inside the
 * controls column, where a preview and a layer list do not fit.
 */
export function CadDialog({
  onConfirmed,
  onClose,
}: {
  onConfirmed: (
    asset: Asset,
    depthAssetId?: number,
    albedoAssetId?: number,
    changeAssetId?: number,
  ) => void;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const el = dialog.current;
    if (el && !el.open) el.showModal();
    return () => el?.close();
  }, []);

  return (
    <dialog
      ref={dialog}
      className="modal"
      aria-label="Convert a DXF, IFC or SKP"
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
    >
      <CadPanel onConfirmed={onConfirmed} onCancel={onClose} />
    </dialog>
  );
}
