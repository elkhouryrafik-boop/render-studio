import { useRef } from "react";
import type { KeyboardEvent, ReactNode } from "react";

export interface RadioItem<T extends string> {
  value: T;
  label: ReactNode;
  /** Explains why it cannot be chosen; the option stays focusable and readable. */
  disabled?: string;
  hint?: string;
}

/**
 * A real radiogroup: one tab stop, arrow keys move the selection, and the
 * selected item says so with `aria-checked` and a marker, never colour alone.
 */
export function RadioGroup<T extends string>({
  label,
  value,
  items,
  onPick,
  className,
  itemClassName,
}: {
  label: string;
  value: T | null;
  items: RadioItem<T>[];
  onPick: (value: T) => void;
  className: string;
  itemClassName: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const pickable = items.filter((i) => !i.disabled);
  const tabbable = items.find((i) => i.value === value && !i.disabled)?.value ?? pickable[0]?.value;

  const onKey = (e: KeyboardEvent) => {
    // Enter chooses the focused option, as Space does. On the option that is
    // already chosen it is left alone, so it still reaches the form and renders.
    if (e.key === "Enter" && !e.shiftKey) {
      const at = refs.current.findIndex((el) => el === e.target);
      const item = items[at];
      if (item && !item.disabled && item.value !== value) {
        e.preventDefault();
        e.stopPropagation();
        onPick(item.value);
      }
      return;
    }
    const forward = e.key === "ArrowRight" || e.key === "ArrowDown";
    const back = e.key === "ArrowLeft" || e.key === "ArrowUp";
    if (!forward && !back) return;
    e.preventDefault();
    if (pickable.length === 0) return;
    const at = pickable.findIndex((i) => i.value === value);
    const next = pickable[(at + (forward ? 1 : -1) + pickable.length) % pickable.length];
    if (!next) return;
    onPick(next.value);
    const idx = items.findIndex((i) => i.value === next.value);
    refs.current[idx]?.focus();
  };

  return (
    <div className={className} role="radiogroup" aria-label={label} onKeyDown={onKey}>
      {items.map((item, i) => (
        <button
          key={item.value}
          ref={(el) => {
            refs.current[i] = el;
          }}
          type="button"
          role="radio"
          aria-checked={item.value === value}
          aria-disabled={item.disabled ? true : undefined}
          tabIndex={item.value === tabbable ? 0 : -1}
          className={itemClassName}
          title={item.disabled ?? item.hint}
          onClick={() => {
            if (!item.disabled) onPick(item.value);
          }}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
