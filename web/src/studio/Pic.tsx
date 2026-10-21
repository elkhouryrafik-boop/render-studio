import { useEffect, useState } from "react";

/**
 * An image that asks again, twice, if the first request fails. A lone failed
 * image on an office LAN is almost always a hiccup, not a missing file, and a
 * blank tile reads as a lost render.
 */
export function Pic({
  src,
  alt,
  className,
}: {
  src: string;
  alt: string;
  className?: string;
}) {
  const [attempt, setAttempt] = useState(0);
  useEffect(() => setAttempt(0), [src]);

  const url = attempt === 0 ? src : `${src}${src.includes("?") ? "&" : "?"}again=${attempt}`;
  return (
    <img
      className={className}
      src={url}
      alt={alt}
      draggable={false}
      onError={() => {
        if (attempt < 2) window.setTimeout(() => setAttempt((n) => n + 1), 400 * (attempt + 1));
      }}
    />
  );
}
