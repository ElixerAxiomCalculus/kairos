import { useLayoutEffect, useRef, useState } from "react";

/**
 * Element layout size for D3 layouts. Measured synchronously on mount and from ResizeObserver
 * entries (already batched per frame by the browser). contentRect/clientWidth are layout sizes,
 * so transient transforms never leak into graph coordinates.
 */
export function useElementSize<T extends HTMLElement>(): [React.RefObject<T | null>, { width: number; height: number }] {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const apply = (w: number, h: number) =>
      setSize((prev) => (Math.abs(prev.width - w) < 1 && Math.abs(prev.height - h) < 1 ? prev : { width: Math.round(w), height: Math.round(h) }));
    apply(el.clientWidth, el.clientHeight);
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (r) apply(r.width, r.height);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}
