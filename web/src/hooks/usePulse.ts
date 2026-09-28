import { useEffect, useRef, useState } from "react";
import { PULSE_HOLD_MS } from "@/tokens/motion";

/**
 * Returns true for PULSE_HOLD_MS after `key` changes to a truthy value. Retrievals complete in
 * milliseconds; this keeps a real event perceivable without inventing one.
 */
export function usePulse(key: string | number | null | undefined, hold = PULSE_HOLD_MS): boolean {
  const [on, setOn] = useState(false);
  const last = useRef(key);
  useEffect(() => {
    if (key === last.current) return;
    last.current = key;
    if (!key) return;
    setOn(true);
    const t = window.setTimeout(() => setOn(false), hold);
    return () => window.clearTimeout(t);
  }, [key, hold]);
  return on;
}
