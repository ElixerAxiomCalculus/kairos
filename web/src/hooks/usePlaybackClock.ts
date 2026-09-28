import { useEffect } from "react";
import { useStream } from "@/stores/streamStore";

/** Advances the session clock while a recording plays (requestAnimationFrame, speed-scaled). */
export function usePlaybackClock(): void {
  const playing = useStream((s) => s.playing);
  const mode = useStream((s) => s.mode);

  useEffect(() => {
    if (!playing || mode !== "MOCK") return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const s = useStream.getState();
      const dt = (now - last) / 1000;
      last = now;
      s.setClock(s.clock + dt * s.speed);
      if (useStream.getState().playing) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, mode]);
}
