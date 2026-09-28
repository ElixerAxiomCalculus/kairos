import { motion } from "framer-motion";

/** Brand motif: a single line that branches and reconnects — "continuously evolving evidence". */
export function BranchLine({ width = 120, height = 18, animate = false, strokeWidth = 1.25 }: {
  width?: number;
  height?: number;
  animate?: boolean;
  strokeWidth?: number;
}) {
  const mid = height / 2;
  const a = width * 0.28;
  const b = width * 0.72;
  const d = `M0 ${mid} H${a} C${a + 10} ${mid} ${a + 12} 2 ${a + 24} 2 H${b - 24} C${b - 12} 2 ${b - 10} ${mid} ${b} ${mid} H${width}`;
  const d2 = `M${a} ${mid} C${a + 10} ${mid} ${a + 12} ${height - 2} ${a + 24} ${height - 2} H${b - 24} C${b - 12} ${height - 2} ${b - 10} ${mid} ${b} ${mid}`;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden style={{ overflow: "visible" }}>
      <path d={d} fill="none" stroke="currentColor" strokeWidth={strokeWidth} />
      <path d={d2} fill="none" stroke="currentColor" strokeWidth={strokeWidth} opacity={0.45} />
      {animate ? (
        <motion.circle
          r={2.2}
          fill="var(--status-retrieve)"
          initial={{ offsetDistance: "0%" }}
          animate={{ offsetDistance: "100%" }}
          transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
          style={{ offsetPath: `path("${d}")` }}
        />
      ) : null}
    </svg>
  );
}
