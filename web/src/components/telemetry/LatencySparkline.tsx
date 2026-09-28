import { memo } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, YAxis } from "recharts";
import type { LatencyMetric } from "@/types/domain";
import { ms } from "@/utils/format";

/** Sparkline of measured latencies. Hover shows the exact value and what it measured. */
function LatencySparklineImpl({ data, color = "var(--text-secondary)" }: { data: LatencyMetric[]; color?: string }) {
  if (data.length < 2) return <div style={{ height: 26 }} aria-hidden />;
  return (
    <ResponsiveContainer width="100%" height={26}>
      <LineChart data={data} margin={{ top: 3, right: 2, bottom: 3, left: 2 }}>
        <YAxis hide domain={[0, "dataMax"]} />
        <Tooltip
          cursor={false}
          contentStyle={{ background: "var(--surface-raised)", border: "1px solid var(--border-strong)", borderRadius: 6, fontSize: 11, fontFamily: "var(--font-mono)" }}
          labelFormatter={() => ""}
          formatter={(v: number, _n, item) => [ms(v), (item?.payload as LatencyMetric | undefined)?.label ?? ""]}
        />
        <Line type="monotone" dataKey="ms" stroke={color} strokeWidth={1.3} dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export const LatencySparkline = memo(LatencySparklineImpl);
