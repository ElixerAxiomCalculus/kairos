import { motion } from "framer-motion";
import { memo, type ReactNode } from "react";
import { Tip } from "@/components/common/Tip";
import { transition } from "@/tokens/motion";
import styles from "./telemetry.module.css";

interface Props {
  label: string;
  value: ReactNode;
  unit?: string;
  detail?: ReactNode;
  hint?: ReactNode;
  wide?: boolean;
  accent?: string;
  children?: ReactNode;
  testId?: string;
}

/** Level-1 telemetry: one number, one label; hover for the exact definition (level 2). */
function MetricCardImpl({ label, value, unit, detail, hint, wide, accent, children, testId }: Props) {
  const card = (
    <div className={styles.card} data-wide={wide} data-testid={testId} tabIndex={hint ? 0 : undefined}
      style={accent ? { boxShadow: `inset 2px 0 0 ${accent}` } : undefined}>
      <motion.div key={String(value)} className={styles.value} initial={{ opacity: 0.4 }} animate={{ opacity: 1 }} transition={transition.fast}>
        {value}{unit ? <small>{unit}</small> : null}
      </motion.div>
      <span className={`px ${styles.label}`}>{label}</span>
      {detail ? <span className={styles.detail}>{detail}</span> : null}
      {children ? <div className={styles.spark}>{children}</div> : null}
    </div>
  );
  return hint ? <Tip content={hint}>{card}</Tip> : card;
}

export const MetricCard = memo(MetricCardImpl);
