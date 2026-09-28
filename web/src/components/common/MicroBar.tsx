import { motion } from "framer-motion";
import { score } from "@/utils/format";
import { transition } from "@/tokens/motion";
import styles from "./common.module.css";

interface MicroBarProps {
  name: string;
  value?: number;
  hint?: string;
  color?: string;
}

/** Compact measurement: label, value, hairline bar. Absent values render as an em dash, never 0. */
export function MicroBar({ name, value, hint, color }: MicroBarProps) {
  const v = value === undefined ? 0 : Math.max(0, Math.min(1, value));
  return (
    <div className={styles.micro} title={hint}>
      <span className={`px ${styles.name}`}>{name}</span>
      <span className={`mono ${styles.value}`}>{score(value)}</span>
      <div className={styles.track} aria-hidden>
        <motion.div
          className={styles.fill}
          initial={false}
          animate={{ scaleX: v }}
          transition={transition.normal}
          style={{ background: color }}
        />
      </div>
    </div>
  );
}
