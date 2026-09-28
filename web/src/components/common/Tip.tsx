import * as Tooltip from "@radix-ui/react-tooltip";
import type { ReactNode } from "react";
import styles from "./common.module.css";

/** Radix tooltip with the Kairos surface — behaviour from Radix, look from tokens. */
export function Tip({ content, children, side = "top" }: { content: ReactNode; children: ReactNode; side?: "top" | "bottom" | "left" | "right" }) {
  return (
    <Tooltip.Root delayDuration={180}>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content className={styles.tooltip} side={side} sideOffset={6} collisionPadding={12}>
          {content}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}
