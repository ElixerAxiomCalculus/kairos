import { useMemo } from "react";
import { Panel } from "@/components/common/Panel";
import { IntentGraph } from "@/components/intent/IntentGraph";
import { useStream } from "@/stores/streamStore";
import { useUi } from "@/stores/uiStore";
import { QueryState } from "./QueryState";
import { RetrievalController } from "./RetrievalController";
import styles from "./controller.module.css";

/** Nested bento: controller (iconic) · evolving query · intent graph. */
export function ControlCenter({ className }: { className?: string }) {
  const controller = useStream((s) => s.view.controller);
  const queryVersions = useStream((s) => s.view.queryVersions);
  const intentsMap = useStream((s) => s.view.intents);
  const intentOrder = useStream((s) => s.view.intentOrder);
  const active = useStream((s) => s.view.activeRetrievals);
  const delta = useStream((s) => s.view.delta);
  const versions = useStream((s) => s.view.versions);
  const focus = useUi((s) => s.focus);

  const current = queryVersions.at(-1) ?? null;
  const previous = queryVersions.at(-2) ?? null;

  // Show the intents of the latest information-seeking utterance (constraints/transforms reuse them).
  const intents = useMemo(() => {
    const lastUtt = intentOrder.length ? intentsMap[intentOrder[intentOrder.length - 1]].utteranceId : null;
    return intentOrder.map((id) => intentsMap[id]).filter((i) => i.utteranceId === lastUtt);
  }, [intentsMap, intentOrder]);
  const recent = useMemo(() => Object.fromEntries(intents.map((i) => [i.id, i.retrievalCount])), [intents]);
  // highlight affected branches while the delta is in progress or is the latest answer
  const latestUtt = versions.at(-1)?.utteranceId;
  const deltaForView = delta && (latestUtt === delta.utteranceId || !versions.some((v) => v.utteranceId === delta.utteranceId)) ? delta : null;
  // answer focus: controller + query only; evidence focus: the evidence graph already shows intents
  const compact = focus === "answer" || focus === "evidence";

  return (
    <Panel label="SYSTEM / CONTROL" meta={current ? `Q${String(current.version).padStart(2, "0")}` : undefined}
      className={className} testId="panel-control">
      <div className={styles.wrapQuery}>
      <div className={styles.grid} data-compact={compact}>
        <RetrievalController decision={controller} queryVersion={current?.version ?? 0} />
        <QueryState current={current} previous={previous} />
        {!compact ? (
          <div className={styles.intents}>
            <IntentGraph intents={intents} queryVersion={current?.version ?? 0} active={active} recent={recent} delta={deltaForView} />
          </div>
        ) : null}
      </div>
      </div>
    </Panel>
  );
}
