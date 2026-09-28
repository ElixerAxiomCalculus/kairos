import { AnimatePresence, motion } from "framer-motion";
import { GitCompareArrows } from "lucide-react";
import { useMemo, useState } from "react";
import { EmptyState } from "@/components/common/EmptyState";
import { Panel } from "@/components/common/Panel";
import { StatusPill } from "@/components/common/StatusPill";
import { useStream } from "@/stores/streamStore";
import { useUi } from "@/stores/uiStore";
import type { AnswerBlock, AnswerVersion, Claim } from "@/types/domain";
import { transition } from "@/tokens/motion";
import { pad } from "@/utils/format";
import { AnswerVersionRail } from "./AnswerVersionRail";
import { ClaimBlock } from "./ClaimBlock";
import { DeltaBanner } from "./DeltaBanner";
import { GroundingStrip } from "./GroundingStrip";
import { SuppressionSequence } from "./SuppressionSequence";
import styles from "./answer.module.css";

function retractedIds(v: AnswerVersion): string[] {
  return v.diff.retracted.map((r) => (typeof r === "string" ? r : r.claim_id));
}

function Block({ block, claims, showDiff, retracted }: {
  block: AnswerBlock;
  claims: Record<string, Claim>;
  showDiff: boolean;
  retracted: Claim[];
}) {
  const items = block.claim_ids.map((id) => claims[id]).filter(Boolean);
  const body = (
    <>
      {items.map((c) => <ClaimBlock key={c.id} claim={c} showDiff={showDiff} />)}
      {showDiff ? retracted.map((c) => <span key={`x-${c.id}`} className={styles.retracted} title={`${c.id} removed`}>{c.text} </span>) : null}
    </>
  );
  return (
    <div className={styles.block}>
      {block.style === "paragraph" ? (
        <div className={`${styles.blockHead} px`}>
          <span>{block.sub_query_id ?? "—"}</span>
          <span>{block.title}</span>
        </div>
      ) : null}
      {block.style === "bullet" ? <ul className={styles.bullets}><li>{body}</li></ul> : <p className={styles.prose}>{body}</p>}
      {block.uncertainty ? (
        <div className={styles.uncertain} data-testid="uncertainty">
          <span className="px" style={{ color: "var(--status-fail)" }}>INSUFFICIENT EVIDENCE</span>
          <span>{block.uncertainty.replace(/^Insufficient evidence in the provided corpus to verify:\s*/, "")}</span>
        </div>
      ) : null}
    </div>
  );
}

/** Editorial research document, not chat: claims are the unit, citations are objects, versions have lineage. */
export function GroundedAnswer({ className }: { className?: string }) {
  const versions = useStream((s) => s.view.versions);
  const delta = useStream((s) => s.view.delta);
  const suppression = useStream((s) => s.view.suppression);
  const stage = useStream((s) => s.view.stage);
  const allClaims = useStream((s) => s.view.claims);
  const unavailable = useStream((s) => s.view.transformUnavailable);
  const grounding = useStream((s) => s.view.grounding);
  const viewVersion = useUi((s) => s.viewVersion);
  const setViewVersion = useUi((s) => s.setViewVersion);
  const [showDiff, setShowDiff] = useState(true);

  const version = useMemo(() => {
    if (!versions.length) return null;
    return versions.find((v) => v.id === viewVersion) ?? versions[versions.length - 1];
  }, [versions, viewVersion]);
  const claims = useMemo(() => Object.fromEntries((version?.claims ?? []).map((c) => [c.id, c])), [version]);
  const verified = version?.claims.filter((c) => c.supportStatus === "SUPPORTED" || c.supportStatus === "CONFLICTING" || c.supportStatus === "PARTIAL").length ?? 0;

  const retractedByBlock = useMemo(() => {
    const out: Record<string, Claim[]> = {};
    if (!version) return out;
    for (const id of retractedIds(version)) {
      const c = allClaims[id];
      if (c && version.kind === "SEMANTIC") (out[c.subQueryId] ??= []).push(c);
    }
    return out;
  }, [version, allClaims]);

  const conflicts = version?.conflicts.filter((c) => c.note) ?? [];
  const showSuppression = version?.kind === "PRESENTATION" && suppression && suppression.utteranceId === version.utteranceId;

  return (
    <Panel
      label="ANSWER WORKSPACE"
      meta={version ? `${version.kind === "PRESENTATION" ? "presentation" : "semantic"} · parent ${version.parentVersion ? `V${pad(version.parentVersion)}` : "—"}` : undefined}
      className={className}
      noPadding
      testId="panel-answer"
      actions={
        version && version.parentVersion ? (
          <button type="button" className="btn" aria-pressed={showDiff} onClick={() => setShowDiff((s) => !s)}>
            <GitCompareArrows size={14} /> <span className="px">DIFF</span>
          </button>
        ) : null
      }
    >
      <div className={styles.wrap}>
        <div className={styles.head}>
          <div>
            <h2 className={styles.title}>Grounded <em>Answer</em></h2>
            <div className={`${styles.subtitle} px`} data-testid="answer-subtitle">
              {version ? `ANSWER V${pad(version.id)} / ${verified} VERIFIED CLAIM${verified === 1 ? "" : "S"}` : "NO ANSWER VERSION YET"}
            </div>
          </div>
          {version ? (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 6 }}>
              <StatusPill state={version.kind === "PRESENTATION" ? "SUPPRESS" : version.trigger === "CONSTRAINT" ? "REFINE" : "RETRIEVE"}
                label={version.kind === "PRESENTATION" ? "PRESENTATION CHANGE" : version.trigger === "CONSTRAINT" ? "DELTA UPDATE" : "NEW ANSWER"} />
              <GroundingStrip version={version} grounding={grounding} />
            </div>
          ) : null}
        </div>
        <AnswerVersionRail versions={versions} current={version?.id ?? null}
          onSelect={(v) => setViewVersion(v === versions.at(-1)?.id ? null : v)} />
        {version?.trigger === "CONSTRAINT" && version.kind === "SEMANTIC" ? <DeltaBanner version={version} delta={delta} /> : null}
        {showSuppression && suppression ? <SuppressionSequence info={suppression} /> : null}
        {unavailable && (!version || unavailable.timestamp > version.timestamp) ? (
          <div className={styles.banner} data-kind="suppress"><span className="secondary">{unavailable.reason}</span></div>
        ) : null}
        {conflicts.length ? (
          <div className={styles.banner} data-kind="conflict" data-testid="conflict-banner">
            <div className={styles.bannerCell}>
              <span className="px" style={{ color: "var(--status-conflict)" }}>CONFLICT DETECTED</span>
              <span className="secondary" style={{ fontSize: 13 }}>{conflicts[0].note}</span>
            </div>
          </div>
        ) : null}
        <div className={styles.body} aria-live="polite">
          <AnimatePresence mode="wait" initial={false}>
            {version ? (
              <motion.div key={version.id} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -12 }}
                transition={transition.normal} data-testid="answer-body" data-version={version.id}>
                {version.blocks.map((b, i) => (
                  <Block key={`${b.sub_query_id}-${i}`} block={b} claims={claims} showDiff={showDiff && version.parentVersion !== null}
                    retracted={b.sub_query_id ? retractedByBlock[b.sub_query_id] ?? [] : []} />
                ))}
                {version.notes.map((n) => <p key={n} className={styles.notes}>{n}</p>)}
              </motion.div>
            ) : (
              <motion.div key="empty" className={stage === "retrieving" ? "scan" : undefined} style={{ height: "100%" }}>
                <EmptyState label="Awaiting grounded evidence" animate={stage === "retrieving"}>
                  Claims appear only after the grounding firewall verifies them against a retrieved chunk.
                </EmptyState>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </Panel>
  );
}
