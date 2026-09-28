import * as Dialog from "@radix-ui/react-dialog";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { useMemo } from "react";
import { StatusPill } from "@/components/common/StatusPill";
import { useStream } from "@/stores/streamStore";
import { useUi } from "@/stores/uiStore";
import type { Claim, EvidenceChunk } from "@/types/domain";
import { transition } from "@/tokens/motion";
import { DASH, ms, pad, score } from "@/utils/format";
import styles from "./evidence.module.css";

function highlight(text: string, quote?: string) {
  if (!quote) return text;
  const i = text.indexOf(quote);
  if (i < 0) return text;
  return (
    <>
      {text.slice(0, i)}
      <mark>{quote}</mark>
      {text.slice(i + quote.length)}
    </>
  );
}

function LinkChips({ ids, kind }: { ids: string[]; kind: "claim" | "chunk" | "intent" }) {
  const select = useUi((s) => s.select);
  if (!ids.length) return <span className="muted mono">{DASH}</span>;
  return (
    <div className={styles.linkList}>
      {ids.map((id) => (
        <button key={id} type="button" className="chip mono" style={{ cursor: "pointer", fontSize: 11 }}
          onClick={() => select({ kind, id } as never)}>{id}</button>
      ))}
    </div>
  );
}

function ChunkView({ ev, claimId }: { ev: EvidenceChunk; claimId?: string }) {
  const claims = useStream((s) => s.view.claims);
  const grounding = useStream((s) => s.view.grounding);
  const contradictions = useStream((s) => s.view.contradictions);
  const evidence = useStream((s) => s.view.evidence);
  const supports = Object.values(claims).filter((c) => c.evidenceIds.includes(ev.chunkId));
  const quote = claimId ? claims[claimId]?.quote : supports[0]?.quote;
  const verdicts = grounding.flatMap((g) => g.results.filter((r) => r.evidence.includes(ev.chunkId)));
  const invalid = grounding.some((g) => g.results.some((r) => r.invalid_ids.includes(ev.chunkId)));
  const conflict = contradictions.find((c) => c.chunks.includes(ev.chunkId));
  const other = conflict ? evidence[conflict.chunks.find((c) => c !== ev.chunkId) ?? ""] : undefined;
  return (
    <>
      <div className={styles.idRow}>
        <div className={styles.idCell}><div className="px muted">DOCUMENT</div><div className="v">{ev.documentId}</div></div>
        <div className={styles.idCell}><div className="px muted">CHUNK</div><div className="v">{ev.chunkId.split(":")[1]}</div></div>
      </div>
      <div>
        <div style={{ fontWeight: 500 }}>{ev.docTitle}</div>
        <div className="secondary" style={{ fontSize: 13 }}>§{ev.section} · {ev.sectionPath}</div>
      </div>
      <div className={styles.scores}>
        {[["SPARSE", ev.sparseScore], ["DENSE", ev.denseScore], ["FUSED", ev.fusedScore], ["RERANK", ev.rerankScore]].map(([k, v]) => (
          <div key={k as string} className={styles.scoreCell}><div className="px muted">{k}</div><div className={styles.v ?? ""} style={{ fontFamily: "var(--font-mono)" }}>{score(v as number | undefined, 3)}</div></div>
        ))}
      </div>
      <div className={styles.passage} data-testid="inspector-passage">{ev.text ? highlight(ev.text, quote) : <span className="muted">Content not included in this trace.</span>}</div>
      <div className={styles.section}>
        <h4 className="px muted">Citation validity</h4>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          <StatusPill state={invalid ? "FAIL" : "PASS"} label={invalid ? "INVALID CITATION" : "IN SESSION RETRIEVAL · SHA MATCH"} />
          {verdicts.slice(-3).map((r) => <StatusPill key={`${r.claim_id}`} state={r.verdict} label={`${r.claim_id} ${r.verdict} ${score(r.support)}`} />)}
        </div>
      </div>
      <div className={styles.section}><h4 className="px muted">Intents</h4><LinkChips ids={ev.intents} kind="intent" /></div>
      <div className={styles.section}><h4 className="px muted">Supports claims</h4><LinkChips ids={supports.map((c) => c.id)} kind="claim" /></div>
      <div className={styles.section}>
        <h4 className="px muted">Metadata</h4>
        <div className="mono secondary">effective {ev.effectiveDate ?? DASH} · {ev.docType ?? DASH} · facets {Object.values(ev.facets).join(", ") || "none"}</div>
      </div>
      {conflict && other ? (
        <div className={styles.section} data-testid="conflict-compare">
          <h4 className="px" style={{ color: "var(--status-conflict)" }}>Conflict detected · {conflict.unit}</h4>
          <div className={styles.compare}>
            <div><div className="px muted">{ev.chunkId} · {ev.effectiveDate}</div>{ev.text}</div>
            <div><div className="px muted">{other.chunkId} · {other.effectiveDate}</div>{other.text}</div>
          </div>
          <div className="secondary" style={{ fontSize: 12.5, marginTop: 6 }}>More recent source: {conflict.moreRecent}. Both are shown in the answer; neither is silently dropped.</div>
        </div>
      ) : null}
    </>
  );
}

function ClaimView({ claim }: { claim: Claim }) {
  const versions = useStream((s) => s.view.versions);
  const evidence = useStream((s) => s.view.evidence);
  const inVersions = versions.filter((v) => v.claims.some((c) => c.id === claim.id)).map((v) => `V${pad(v.id)}`);
  return (
    <>
      <div className={styles.idRow}>
        <div className={styles.idCell}><div className="px muted">CLAIM</div><div className="v">{claim.id}</div></div>
        <div className={styles.idCell}><div className="px muted">GROUNDING</div><div className="v">{score(claim.support)}</div></div>
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        <StatusPill state={claim.supportStatus} />
        <StatusPill state={claim.status === "REVISED" || claim.status === "ADDED" ? "REFINE" : "WAIT"} label={claim.status} />
        <span className="chip mono">lineage {claim.lineageId}</span>
        <span className="chip mono">intent {claim.subQueryId}</span>
      </div>
      <div className={styles.passage}>{claim.text}</div>
      <div className={styles.section}><h4 className="px muted">Answer versions</h4><div className="mono">{inVersions.join(" → ") || DASH}</div></div>
      <div className={styles.section}><h4 className="px muted">Derived from</h4><LinkChips ids={claim.derivedFrom} kind="claim" /></div>
      <div className={styles.section}><h4 className="px muted">Support</h4><LinkChips ids={claim.evidenceIds} kind="chunk" /></div>
      {claim.evidenceIds.map((id) => evidence[id]?.text ? (
        <div key={id} className={styles.section}>
          <h4 className="px muted">{id} · verbatim quote</h4>
          <div className={styles.passage} style={{ fontSize: 16 }}>{highlight(evidence[id].text ?? "", claim.quote)}</div>
        </div>
      ) : null)}
    </>
  );
}

function IntentView({ id }: { id: string }) {
  const intent = useStream((s) => s.view.intents[id]);
  const events = useStream((s) => s.view.events);
  if (!intent) return null;
  const retrievals = events.filter((e) => e.eventType === "RetrievalCompleted" && e.subQueryId === id);
  return (
    <>
      <div className={styles.idRow}>
        <div className={styles.idCell}><div className="px muted">INTENT</div><div className="v">{intent.id}</div></div>
        <div className={styles.idCell}><div className="px muted">RETRIEVALS</div><div className="v">{intent.retrievalCount}</div></div>
      </div>
      <div className={styles.passage} style={{ fontSize: 17 }}>{intent.normalizedQuery}</div>
      <div className={styles.section}><h4 className="px muted">Label</h4><div>{intent.label}</div></div>
      <div className={styles.section}><h4 className="px muted">Dependencies</h4><LinkChips ids={intent.dependencies} kind="intent" /></div>
      <div className={styles.section}>
        <h4 className="px muted">Retrieval history</h4>
        {retrievals.map((e) => (
          <div key={e.id} className="mono secondary" style={{ marginBottom: 4 }}>
            {e.retrievalTrigger} · {ms(e.latency)} · {e.chunkIds.join(" ")}
          </div>
        ))}
      </div>
      <div className={styles.section}><h4 className="px muted">Evidence footprint</h4><div className="mono secondary">{intent.footprint.join(" · ") || DASH}</div></div>
    </>
  );
}

/** Right-side contextual inspector. Non-modal: the dashboard stays interactive underneath. */
export function EvidenceInspector() {
  const open = useUi((s) => s.inspectorOpen);
  const selection = useUi((s) => s.selection);
  const close = useUi((s) => s.closeInspector);
  const evidence = useStream((s) => s.view.evidence);
  const claims = useStream((s) => s.view.claims);

  const title = useMemo(() => {
    if (!selection) return "";
    if (selection.kind === "chunk") return "EVIDENCE INSPECTOR";
    if (selection.kind === "claim") return "CLAIM INSPECTOR";
    if (selection.kind === "intent") return "INTENT INSPECTOR";
    return "";
  }, [selection]);

  return (
    <Dialog.Root open={open} onOpenChange={(o) => !o && close()} modal={false}>
      <AnimatePresence>
        {open && selection ? (
          <Dialog.Portal forceMount>
            <Dialog.Content asChild forceMount onInteractOutside={(e) => e.preventDefault()} aria-describedby={undefined}>
              <motion.aside className={styles.drawer} initial={{ x: 24, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
                exit={{ x: 24, opacity: 0 }} transition={transition.normal} data-testid="evidence-inspector">
                <header className="panel-head">
                  <Dialog.Title className="px" style={{ margin: 0, fontWeight: 400 }}>{title}</Dialog.Title>
                  <Dialog.Close className="btn btn-icon" aria-label="Close inspector"><X size={15} /></Dialog.Close>
                </header>
                <div className={styles.drawerBody}>
                  {selection.kind === "chunk" && evidence[selection.id] ? <ChunkView ev={evidence[selection.id]} claimId={selection.claimId} /> : null}
                  {selection.kind === "chunk" && !evidence[selection.id] ? <p className="secondary">Chunk {selection.id} was not retrieved in this session.</p> : null}
                  {selection.kind === "claim" && claims[selection.id] ? <ClaimView claim={claims[selection.id]} /> : null}
                  {selection.kind === "intent" ? <IntentView id={selection.id} /> : null}
                </div>
              </motion.aside>
            </Dialog.Content>
          </Dialog.Portal>
        ) : null}
      </AnimatePresence>
    </Dialog.Root>
  );
}
