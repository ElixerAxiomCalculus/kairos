import * as Popover from "@radix-ui/react-popover";
import { useStream } from "@/stores/streamStore";
import { DASH, wall } from "@/utils/format";
import styles from "./common.module.css";

/** Answers must be corpus-grounded, so corpus readiness is always one click away. */
export function CorpusStatus() {
  const corpus = useStream((s) => s.view.session.corpus);
  const config = useStream((s) => s.view.session.config);
  const ready = Boolean(corpus);
  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button className={styles.status} aria-label="Corpus status" data-testid="corpus-status">
          <span className="dot" style={{ background: ready ? "var(--status-pass)" : "var(--status-wait)" }} />
          <span className="px">CORPUS / {ready ? "READY" : "—"}</span>
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className={styles.popover} sideOffset={8} align="end" collisionPadding={12}>
          <div className="px muted" style={{ marginBottom: 10 }}>CORPUS INDEX</div>
          {corpus ? (
            <dl className={`${styles.kv} mono`}>
              <dt>Documents</dt><dd>{corpus.documents}</dd>
              <dt>Chunks</dt><dd>{corpus.chunks}</dd>
              <dt>Dense</dt><dd>{corpus.embedding_model}</dd>
              <dt>Sparse</dt><dd>{corpus.sparse_index}</dd>
              <dt>Reranker</dt><dd>{corpus.reranker}</dd>
              <dt>Indexed</dt><dd>{wall(corpus.indexed_at)}</dd>
              <dt>Facets</dt><dd>{corpus.facet_dimensions ? Object.values(corpus.facet_dimensions).map((v) => v.join("/")).join("; ") : DASH}</dd>
              {config ? (<><dt>Synthesis</dt><dd>{config.synthesis_backend}</dd><dt>Controller</dt><dd>{config.controller_mode}</dd></>) : null}
            </dl>
          ) : (
            <p className="secondary" style={{ margin: 0, fontSize: 13 }}>No session yet — corpus details arrive with the SessionStarted event.</p>
          )}
          <Popover.Arrow style={{ fill: "var(--border-strong)" }} />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
