import { memo } from "react";
import type { GroundingResult, TelemetryEvent } from "@/types/domain";
import { STATUS_VAR } from "@/tokens/motion";
import { DASH } from "@/utils/format";
import styles from "./telemetry.module.css";

interface Props {
  result: GroundingResult | null;
  transform: TelemetryEvent | null; // latest TransformVerified, when the latest version is a presentation change
}

/** The firewall never pretends: BLOCKED claims and fabricated IDs are shown explicitly. */
function GroundingFirewallImpl({ result, transform }: Props) {
  if (transform) {
    const p = transform.payload as { passed?: boolean; violations?: { claim_id: string; rule: string }[]; checks?: string[] };
    const ok = Boolean(p.passed);
    return (
      <div className={styles.firewall} data-testid="grounding-firewall" data-status={ok ? "PASS" : "BLOCKED"}>
        <div className={styles.fwHead}>
          <span className="px muted">TRANSFORMATION FIREWALL</span>
          <span className={styles.fwStatus} style={{ color: ok ? STATUS_VAR.PASS : STATUS_VAR.FAIL }}>{ok ? "PASS" : "BLOCKED"}</span>
        </div>
        <div className={styles.fwRows}>
          <span>New facts introduced</span><span>{p.violations?.filter((v) => v.rule === "NEW_NUMERIC_ANCHOR").length ?? 0}</span>
          <span>New citations</span><span>{p.violations?.filter((v) => v.rule === "NEW_CITATION").length ?? 0}</span>
          <span>Derived from parent</span><span>{ok ? "all" : "partial"}</span>
        </div>
        <div className={styles.checks}>{(p.checks ?? []).map((c) => <span key={c} className={styles.check}>{c}</span>)}</div>
      </div>
    );
  }
  if (!result) {
    return (
      <div className={styles.firewall} data-testid="grounding-firewall" data-status="IDLE">
        <div className={styles.fwHead}>
          <span className="px muted">GROUNDING FIREWALL</span>
          <span className={styles.fwStatus} style={{ color: "var(--text-muted)" }}>{DASH}</span>
        </div>
        <span className="px muted">NO CLAIMS CHECKED YET</span>
      </div>
    );
  }
  const pass = result.status === "PASS";
  const blocked = result.results.filter((r) => r.verdict === "UNSUPPORTED" || r.verdict === "INVALID_CITATION");
  return (
    <div className={styles.firewall} data-testid="grounding-firewall" data-status={result.status}>
      <div className={styles.fwHead}>
        <span className="px muted">GROUNDING FIREWALL</span>
        <span className={styles.fwStatus} style={{ color: pass ? STATUS_VAR.PASS : STATUS_VAR.FAIL }}>
          {pass ? "PASS" : "BLOCKED"}
        </span>
      </div>
      <div className={styles.fwRows}>
        <span>Claims verified</span><span>{result.emitted} / {result.checked}</span>
        <span>Supported</span><span>{result.supported}</span>
        <span>Conflicting (shown both)</span><span style={{ color: result.conflicting ? STATUS_VAR.CONFLICTING : undefined }}>{result.conflicting}</span>
        <span>Partial</span><span>{result.partial}</span>
        <span>Unsupported</span><span style={{ color: result.unsupported ? STATUS_VAR.FAIL : undefined }}>{result.unsupported}</span>
        <span>Citations valid</span><span>{result.citationsValid}</span>
        <span>Fabricated IDs blocked</span><span style={{ color: result.fabricatedIdsBlocked ? STATUS_VAR.FAIL : undefined }}>{result.fabricatedIdsBlocked}</span>
      </div>
      {blocked.map((b) => (
        <div key={b.claim_id} className={styles.blocked}>
          <span className="px" style={{ color: STATUS_VAR.FAIL }}>GROUNDING BLOCKED · {b.claim_id}</span>
          <div className="secondary">{b.verdict === "INVALID_CITATION" ? "Citation does not resolve to retrieved evidence" : "Insufficient evidence"}</div>
        </div>
      ))}
      <div className={styles.checks}>{result.checks.map((c) => <span key={c} className={styles.check}>{c.replace(/_/g, " ").toLowerCase()}</span>)}</div>
    </div>
  );
}

export const GroundingFirewall = memo(GroundingFirewallImpl);
