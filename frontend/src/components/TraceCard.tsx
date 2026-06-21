import type { AuditVerdict, ConsistencyComparison, TraceEntry } from "../types";
import { statusColor } from "../format";

interface Props {
  trace: TraceEntry[];
  audit: AuditVerdict[];
  consistency: ConsistencyComparison[];
}

export function TraceCard({ trace, audit, consistency }: Props) {
  return (
    <div className="panel card full">
      <h2>Pipeline trace</h2>
      <div className="timeline">
        {trace.map((entry, i) => (
          <div className="step" key={i}>
            <div className="name">
              {entry.step}
              {entry.duration_ms > 0 && (
                <span className="ms">{Math.round(entry.duration_ms)} ms</span>
              )}
            </div>
            <div className="detail">{entry.detail}</div>
          </div>
        ))}
      </div>

      {audit.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h2>LLM auditor</h2>
          {audit.map((verdict, i) => (
            <div className="row" key={i}>
              <span className="dim">
                <span
                  className="status-dot"
                  style={{ background: statusColor(verdict.trustworthy ? "passed" : "failed") }}
                />
                {verdict.field}
              </span>
              <span className="detail">{verdict.reason || (verdict.trustworthy ? "supported" : "not supported")}</span>
            </div>
          ))}
        </div>
      )}

      {consistency.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h2>Self-consistency (2nd model)</h2>
          {consistency.map((c, i) => (
            <div className="row" key={i}>
              <span className="dim">
                <span className="status-dot" style={{ background: statusColor(c.agreement) }} />
                {c.field}
              </span>
              <span className="detail">
                {c.agreement === "agrees" ? "both agree" : `${c.primary} ≠ ${c.secondary}`}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
