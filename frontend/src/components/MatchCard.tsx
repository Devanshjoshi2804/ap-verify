import type { MatchData } from "../types";
import { statusColor } from "../format";

export function MatchCard({ match }: { match: MatchData }) {
  const color = statusColor(match.outcome.toLowerCase());
  return (
    <div className="panel card">
      <h2>3-way match</h2>
      <span className="tag" style={{ color, border: `1px solid ${color}` }}>
        {match.outcome}
      </span>
      <div style={{ marginTop: 14 }}>
        {match.findings.map((finding, i) => (
          <div className="row" key={i}>
            <span className="dim">
              <span className="status-dot" style={{ background: statusColor(finding.status) }} />
              {finding.dimension}
            </span>
            <span className="detail">{finding.detail}</span>
          </div>
        ))}
        {match.line_matches.map((line, i) => (
          <div className="row" key={`l${i}`}>
            <span className="dim">
              <span className="status-dot" style={{ background: statusColor(line.status) }} />
              line
            </span>
            <span className="detail">
              {line.invoice_description} — {line.detail}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
