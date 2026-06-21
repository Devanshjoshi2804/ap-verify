import type { ReviewResult } from "../types";
import { decisionColor, decisionLabel, pct } from "../format";

export function Verdict({ result }: { result: ReviewResult }) {
  const color = decisionColor(result.decision);
  return (
    <div className="panel verdict" style={{ ["--accent" as string]: color }}>
      <div className="stamp" style={{ ["--accent" as string]: color }}>
        {decisionLabel(result.decision)}
      </div>
      <div className="gauge">
        <div className="pct" style={{ color }}>
          {pct(result.overall_confidence)}
          <span style={{ fontSize: 13, color: "var(--text-faint)", marginLeft: 8 }}>
            extraction confidence
          </span>
        </div>
        <ul className="reasons">
          {result.reasons.map((reason, i) => (
            <li key={i}>{reason}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
