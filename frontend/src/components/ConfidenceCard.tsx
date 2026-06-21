import type { FieldConfidence } from "../types";
import { meterColor, pct } from "../format";

export function ConfidenceCard({ fields }: { fields: FieldConfidence[] }) {
  return (
    <div className="panel card">
      <h2>Per-field confidence</h2>
      {fields.map((field) => (
        <div className="field" key={field.field}>
          <span className="name">{field.field}</span>
          <span className="val">{field.value || "—"}</span>
          <div className="meter">
            <span
              style={{ width: pct(field.confidence), background: meterColor(field.confidence) }}
            />
          </div>
          <div className="checks">
            {field.checks.map((check, i) => (
              <span className={`pill ${check.status}`} key={i} title={check.detail}>
                {check.category} {check.status}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
