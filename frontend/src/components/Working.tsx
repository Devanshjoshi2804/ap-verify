const STEPS = [
  "Vision extraction",
  "Critic — checks & confidence",
  "3-way match",
  "Decision",
];

/** What the user sees while a verification is in flight: the pipeline narrating
 * itself, plus skeletons shaped like the result cards — so the wait reads as the
 * agent working, not a dead screen. */
export function Working() {
  return (
    <div className="results" aria-busy="true" aria-live="polite">
      <div className="panel working">
        <div className="working-head">
          <span className="spinner" aria-hidden="true" />
          <span>Verifying invoice…</span>
        </div>
        <ul className="pipeline">
          {STEPS.map((step, i) => (
            <li key={step} style={{ animationDelay: `${i * 0.18}s` }}>
              {step}
            </li>
          ))}
        </ul>
      </div>
      <div className="grid">
        {[0, 1].map((i) => (
          <div className="panel card skeleton-card" key={i}>
            <div className="sk sk-title" />
            <div className="sk sk-line" />
            <div className="sk sk-line" />
            <div className="sk sk-line short" />
          </div>
        ))}
      </div>
    </div>
  );
}
