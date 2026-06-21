import { useEffect, useState } from "react";
import { fetchSamples, loadSample, reviewInvoice } from "./api";
import type { ReviewResult } from "./types";
import { UploadConsole } from "./components/UploadConsole";
import { ReviewView } from "./components/ReviewView";
import { Working } from "./components/Working";

export function App() {
  const [file, setFile] = useState<File | null>(null);
  const [audit, setAudit] = useState(false);
  const [crossCheck, setCrossCheck] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ReviewResult | null>(null);
  const [samples, setSamples] = useState<string[]>([]);

  useEffect(() => {
    fetchSamples().then(setSamples).catch(() => setSamples([]));
  }, []);

  const pickSample = async (name: string) => {
    setError(null);
    try {
      setFile(await loadSample(name));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load sample");
    }
  };

  const run = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await reviewInvoice(file, { audit, crossCheck }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <div className="wordmark">ap-verify</div>
          <h1>The accounts-payable agent that knows when it&rsquo;s wrong.</h1>
        </div>
        <p>
          A vision model extracts the invoice; an independent critic and a 3-way match decide
          whether it is safe to pay — never approving a hallucinated total.
        </p>
      </header>

      <UploadConsole
        file={file}
        onFile={(f) => {
          setFile(f);
          setError(null);
        }}
        audit={audit}
        crossCheck={crossCheck}
        onAudit={setAudit}
        onCrossCheck={setCrossCheck}
        onRun={run}
        busy={busy}
        samples={samples}
        onSample={pickSample}
      />

      {busy && <Working />}

      {!busy && error && <div className="error">{error}</div>}

      {!busy && result && <ReviewView result={result} />}
    </div>
  );
}
