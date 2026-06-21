import type { ReviewResult } from "./types";
import demoResults from "./demo/results.json";

/** Static demo (Vercel): no backend, no API keys. Bundled sample invoices return
 * precomputed real results; live uploads are directed to the local pipeline. */
export const DEMO = import.meta.env.VITE_DEMO === "true";

const SAMPLES = demoResults as unknown as Record<string, ReviewResult>;
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export interface ReviewOptions {
  audit: boolean;
  crossCheck: boolean;
}

export async function reviewInvoice(file: File, options: ReviewOptions): Promise<ReviewResult> {
  if (DEMO) {
    await delay(1100); // let the "verifying" pipeline animation play through
    const canned = SAMPLES[file.name];
    if (!canned) {
      throw new Error(
        "This is a static demo — pick a bundled sample to see a full audit. " +
          "For live uploads, run ap-verify locally (see the README).",
      );
    }
    return canned;
  }

  const body = new FormData();
  body.append("file", file);
  body.append("audit", String(options.audit));
  body.append("cross_check", String(options.crossCheck));

  const response = await fetch("/api/review", { method: "POST", body });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Review failed (${response.status})`);
  }
  return response.json() as Promise<ReviewResult>;
}

export async function fetchSamples(): Promise<string[]> {
  if (DEMO) return Object.keys(SAMPLES);
  const response = await fetch("/api/samples");
  if (!response.ok) return [];
  return response.json() as Promise<string[]>;
}

export async function loadSample(name: string): Promise<File> {
  if (DEMO) return new File([], name, { type: "application/pdf" });
  const response = await fetch(`/api/samples/${name}`);
  if (!response.ok) throw new Error(`Could not load sample ${name}`);
  const blob = await response.blob();
  return new File([blob], name, { type: "application/pdf" });
}
