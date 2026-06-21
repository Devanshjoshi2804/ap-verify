import type { ReviewResult } from "./types";

export interface ReviewOptions {
  audit: boolean;
  crossCheck: boolean;
}

export async function reviewInvoice(file: File, options: ReviewOptions): Promise<ReviewResult> {
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
  const response = await fetch("/api/samples");
  if (!response.ok) return [];
  return response.json() as Promise<string[]>;
}

export async function loadSample(name: string): Promise<File> {
  const response = await fetch(`/api/samples/${name}`);
  if (!response.ok) throw new Error(`Could not load sample ${name}`);
  const blob = await response.blob();
  return new File([blob], name, { type: "application/pdf" });
}
