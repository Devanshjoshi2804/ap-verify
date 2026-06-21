import type { Decision } from "./types";

const DECISION_COLOR: Record<Decision, string> = {
  AUTO_APPROVE: "var(--approve)",
  HUMAN_REVIEW: "var(--review)",
  HOLD: "var(--hold)",
};

const DECISION_LABEL: Record<Decision, string> = {
  AUTO_APPROVE: "Auto-approve",
  HUMAN_REVIEW: "Human review",
  HOLD: "Hold",
};

export function decisionColor(decision: Decision): string {
  return DECISION_COLOR[decision];
}

export function decisionLabel(decision: Decision): string {
  return DECISION_LABEL[decision];
}

export function statusColor(status: string): string {
  if (status === "matched" || status === "agrees" || status === "passed") return "var(--approve)";
  if (status === "partial" || status === "skipped") return "var(--review)";
  return "var(--hold)";
}

export function meterColor(confidence: number): string {
  if (confidence >= 0.9) return "var(--approve)";
  if (confidence >= 0.5) return "var(--review)";
  return "var(--hold)";
}

export function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}
