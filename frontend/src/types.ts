export type Decision = "AUTO_APPROVE" | "HOLD" | "HUMAN_REVIEW";

export interface Check {
  category: string;
  status: "passed" | "failed" | "skipped";
  detail: string;
}

export interface FieldConfidence {
  field: string;
  value: string;
  confidence: number;
  checks: Check[];
}

export interface LineItem {
  description: string;
  quantity: number;
  unit_price: string;
  line_total: string;
  hsn_sac: string | null;
}

export interface InvoiceData {
  vendor_name: string;
  vendor_gstin: string | null;
  invoice_number: string;
  invoice_date: string;
  currency: string;
  subtotal: string;
  tax: string;
  total: string;
  purchase_order_ref: string | null;
  line_items: LineItem[];
}

export interface MatchFinding {
  dimension: string;
  status: string;
  detail: string;
}

export interface LineMatch {
  invoice_description: string;
  status: string;
  detail: string;
}

export interface MatchData {
  outcome: string;
  findings: MatchFinding[];
  line_matches: LineMatch[];
}

export interface TraceEntry {
  step: string;
  detail: string;
  duration_ms: number;
}

export interface AuditVerdict {
  field: string;
  trustworthy: boolean;
  confidence: number;
  reason: string;
}

export interface ConsistencyComparison {
  field: string;
  agreement: "agrees" | "differs";
  primary: string;
  secondary: string;
}

export interface ReviewResult {
  decision: Decision;
  reasons: string[];
  overall_confidence: number;
  invoice: InvoiceData;
  fields: FieldConfidence[];
  match: MatchData;
  trace: TraceEntry[];
  audit: AuditVerdict[];
  consistency: ConsistencyComparison[];
}
