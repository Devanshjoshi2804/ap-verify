import type { InvoiceData } from "../types";

export function InvoiceCard({ invoice }: { invoice: InvoiceData }) {
  const amount = (value: string) => `${invoice.currency} ${value}`;
  return (
    <div className="panel card">
      <h2>Extracted invoice</h2>
      <dl className="kv">
        <dt>Vendor</dt>
        <dd>{invoice.vendor_name}</dd>
        <dt>GSTIN</dt>
        <dd>{invoice.vendor_gstin ?? "—"}</dd>
        <dt>Invoice no.</dt>
        <dd>{invoice.invoice_number}</dd>
        <dt>Date</dt>
        <dd>{invoice.invoice_date}</dd>
        <dt>PO ref</dt>
        <dd>{invoice.purchase_order_ref ?? "—"}</dd>
        <dt>Subtotal</dt>
        <dd className="amount">{amount(invoice.subtotal)}</dd>
        <dt>Tax</dt>
        <dd className="amount">{amount(invoice.tax)}</dd>
        <dt>Total</dt>
        <dd className="amount">{amount(invoice.total)}</dd>
      </dl>
    </div>
  );
}
