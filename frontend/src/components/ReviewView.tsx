import { motion } from "framer-motion";
import type { ReactNode } from "react";
import type { ReviewResult } from "../types";
import { Verdict } from "./Verdict";
import { InvoiceCard } from "./InvoiceCard";
import { ConfidenceCard } from "./ConfidenceCard";
import { MatchCard } from "./MatchCard";
import { TraceCard } from "./TraceCard";

const reveal = {
  hidden: { opacity: 0, y: 16 },
  show: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.09, duration: 0.45, ease: [0.22, 1, 0.36, 1] as const },
  }),
};

export function ReviewView({ result }: { result: ReviewResult }) {
  const sections: ReactNode[] = [
    <Verdict result={result} />,
    <div className="grid">
      <InvoiceCard invoice={result.invoice} />
      <ConfidenceCard fields={result.fields} />
    </div>,
    <MatchCard match={result.match} />,
    <TraceCard trace={result.trace} audit={result.audit} consistency={result.consistency} />,
  ];

  return (
    <div className="results">
      {sections.map((section, i) => (
        <motion.div key={i} custom={i} initial="hidden" animate="show" variants={reveal}>
          {section}
        </motion.div>
      ))}
    </div>
  );
}
