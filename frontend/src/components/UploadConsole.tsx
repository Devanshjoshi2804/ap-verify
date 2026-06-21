import { useRef, useState } from "react";
import type { DragEvent } from "react";

interface Props {
  file: File | null;
  onFile: (file: File) => void;
  audit: boolean;
  crossCheck: boolean;
  onAudit: (value: boolean) => void;
  onCrossCheck: (value: boolean) => void;
  onRun: () => void;
  busy: boolean;
  samples: string[];
  onSample: (name: string) => void;
}

export function UploadConsole(props: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const accept = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files[0];
    if (dropped) props.onFile(dropped);
  };

  return (
    <section className="console">
      <div
        className={`panel dropzone${dragging ? " is-active" : ""}${props.file ? " has-file" : ""}`}
        role="button"
        tabIndex={0}
        aria-label="Upload an invoice: drop a file here, or activate to browse"
        onClick={() => input.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            input.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={accept}
      >
        <span className="eyebrow">Document</span>
        {props.file ? (
          <span className="filename">{props.file.name}</span>
        ) : (
          <span className="filename">Drop an invoice, or click to browse</span>
        )}
        <span className="hint">PDF, PNG or JPG — a single supplier invoice.</span>
        {props.samples.length > 0 && (
          <div className="samples" onClick={(e) => e.stopPropagation()}>
            <span className="samples-label">or try a sample</span>
            {props.samples.map((name) => (
              <button
                key={name}
                className="chip"
                type="button"
                onClick={() => props.onSample(name)}
              >
                {name.replace(/\.pdf$/, "")}
              </button>
            ))}
          </div>
        )}
        <input
          ref={input}
          type="file"
          accept=".pdf,.png,.jpg,.jpeg"
          hidden
          onChange={(e) => {
            const picked = e.target.files?.[0];
            if (picked) props.onFile(picked);
          }}
        />
      </div>

      <div className="panel controls">
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <label className="toggle">
            <input
              type="checkbox"
              checked={props.audit}
              onChange={(e) => props.onAudit(e.target.checked)}
            />
            <span>
              <div className="label">LLM auditor</div>
              <div className="sub">groq · low-confidence fields</div>
            </span>
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={props.crossCheck}
              onChange={(e) => props.onCrossCheck(e.target.checked)}
            />
            <span>
              <div className="label">Self-consistency</div>
              <div className="sub">mistral · second extraction</div>
            </span>
          </label>
        </div>
        <button className="run" disabled={!props.file || props.busy} onClick={props.onRun}>
          {props.busy ? (
            <>
              <span className="run-spinner" aria-hidden="true" />
              Verifying…
            </>
          ) : (
            "Run verification"
          )}
        </button>
      </div>
    </section>
  );
}
