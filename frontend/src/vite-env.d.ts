/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "true" in the static Vercel demo: serve precomputed sample results, no backend. */
  readonly VITE_DEMO?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
