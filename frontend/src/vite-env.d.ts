/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_ML_HOLDOUT_ACCURACY?: string;
  readonly VITE_LLM_HOLDOUT_ACCURACY?: string;
  readonly VITE_ML_HOLDOUT_F1?: string;
  readonly VITE_EMBEDDING_DIMS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
