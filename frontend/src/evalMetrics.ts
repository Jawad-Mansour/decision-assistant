/**
 * Optional hold-out metrics from `notebook.ipynb` (paste into frontend `.env` after you run evaluation).
 * The `/query` API does not return test-set accuracy — only live latency and cost.
 */
function envStr(key: string): string | null {
  const v = import.meta.env[key];
  if (typeof v !== "string") return null;
  const t = v.trim();
  return t.length ? t : null;
}

export function mlHoldoutAccuracy(): string | null {
  return envStr("VITE_ML_HOLDOUT_ACCURACY");
}

export function llmHoldoutAccuracy(): string | null {
  return envStr("VITE_LLM_HOLDOUT_ACCURACY");
}

export function mlHoldoutF1(): string | null {
  return envStr("VITE_ML_HOLDOUT_F1");
}

export function embeddingDims(): string | null {
  return envStr("VITE_EMBEDDING_DIMS");
}
