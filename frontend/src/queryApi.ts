export type PriorityBlock = {
  priority: string;
  confidence: number | null;
  latency_ms: number;
  cost_dollars: number;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
};

export type Retrieved = {
  id: string;
  text: string;
  distance: number | null;
  similarity_score?: number | null;
};

export type AnswerBlock = {
  mode: string;
  answer_text: string;
  contexts: Retrieved[];
  latency_ms: number;
  cost_dollars: number;
};

export type QueryResponseBody = {
  text: string;
  ml_priority: PriorityBlock;
  llm_priority: PriorityBlock;
  rag_answer: AnswerBlock;
  non_rag_answer: AnswerBlock;
};

export function apiBase(): string {
  const raw = import.meta.env.VITE_API_BASE_URL ?? "";
  return raw.replace(/\/$/, "");
}

export function queryUrl(): string {
  const base = apiBase();
  return base ? `${base}/query` : "/query";
}

export function healthUrl(): string {
  const base = apiBase();
  return base ? `${base}/health` : "/health";
}

export function costPerTenThousand(costPerRequest: number): number {
  return costPerRequest * 10_000;
}

export function formatCallCostUsd(dollars: number, isMl: boolean): string {
  if (isMl || dollars === 0) return "$0.00";
  const cents = dollars * 100;
  if (cents < 1) return `~$${dollars.toFixed(5)} (~${cents.toFixed(3)}¢)`;
  return `$${dollars.toFixed(4)}`;
}

export function priorityUrgent(priority: string): boolean {
  return priority.toLowerCase() === "urgent";
}
