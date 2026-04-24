import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import packageJson from "../package.json";
import {
  Activity,
  BarChart3,
  BookOpen,
  Brain,
  CheckCircle2,
  Clock,
  Database,
  Headset,
  Heart,
  History,
  Layers,
  RefreshCw,
  ShieldAlert,
  LayoutDashboard,
  ListOrdered,
  Loader2,
  Menu,
  MessageSquare,
  MessageSquarePlus,
  Scale,
  Search,
  Send,
  Sparkles,
  Trash2,
  X,
  Zap,
  type LucideIcon,
} from "lucide-react";
import type { PriorityBlock, QueryResponseBody, Retrieved } from "./queryApi";
import {
  apiBase,
  costPerTenThousand,
  formatCallCostUsd,
  healthUrl,
  priorityUrgent,
  queryUrl,
} from "./queryApi";
import { llmHoldoutAccuracy, mlHoldoutAccuracy } from "./evalMetrics";

type NavId =
  | "dashboard"
  | "model-comparison"
  | "knowledge"
  | "history"
  | "system-health";

function cn(...parts: (string | false | undefined | null)[]) {
  return parts.filter(Boolean).join(" ");
}

/* ─────────────────────────────── Chatbot brand ──────────────────────────── */

const BRAND_NAME = "Zap";
const BRAND_TAGLINE = "Decision Intelligence Assistant";

/* ─────────────────────────────── History store ──────────────────────────── */

type HistoryEntry = {
  id: string;
  ts: number;
  text: string;
  data: QueryResponseBody;
};

const HISTORY_KEY = "triage.history.v1";
const HISTORY_MAX = 50;

function loadHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

function saveHistory(entries: HistoryEntry[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, HISTORY_MAX)));
  } catch {
    /* quota or privacy mode — ignore */
  }
}

function formatRelativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const s = Math.round(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.round(h / 24);
  return `${d}d ago`;
}

function formatClockTime(ts: number): string {
  try {
    return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "—";
  }
}

function makeId(): string {
  try {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  } catch {
    /* fall through */
  }
  return `h_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

const EXAMPLE_QUERIES = [
  "Refund still not processed after 10 days",
  "Unable to verify email for password reset",
  "Agent promised a callback but never called",
];

const PIPELINE_STEPS = [
  { n: 1, label: "Ask", detail: "User query" },
  { n: 2, label: "Retrieve", detail: "Top-k from vector store" },
  { n: 3, label: "Generate", detail: "RAG + non-RAG" },
  { n: 4, label: "Features", detail: "Engineered inputs" },
  { n: 5, label: "ML priority", detail: "Trained classifier" },
  { n: 6, label: "LLM priority", detail: "Zero-shot triage" },
  { n: 7, label: "Compare", detail: "Quality · latency · cost" },
];

/* ──────────────────────────────── Brand logo ──────────────────────────────── */

function BrandMark({ size = 36, animate = false }: { size?: number; animate?: boolean }) {
  return (
    <div
      className={cn(
        "relative flex items-center justify-center rounded-2xl brand-gradient text-white shadow-glow",
        animate && "animate-logo-float",
      )}
      style={{ width: size, height: size }}
      aria-hidden
    >
      <Headset size={Math.round(size * 0.55)} strokeWidth={2.1} aria-hidden />
      <span
        className="pointer-events-none absolute -bottom-0.5 -right-0.5 h-2 w-2 rounded-full bg-emerald-400 ring-2 ring-white"
        aria-hidden
      />
    </div>
  );
}

/* ─────────────────────────────── Small atoms ──────────────────────────────── */

function PriorityBadge({ priority }: { priority: string }) {
  const urgent = priorityUrgent(priority);
  const label = priority.toUpperCase();
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-bold uppercase tracking-wide",
        urgent ? "bg-red-50 text-skin-urgent ring-1 ring-red-200" : "bg-emerald-50 text-skin-normal ring-1 ring-emerald-200",
      )}
    >
      {label}
    </span>
  );
}

function ConfidenceGauge({ value }: { value: number | null }) {
  const pct = value != null ? Math.min(100, Math.max(0, value * 100)) : 0;
  return (
    <div className="mt-3">
      <div className="mb-1 flex justify-between text-xs font-medium text-skin-muted">
        <span>Confidence</span>
        <span>{value != null ? `${pct.toFixed(1)}%` : "—"}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-200">
        <div
          className="h-full rounded-full bg-skin-primary transition-all duration-700"
          style={{ width: value != null ? `${pct}%` : "0%" }}
        />
      </div>
    </div>
  );
}

function LlmCostMeter({ dollars }: { dollars: number }) {
  const cap = 0.25;
  const pct = Math.min(100, (dollars / cap) * 100);
  return (
    <div className="mt-3 rounded-lg border border-skin-border bg-slate-50/60 p-3">
      <div className="mb-1 text-xs font-semibold text-skin-muted">API cost (this call)</div>
      <div className="text-sm font-bold text-skin-text">{formatCallCostUsd(dollars, false)}</div>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-200">
        <div className="h-full rounded-full bg-amber-500/90 transition-all duration-700" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function CardShell({
  title,
  subtitle,
  icon: Icon,
  children,
  className,
}: {
  title: string;
  subtitle: string;
  icon: LucideIcon;
  children: ReactNode;
  className?: string;
}) {
  return (
    <article
      className={cn(
        "flex min-h-[200px] flex-col rounded-2xl border border-skin-border bg-skin-card shadow-card animate-fade-in-up",
        className,
      )}
    >
      <header className="flex items-start gap-3 border-b border-skin-border px-4 py-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-indigo-50 text-skin-primary">
          <Icon className="h-5 w-5" strokeWidth={1.75} />
        </div>
        <div className="min-w-0">
          <h2 className="text-sm font-bold text-skin-text">{title}</h2>
          <p className="text-xs text-skin-muted">{subtitle}</p>
        </div>
      </header>
      <div className="flex flex-1 flex-col px-4 py-3">{children}</div>
    </article>
  );
}

/* ───────────────────────────── Results blocks ─────────────────────────────── */

function RagGroundingAlert({ contexts }: { contexts: Retrieved[] }) {
  if (!contexts.length) return null;
  const maxSim = Math.max(...contexts.map((c) => c.similarity_score ?? 0));
  if (maxSim >= 0.35) return null;
  return (
    <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-relaxed text-amber-950">
      <strong>Low retrieval match:</strong> best cosine similarity is {maxSim.toFixed(3)}. The RAG branch may be weakly
      grounded — compare carefully with the non-RAG answer.
    </div>
  );
}

function FourWayMetricsTable({
  ml,
  llm,
  rag,
  nonRag,
}: {
  ml: PriorityBlock;
  llm: PriorityBlock;
  rag: NonNullable<QueryResponseBody["rag_answer"]>;
  nonRag: NonNullable<QueryResponseBody["non_rag_answer"]>;
}) {
  const mlAcc = mlHoldoutAccuracy();
  const llmAcc = llmHoldoutAccuracy();
  const ragPeek = (rag.answer_text || "").slice(0, 72) + ((rag.answer_text || "").length > 72 ? "…" : "");
  const nonPeek = (nonRag.answer_text || "").slice(0, 72) + ((nonRag.answer_text || "").length > 72 ? "…" : "");

  const rows = [
    {
      branch: "RAG answer",
      role: "LLM + retrieved tickets",
      summary: ragPeek || "—",
      accuracy: "— (qualitative)",
      ms: rag.latency_ms,
      cost: rag.cost_dollars,
      isMl: false,
    },
    {
      branch: "Non-RAG answer",
      role: "LLM only",
      summary: nonPeek || "—",
      accuracy: "— (qualitative)",
      ms: nonRag.latency_ms,
      cost: nonRag.cost_dollars,
      isMl: false,
    },
    {
      branch: "ML priority",
      role: "Random Forest (features)",
      summary: ml.priority,
      accuracy: mlAcc ?? "— (set env)",
      ms: ml.latency_ms,
      cost: ml.cost_dollars,
      isMl: true,
    },
    {
      branch: "LLM zero-shot",
      role: "API triage prompt",
      summary: llm.priority,
      accuracy: llmAcc ?? "— (set env)",
      ms: llm.latency_ms,
      cost: llm.cost_dollars,
      isMl: false,
    },
  ];

  return (
    <section
      className="rounded-2xl border border-skin-border bg-skin-card shadow-card animate-fade-in-up"
      aria-label="Four-way comparison"
    >
      <div className="border-b border-skin-border px-4 py-3">
        <h2 className="text-sm font-bold text-skin-text">Four-way comparison</h2>
        <p className="mt-1 text-xs text-skin-muted">
          Same query → RAG vs non-RAG answers, ML vs LLM priority. Accuracy columns read from optional env values (set
          after offline evaluation), never fabricated.
        </p>
      </div>
      <div className="overflow-x-auto p-2 sm:p-4">
        <table className="w-full min-w-[640px] border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-skin-border text-[11px] font-bold uppercase tracking-wide text-skin-muted">
              <th className="px-2 py-2">Output</th>
              <th className="px-2 py-2">Mechanism</th>
              <th className="px-2 py-2">Preview / label</th>
              <th className="px-2 py-2">Hold-out acc.*</th>
              <th className="px-2 py-2">Latency</th>
              <th className="px-2 py-2">Cost (call)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.branch} className="border-b border-slate-100 last:border-0">
                <td className="px-2 py-3 font-semibold text-skin-text">{r.branch}</td>
                <td className="px-2 py-3 text-xs text-skin-muted">{r.role}</td>
                <td className="max-w-[220px] px-2 py-3 text-xs text-skin-text">{r.summary}</td>
                <td className="whitespace-nowrap px-2 py-3 font-mono text-xs text-skin-text">{r.accuracy}</td>
                <td className="whitespace-nowrap px-2 py-3 font-mono text-xs text-skin-muted">{r.ms.toFixed(1)} ms</td>
                <td className="whitespace-nowrap px-2 py-3 font-mono text-xs font-semibold text-skin-text">
                  {formatCallCostUsd(r.cost, r.isMl)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 px-1 text-[10px] leading-snug text-skin-muted">
          *Hold-out accuracy is static metadata from your notebook, not computed per request.
        </p>
      </div>
    </section>
  );
}

function RetrievedSourcesPanel({ contexts }: { contexts: Retrieved[] | null }) {
  return (
    <section className="flex min-h-[240px] flex-col rounded-2xl border border-skin-border bg-skin-card shadow-card animate-fade-in-up">
      <div className="border-b border-skin-border px-4 py-3">
        <h2 className="flex items-center gap-2 text-sm font-bold text-skin-text">
          <Search className="h-4 w-4 text-skin-primary" strokeWidth={2} />
          Source panel — retrieved tickets
        </h2>
        <p className="mt-1 text-xs leading-relaxed text-skin-muted">
          Cosine similarity in <span className="font-mono">[0, 1]</span> (higher = closer). These are the exact rows the
          RAG branch grounded its answer on.
        </p>
      </div>
      <div className="scrollbar-thin flex-1 overflow-y-auto p-4">
        {!contexts || contexts.length === 0 ? (
          <p className="text-sm text-skin-muted">No retrieval results for this query.</p>
        ) : (
          <ol className="space-y-3">
            {contexts.map((c, i) => (
              <li key={`${c.id}-${i}`} className="rounded-xl border border-skin-border bg-slate-50/80 p-3">
                <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                  <span className="rounded-md bg-indigo-50 px-2 py-0.5 font-mono font-bold text-skin-primary">
                    sim {(c.similarity_score != null ? c.similarity_score : 0).toFixed(4)}
                  </span>
                  {c.distance != null ? <span className="text-skin-muted">d {c.distance.toFixed(4)}</span> : null}
                  <span className="text-skin-muted">id {c.id}</span>
                </div>
                <p className="text-sm leading-relaxed text-skin-text">
                  {(c.text || "").slice(0, 520)}
                  {(c.text || "").length > 520 ? "…" : ""}
                </p>
              </li>
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}

function DeploymentTable({ ml, llm, anchorId = "deploy-block" }: { ml: PriorityBlock | null; llm: PriorityBlock | null; anchorId?: string }) {
  if (!ml || !llm) {
    return (
      <section
        id={anchorId}
        className="flex min-h-[200px] flex-col rounded-2xl border border-dashed border-skin-border bg-white/50 p-6 shadow-card"
      >
        <h2 className="text-sm font-bold text-skin-text">Deploy at ~10k tickets / hour?</h2>
        <p className="mt-2 text-sm text-skin-muted">
          Run a query to compare ML ($0 API) vs LLM token cost × 10,000 — the core deployment tradeoff.
        </p>
      </section>
    );
  }
  const rows = [
    {
      name: "ML (Random Forest, on-device)",
      prediction: ml.priority,
      latency: `${ml.latency_ms.toFixed(2)} ms`,
      cost10k: costPerTenThousand(ml.cost_dollars),
      acc: mlHoldoutAccuracy() ?? "—",
    },
    {
      name: "LLM zero-shot (API)",
      prediction: llm.priority,
      latency: `${llm.latency_ms.toFixed(2)} ms`,
      cost10k: costPerTenThousand(llm.cost_dollars),
      acc: llmHoldoutAccuracy() ?? "—",
    },
  ];
  return (
    <section
      id={anchorId}
      className="flex flex-col rounded-2xl border border-skin-border bg-skin-card shadow-card animate-fade-in-up"
    >
      <div className="border-b border-skin-border px-4 py-3">
        <h2 className="text-sm font-bold text-skin-text">Priority predictors — scale economics</h2>
        <p className="mt-1 text-xs text-skin-muted">Wall-clock and list-price token estimates from this run.</p>
      </div>
      <div className="overflow-x-auto p-4">
        <table className="w-full min-w-[480px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-skin-border text-left text-[11px] font-bold uppercase tracking-wide text-skin-muted">
              <th className="pb-2 pr-3">Predictor</th>
              <th className="pb-2 pr-3">Prediction</th>
              <th className="pb-2 pr-3">Hold-out acc.*</th>
              <th className="pb-2 pr-3">Latency</th>
              <th className="pb-2">Est. $ / 10k tickets</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name} className="border-b border-slate-100 last:border-0">
                <td className="py-3 pr-3 align-middle text-skin-text">{r.name}</td>
                <td className="py-3 pr-3 align-middle">
                  <PriorityBadge priority={r.prediction} />
                </td>
                <td className="py-3 pr-3 align-middle font-mono text-xs text-skin-text">{r.acc}</td>
                <td className="py-3 pr-3 align-middle font-mono text-xs text-skin-muted">{r.latency}</td>
                <td className="py-3 align-middle font-mono text-sm font-semibold text-skin-text">${r.cost10k.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 text-[10px] text-skin-muted">*From optional <code className="text-[10px]">VITE_*</code> env — not live.</p>
      </div>
    </section>
  );
}

/* ───────────────────────────── Info subpages ──────────────────────────────── */

function PipelinePage() {
  return (
    <section className="rounded-2xl border border-skin-border bg-skin-card p-5 shadow-card animate-fade-in-up">
      <h2 className="flex items-center gap-2 text-base font-bold text-skin-text">
        <ListOrdered className="h-5 w-5 text-skin-primary" strokeWidth={2} />
        Request pipeline
      </h2>
      <ol className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {PIPELINE_STEPS.map((s) => (
          <li key={s.n} className="rounded-xl border border-skin-border bg-slate-50/70 px-3 py-3">
            <span className="font-mono text-[10px] font-bold text-skin-primary">Step {s.n}</span>
            <div className="mt-0.5 text-sm font-semibold text-skin-text">{s.label}</div>
            <div className="text-xs text-skin-muted">{s.detail}</div>
          </li>
        ))}
      </ol>
    </section>
  );
}

function DataAndRagPage() {
  return (
    <section className="space-y-4">
      <header className="rounded-2xl border border-skin-border bg-skin-card p-5 shadow-card animate-fade-in-up">
        <div className="flex items-start gap-3">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl brand-gradient text-white shadow-glow">
            <BookOpen className="h-5 w-5" strokeWidth={2} />
          </span>
          <div>
            <h1 className="text-lg font-bold text-skin-text">Data &amp; RAG</h1>
            <p className="mt-0.5 text-xs text-skin-muted">
              How the corpus was built, how labels were assigned, and why you can trust (or distrust) the retrieved
              sources.
            </p>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-skin-border bg-skin-card p-5 shadow-card animate-fade-in-up">
          <h2 className="flex items-center gap-2 text-base font-bold text-skin-text">
            <Database className="h-5 w-5 text-skin-primary" strokeWidth={1.75} />
            Training data &amp; labels
          </h2>
          <ul className="mt-3 space-y-2 text-sm leading-relaxed text-skin-muted">
            <li className="flex gap-2">
              <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-skin-primary" />
              <span>
                Source:{" "}
                <a
                  className="text-skin-primary underline"
                  href="https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter"
                  target="_blank"
                  rel="noreferrer"
                >
                  Customer Support on Twitter
                </a>{" "}
                — real public support conversations.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-skin-primary" />
              <span>Tweets are cleaned, chunked, and embedded into the vector store for retrieval.</span>
            </li>
            <li className="flex gap-2">
              <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-skin-primary" />
              <span>
                <span className="font-semibold text-skin-text">Weak supervision:</span> priority labels come from an
                explicit rule (keywords, punctuation, length). Risk acknowledged — the ML model may partly learn the
                rule, not pure urgency.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-skin-primary" />
              <span>Engineered tabular features power the ML baseline; models compared on a proper hold-out split.</span>
            </li>
          </ul>
        </div>

        <div className="rounded-2xl border border-skin-border bg-skin-card p-5 shadow-card animate-fade-in-up">
          <h2 className="flex items-center gap-2 text-base font-bold text-skin-text">
            <Search className="h-5 w-5 text-skin-primary" strokeWidth={1.75} />
            RAG honesty
          </h2>
          <ul className="mt-3 space-y-2 text-sm leading-relaxed text-skin-muted">
            <li className="flex gap-2">
              <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-skin-accent" />
              <span>Sources are retrieved past tickets, not fabricated citations.</span>
            </li>
            <li className="flex gap-2">
              <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-skin-accent" />
              <span>Similarity scores are cosine similarity from the vector store — auditable, not guessed.</span>
            </li>
            <li className="flex gap-2">
              <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-skin-accent" />
              <span>
                Compare <span className="font-semibold text-skin-text">RAG vs non-RAG</span> to see grounded synthesis
                vs parametric-only answers.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-skin-accent" />
              <span>When the best match has low similarity, a visible warning discourages over-trusting the result.</span>
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
}

/* ─────────────────────────── System health page ─────────────────────────── */

function StatusTile({
  label,
  value,
  tone,
  icon: Icon,
  sub,
}: {
  label: string;
  value: string;
  tone: "ok" | "warn" | "bad" | "neutral";
  icon: LucideIcon;
  sub?: string;
}) {
  const toneClass =
    tone === "ok"
      ? "text-emerald-600 bg-emerald-50 ring-emerald-100"
      : tone === "warn"
        ? "text-amber-600 bg-amber-50 ring-amber-100"
        : tone === "bad"
          ? "text-red-600 bg-red-50 ring-red-100"
          : "text-skin-primary bg-indigo-50 ring-indigo-100";
  return (
    <div className="flex items-start gap-3 rounded-xl border border-skin-border bg-white p-3">
      <span className={cn("flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ring-1", toneClass)}>
        <Icon className="h-4 w-4" strokeWidth={2} />
      </span>
      <div className="min-w-0">
        <div className="text-[10px] font-bold uppercase tracking-wider text-skin-muted">{label}</div>
        <div className="truncate text-sm font-semibold text-skin-text">{value}</div>
        {sub ? <div className="mt-0.5 truncate text-[11px] text-skin-muted">{sub}</div> : null}
      </div>
    </div>
  );
}

function SystemHealthPanel({
  loading,
  ok,
  latencyMs,
  checkedAt,
  body,
  historyCount,
  onRefresh,
}: {
  loading: boolean;
  ok: boolean | null;
  latencyMs: number | null;
  checkedAt: number | null;
  body: string | null;
  historyCount: number;
  onRefresh: () => void;
}) {
  let parsed: Record<string, unknown> | null = null;
  if (body) {
    try {
      parsed = JSON.parse(body);
    } catch {
      parsed = null;
    }
  }

  const apiHost = apiBase() || "vite proxy · /health";
  const latencyText = latencyMs === null ? "—" : `${latencyMs} ms`;
  const latencyTone: "ok" | "warn" | "bad" | "neutral" =
    latencyMs === null ? "neutral" : latencyMs < 150 ? "ok" : latencyMs < 500 ? "warn" : "bad";
  const statusString = typeof parsed?.status === "string" ? (parsed.status as string) : null;
  const serverStamp = typeof parsed?.timestamp_utc === "string" ? (parsed.timestamp_utc as string) : null;

  return (
    <>
      <section
        className={cn(
          "relative overflow-hidden rounded-2xl border bg-skin-card p-5 shadow-card animate-fade-in-up",
          ok === null ? "border-skin-border" : ok ? "border-emerald-200" : "border-red-200",
        )}
      >
        <div className="pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full bg-gradient-to-br from-indigo-400/10 to-violet-400/10 blur-2xl" aria-hidden />
        <div className="relative flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="relative">
              <span
                className={cn(
                  "flex h-14 w-14 items-center justify-center rounded-2xl text-white shadow-glow",
                  ok === null ? "bg-slate-400" : ok ? "bg-gradient-to-br from-emerald-400 to-emerald-600" : "bg-gradient-to-br from-red-400 to-red-600",
                )}
              >
                {loading ? (
                  <Loader2 className="h-6 w-6 animate-spin" />
                ) : ok ? (
                  <CheckCircle2 className="h-7 w-7" strokeWidth={2.2} />
                ) : ok === false ? (
                  <ShieldAlert className="h-7 w-7" strokeWidth={2.2} />
                ) : (
                  <Activity className="h-7 w-7" strokeWidth={2.2} />
                )}
              </span>
              {ok ? (
                <span className="absolute inset-0 -z-10 animate-ping rounded-2xl bg-emerald-400/40" aria-hidden />
              ) : null}
            </div>
            <div>
              <h2 className="text-lg font-bold text-skin-text">
                {loading ? "Pinging API…" : ok ? "All systems go" : ok === false ? "API unreachable" : "Status unknown"}
              </h2>
              <p className="mt-0.5 text-xs text-skin-muted">
                {checkedAt ? `Checked ${formatRelativeTime(checkedAt)} · ${formatClockTime(checkedAt)}` : "Not yet checked"}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onRefresh}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-xl border border-skin-border bg-white px-3 py-2 text-xs font-semibold text-skin-text transition hover:border-skin-primary hover:bg-indigo-50 hover:text-skin-primary disabled:cursor-not-allowed disabled:opacity-60"
          >
            <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} strokeWidth={2} />
            {loading ? "Checking" : "Re-check"}
          </button>
        </div>

        <div className="relative mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <StatusTile
            label="FastAPI"
            value={ok === null ? "—" : ok ? "Healthy" : "Unreachable"}
            sub={statusString ? `status="${statusString}"` : "GET /health"}
            tone={ok === null ? "neutral" : ok ? "ok" : "bad"}
            icon={Activity}
          />
          <StatusTile
            label="Round-trip"
            value={latencyText}
            sub={latencyMs === null ? "no measurement" : latencyMs < 150 ? "snappy" : latencyMs < 500 ? "ok" : "slow"}
            tone={latencyTone}
            icon={Zap}
          />
          <StatusTile
            label="Server time"
            value={serverStamp ? new Date(serverStamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "—"}
            sub={serverStamp ? "UTC payload" : "not reported"}
            tone="neutral"
            icon={Clock}
          />
          <StatusTile
            label="Logged queries"
            value={`${historyCount}`}
            sub="samples in local history"
            tone="neutral"
            icon={History}
          />
        </div>
      </section>

      <section className="rounded-2xl border border-skin-border bg-skin-card shadow-card animate-fade-in-up">
        <header className="border-b border-skin-border px-5 py-3">
          <h3 className="text-sm font-bold text-skin-text">Endpoint</h3>
          <p className="mt-0.5 text-xs text-skin-muted">
            Hitting <code className="rounded bg-slate-100 px-1 py-0.5 text-[11px]">GET {healthUrl()}</code>
            {apiBase() ? (
              <>
                {" "}· base <code className="rounded bg-slate-100 px-1 py-0.5 text-[11px]">{apiHost}</code>
              </>
            ) : (
              <>
                {" "}· <span className="italic">resolved via Vite dev proxy</span>
              </>
            )}
          </p>
        </header>
        <div className="p-5">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-[10px] font-bold uppercase tracking-wider text-skin-muted">Raw response</span>
            <span className="text-[10px] text-skin-muted">{body ? `${body.length} chars` : "empty"}</span>
          </div>
          <pre className="scrollbar-thin max-h-72 overflow-auto rounded-xl bg-slate-900 p-4 text-xs leading-relaxed text-slate-100">
{body ?? "(no body)"}
          </pre>
        </div>
      </section>

      <section className="rounded-2xl border border-dashed border-skin-border bg-white/60 p-5 text-xs text-skin-muted animate-fade-in-up">
        <p className="font-semibold text-skin-text">How this page is honest</p>
        <ul className="mt-2 list-inside list-disc space-y-1">
          <li>Latency is measured with <code>performance.now()</code> from this browser — it includes network time.</li>
          <li>No fake services are reported. Only what the backend actually returns from <code>/health</code> is shown.</li>
          <li>Query logs live in your browser (localStorage key <code>triage.history.v1</code>), not on a server.</li>
        </ul>
      </section>
    </>
  );
}

/* ───────────────────────────── History page ────────────────────────────── */

function HistoryPage({
  entries,
  onOpen,
  onDelete,
  onClear,
}: {
  entries: HistoryEntry[];
  onOpen: (e: HistoryEntry) => void;
  onDelete: (id: string) => void;
  onClear: () => void;
}) {
  if (entries.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-skin-border bg-white/50 p-10 text-center animate-fade-in">
        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-indigo-50 text-skin-primary">
          <History className="h-6 w-6" strokeWidth={1.75} />
        </div>
        <h2 className="mt-4 text-base font-bold text-skin-text">No history yet</h2>
        <p className="mt-1 text-sm text-skin-muted">
          Every query you run is logged here — the user text, four-branch outputs, latency and cost — so you can audit
          and re-open any past analysis.
        </p>
      </div>
    );
  }

  return (
    <section className="rounded-2xl border border-skin-border bg-skin-card shadow-card animate-fade-in-up">
      <header className="flex items-center justify-between gap-3 border-b border-skin-border px-5 py-3">
        <div>
          <h2 className="text-sm font-bold text-skin-text">Query history</h2>
          <p className="mt-0.5 text-xs text-skin-muted">
            {entries.length} sample{entries.length === 1 ? "" : "s"} · stored locally in your browser
          </p>
        </div>
        <button
          type="button"
          onClick={onClear}
          className="inline-flex items-center gap-1.5 rounded-lg border border-skin-border bg-white px-2.5 py-1.5 text-xs font-medium text-skin-muted transition hover:border-red-300 hover:bg-red-50 hover:text-red-600"
        >
          <Trash2 className="h-3.5 w-3.5" strokeWidth={2} />
          Clear all
        </button>
      </header>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-skin-border bg-slate-50/70 text-left text-[11px] font-bold uppercase tracking-wide text-skin-muted">
              <th className="px-4 py-2.5">When</th>
              <th className="px-4 py-2.5">Query</th>
              <th className="px-4 py-2.5">ML</th>
              <th className="px-4 py-2.5">LLM</th>
              <th className="px-4 py-2.5 text-right">RAG latency</th>
              <th className="px-4 py-2.5 text-right">LLM $/10k</th>
              <th className="px-4 py-2.5" />
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => {
              const ml = e.data.ml_priority;
              const llm = e.data.llm_priority;
              const rag = e.data.rag_answer;
              return (
                <tr key={e.id} className="border-b border-slate-100 last:border-0 transition hover:bg-slate-50/60">
                  <td className="px-4 py-3 align-middle font-mono text-[11px] text-skin-muted" title={new Date(e.ts).toLocaleString()}>
                    {formatClockTime(e.ts)}
                    <div className="text-[10px] text-skin-muted/80">{formatRelativeTime(e.ts)}</div>
                  </td>
                  <td className="max-w-[280px] px-4 py-3 align-middle text-skin-text">
                    <div className="line-clamp-2 text-sm">{e.text}</div>
                  </td>
                  <td className="px-4 py-3 align-middle">
                    {ml ? <PriorityBadge priority={ml.priority} /> : <span className="text-xs text-skin-muted">—</span>}
                  </td>
                  <td className="px-4 py-3 align-middle">
                    {llm ? <PriorityBadge priority={llm.priority} /> : <span className="text-xs text-skin-muted">—</span>}
                  </td>
                  <td className="px-4 py-3 text-right align-middle font-mono text-xs text-skin-muted">
                    {rag ? `${rag.latency_ms.toFixed(0)} ms` : "—"}
                  </td>
                  <td className="px-4 py-3 text-right align-middle font-mono text-xs font-semibold text-skin-text">
                    {llm ? `$${costPerTenThousand(llm.cost_dollars).toFixed(2)}` : "—"}
                  </td>
                  <td className="px-4 py-3 text-right align-middle">
                    <div className="flex items-center justify-end gap-1">
                      <button
                        type="button"
                        onClick={() => onOpen(e)}
                        className="rounded-lg border border-skin-border bg-white px-2 py-1 text-[11px] font-semibold text-skin-primary transition hover:border-skin-primary hover:bg-indigo-50"
                      >
                        Open
                      </button>
                      <button
                        type="button"
                        onClick={() => onDelete(e.id)}
                        aria-label="Delete entry"
                        className="rounded-lg p-1.5 text-skin-muted transition hover:bg-red-50 hover:text-red-600"
                      >
                        <Trash2 className="h-3.5 w-3.5" strokeWidth={2} />
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* ─────────────────────────── About-project modal ─────────────────────────── */

function AboutProjectModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center px-4 animate-fade-in" role="dialog" aria-modal="true" aria-labelledby="about-modal-title">
      <button
        type="button"
        aria-label="Close"
        onClick={onClose}
        className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm"
      />
      <div className="relative w-full max-w-lg overflow-hidden rounded-2xl border border-skin-border bg-skin-card shadow-glow animate-fade-in-up">
        <div className="flex items-start justify-between gap-3 border-b border-skin-border px-5 py-4">
          <div className="flex items-center gap-3">
            <BrandMark size={36} />
            <div>
              <h2 id="about-modal-title" className="text-base font-bold text-skin-text">
                About {BRAND_NAME}
              </h2>
              <p className="text-xs text-skin-muted">{BRAND_TAGLINE}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-lg p-1.5 text-skin-muted transition hover:bg-slate-100 hover:text-skin-text"
          >
            <X className="h-4 w-4" strokeWidth={2} />
          </button>
        </div>
        <div className="scrollbar-thin max-h-[70vh] overflow-y-auto px-5 py-4 text-sm leading-relaxed text-skin-muted">
          <p>
            <span className="font-semibold text-skin-text">{BRAND_NAME}</span> is a full-stack decision-support system
            that answers customer-support queries and predicts ticket priority — with the math to defend each choice.
          </p>
          <p className="mt-3 font-semibold text-skin-text">Four branches, one call</p>
          <ul className="mt-2 space-y-2 text-xs">
            <li className="flex gap-2">
              <Layers className="mt-0.5 h-4 w-4 shrink-0 text-skin-primary" />
              <span>
                <span className="font-semibold text-skin-text">RAG</span> — LLM + top-k retrieved tickets from the vector
                store (Chroma).
              </span>
            </li>
            <li className="flex gap-2">
              <MessageSquare className="mt-0.5 h-4 w-4 shrink-0 text-skin-primary" />
              <span>
                <span className="font-semibold text-skin-text">Non-RAG</span> — same LLM, no retrieval, for a fair
                grounding comparison.
              </span>
            </li>
            <li className="flex gap-2">
              <BarChart3 className="mt-0.5 h-4 w-4 shrink-0 text-skin-primary" />
              <span>
                <span className="font-semibold text-skin-text">ML priority</span> — Random Forest over engineered
                features. On-device inference, $0 API.
              </span>
            </li>
            <li className="flex gap-2">
              <Brain className="mt-0.5 h-4 w-4 shrink-0 text-skin-primary" />
              <span>
                <span className="font-semibold text-skin-text">LLM zero-shot</span> — direct triage prompt; token cost
                metered honestly.
              </span>
            </li>
          </ul>
          <p className="mt-4 font-semibold text-skin-text">The core question</p>
          <p className="mt-1 text-xs">
            At ~10,000 tickets / hour, which priority predictor would you deploy? The UI reports{" "}
            <span className="font-semibold text-skin-text">quality, latency, and cost</span> for both paths so the
            tradeoff is defensible — not marketed.
          </p>
          <p className="mt-4 font-semibold text-skin-text">Honesty rules</p>
          <ul className="mt-1 list-inside list-disc space-y-1 text-xs">
            <li>No invented metrics — hold-out accuracy comes from your notebook, via <code className="text-[11px]">VITE_*</code> env only.</li>
            <li>Similarity scores are cosine similarity from the vector store (auditable).</li>
            <li>Low-similarity RAG triggers a visible warning instead of silent grounding.</li>
          </ul>
          <p className="mt-4 text-[11px] text-skin-muted">Stack: FastAPI · React · Chroma · Docker Compose.</p>
        </div>
        <div className="flex justify-end border-t border-skin-border bg-slate-50/50 px-5 py-3">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg brand-gradient px-4 py-2 text-xs font-semibold text-white shadow-sm transition hover:brightness-110"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
}

/* ──────────────────────────── Composer & empties ──────────────────────────── */

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="mx-auto flex min-h-[55vh] max-w-2xl flex-col items-center justify-center px-6 text-center animate-fade-in">
      <div className="relative">
        <div className="absolute inset-0 -z-10 rounded-full bg-indigo-400/15 blur-2xl" aria-hidden />
        <BrandMark size={72} animate />
      </div>
      <h2 className="mt-6 text-2xl font-bold tracking-tight text-skin-text sm:text-3xl">
        How can I help you <span className="brand-gradient-text">triage</span> today?
      </h2>
      <p className="mt-3 max-w-md text-sm leading-relaxed text-skin-muted">
        Paste a customer message. One call returns a RAG answer, a non-RAG answer, and two priority predictions — with
        honest latency and cost.
      </p>
      <div className="mt-8 flex flex-wrap justify-center gap-2">
        {EXAMPLE_QUERIES.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onPick(q)}
            className="group inline-flex items-center gap-1.5 rounded-full border border-skin-border bg-white px-3 py-1.5 text-xs font-medium text-skin-muted shadow-sm transition hover:border-skin-primary hover:bg-indigo-50 hover:text-skin-primary"
          >
            <Sparkles className="h-3.5 w-3.5 transition group-hover:scale-110" strokeWidth={2} />
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

/* ────────────────────────────── Main component ────────────────────────────── */

export default function App() {
  const [nav, setNav] = useState<NavId>("dashboard");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<QueryResponseBody | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [healthText, setHealthText] = useState<string | null>(null);
  const [healthOk, setHealthOk] = useState<boolean | null>(null);
  const [healthLatency, setHealthLatency] = useState<number | null>(null);
  const [healthCheckedAt, setHealthCheckedAt] = useState<number | null>(null);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory());

  useEffect(() => {
    saveHistory(history);
  }, [history]);

  const openHistoryEntry = useCallback((entry: HistoryEntry) => {
    setData(entry.data);
    setError(null);
    setLoading(false);
    setInput("");
    setNav("dashboard");
    setMobileMenuOpen(false);
    setTimeout(() => resultsAnchorRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 60);
  }, []);

  const deleteHistoryEntry = useCallback((id: string) => {
    setHistory((h) => h.filter((x) => x.id !== id));
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  const resultsAnchorRef = useRef<HTMLDivElement | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);

  const newAnalysis = useCallback(() => {
    setData(null);
    setError(null);
    setInput("");
    setLoading(false);
    setNav("dashboard");
    setMobileMenuOpen(false);
    setTimeout(() => composerRef.current?.focus(), 50);
  }, []);

  const sendText = useCallback(async (raw: string) => {
    const trimmed = raw.trim().slice(0, 500);
    if (trimmed.length < 1) return;
    setError(null);
    setLoading(true);
    setInput("");
    setNav("dashboard");
    try {
      const res = await fetch(queryUrl(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed, top_k: 5 }),
      });
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const errJson = await res.json();
          if (typeof errJson?.detail === "string") {
            detail = errJson.detail;
          } else if (Array.isArray(errJson?.detail)) {
            detail = errJson.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ");
          }
        } catch {
          /* ignore */
        }
        throw new Error(detail || `HTTP ${res.status}`);
      }
      const body = (await res.json()) as QueryResponseBody;
      setData(body);
      setHistory((h) => [{ id: makeId(), ts: Date.now(), text: trimmed, data: body }, ...h].slice(0, HISTORY_MAX));
      setTimeout(() => resultsAnchorRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 60);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const runQuery = useCallback(() => {
    void sendText(input);
  }, [input, sendText]);

  const checkHealth = useCallback(async () => {
    setHealthLoading(true);
    const started = performance.now();
    try {
      const res = await fetch(healthUrl());
      const text = await res.text();
      setHealthLatency(Math.round(performance.now() - started));
      setHealthOk(res.ok);
      setHealthText(text.slice(0, 1200));
      setHealthCheckedAt(Date.now());
    } catch {
      setHealthLatency(null);
      setHealthOk(false);
      setHealthText("Could not reach /health (is the API running?)");
      setHealthCheckedAt(Date.now());
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    if (nav !== "system-health") return;
    void checkHealth();
  }, [nav, checkHealth]);

  const ml = data?.ml_priority ?? null;
  const llm = data?.llm_priority ?? null;
  const rag = data?.rag_answer ?? null;
  const nonRag = data?.non_rag_answer ?? null;

  const navItems: { id: NavId; label: string; icon: LucideIcon }[] = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "model-comparison", label: "Model comparison", icon: Scale },
    { id: "knowledge", label: "Data & RAG", icon: BookOpen },
    { id: "history", label: "History", icon: History },
    { id: "system-health", label: "System health", icon: Activity },
  ];

  const showDashboardResults = nav === "dashboard" && (loading || error || data);
  const showDashboardEmpty = nav === "dashboard" && !loading && !error && !data;

  return (
    <div className="flex min-h-screen bg-skin-bg text-skin-text">
      <AboutProjectModal open={aboutOpen} onClose={() => setAboutOpen(false)} />
      {mobileMenuOpen ? (
        <button
          type="button"
          className="fixed inset-0 z-30 bg-slate-900/40 md:hidden"
          aria-label="Close menu"
          onClick={() => setMobileMenuOpen(false)}
        />
      ) : null}

      {/* ─────────────────── Sidebar ─────────────────── */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-skin-border bg-skin-card shadow-xl transition-transform duration-200 md:static md:z-0 md:shadow-none",
          mobileMenuOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
        )}
      >
        <div className="flex items-center gap-3 border-b border-skin-border px-4 py-4">
          <BrandMark size={40} />
          <div className="min-w-0">
            <div className="truncate text-base font-bold tracking-tight text-skin-text">{BRAND_NAME}</div>
            <div className="truncate text-xs text-skin-muted">{BRAND_TAGLINE}</div>
          </div>
        </div>

        <div className="p-3">
          <button
            type="button"
            onClick={newAnalysis}
            className="flex w-full items-center justify-center gap-2 rounded-xl brand-gradient py-2.5 text-sm font-semibold text-white shadow-glow transition hover:brightness-110"
          >
            <MessageSquarePlus className="h-4 w-4 shrink-0" strokeWidth={2} />
            New analysis
          </button>
        </div>

        <nav className="flex flex-col gap-0.5 px-2" aria-label="Primary">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => {
                setNav(id);
                setMobileMenuOpen(false);
              }}
              className={cn(
                "flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm font-medium transition-colors",
                nav === id
                  ? "bg-indigo-50 text-skin-primary ring-1 ring-indigo-100"
                  : "text-skin-muted hover:bg-slate-50 hover:text-skin-text",
              )}
            >
              <Icon className="h-5 w-5 shrink-0" strokeWidth={1.75} />
              <span className="truncate">{label}</span>
              {id === "history" && history.length > 0 ? (
                <span className="ml-auto rounded-full bg-indigo-100 px-2 py-0.5 text-[10px] font-bold text-skin-primary">
                  {history.length}
                </span>
              ) : null}
            </button>
          ))}
        </nav>

        <div className="mt-4 flex min-h-0 flex-1 flex-col border-t border-skin-border px-3 pt-3">
          <div className="flex items-center justify-between pb-2">
            <span className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-skin-muted">
              <Clock className="h-3 w-3" strokeWidth={2.25} />
              Recent
            </span>
            {history.length > 0 ? (
              <button
                type="button"
                onClick={() => {
                  setNav("history");
                  setMobileMenuOpen(false);
                }}
                className="text-[10px] font-semibold text-skin-primary hover:underline"
              >
                See all
              </button>
            ) : null}
          </div>
          <div className="scrollbar-thin -mx-1 flex-1 overflow-y-auto pb-2">
            {history.length === 0 ? (
              <p className="px-2 py-1 text-[11px] leading-relaxed text-skin-muted">
                Samples from each query land here — timestamp, ML vs LLM call, latency.
              </p>
            ) : (
              <ul className="space-y-1">
                {history.slice(0, 6).map((e) => {
                  const urgent =
                    (e.data.ml_priority && priorityUrgent(e.data.ml_priority.priority)) ||
                    (e.data.llm_priority && priorityUrgent(e.data.llm_priority.priority));
                  return (
                    <li key={e.id}>
                      <button
                        type="button"
                        onClick={() => openHistoryEntry(e)}
                        className="group flex w-full items-start gap-2 rounded-lg px-2 py-1.5 text-left transition hover:bg-slate-50"
                      >
                        <span
                          className={cn(
                            "mt-1 h-1.5 w-1.5 shrink-0 rounded-full",
                            urgent ? "bg-skin-urgent" : "bg-skin-normal",
                          )}
                          aria-hidden
                        />
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-xs font-medium text-skin-text group-hover:text-skin-primary">
                            {e.text}
                          </span>
                          <span className="mt-0.5 flex items-center gap-1.5 text-[10px] text-skin-muted">
                            <span>{formatRelativeTime(e.ts)}</span>
                            {e.data.ml_priority ? (
                              <span className="font-mono">· ML {e.data.ml_priority.priority}</span>
                            ) : null}
                          </span>
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>

        <div className="flex items-center justify-between border-t border-skin-border p-3 text-[11px] text-skin-muted">
          <div className="flex items-center gap-1.5">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500"></span>
            </span>
            UI ready · <span className="font-mono">v{packageJson.version}</span>
          </div>
          <button
            type="button"
            onClick={() => setAboutOpen(true)}
            aria-label={`About ${BRAND_NAME}`}
            title={`About ${BRAND_NAME}`}
            className="group flex h-7 w-7 items-center justify-center rounded-full text-emerald-500 transition hover:scale-110 hover:bg-emerald-50 hover:text-emerald-600 focus:outline-none focus:ring-2 focus:ring-emerald-300"
          >
            <Heart className="h-4 w-4 fill-current animate-heart-pulse" strokeWidth={2} />
          </button>
        </div>
      </aside>

      {/* ─────────────────── Main column ─────────────────── */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex items-center justify-between gap-3 border-b border-skin-border bg-white/80 px-4 py-3 backdrop-blur sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <button
              type="button"
              className="rounded-lg p-2 text-skin-muted hover:bg-slate-100 md:hidden"
              onClick={() => setMobileMenuOpen(true)}
              aria-label="Open menu"
            >
              <Menu className="h-5 w-5" strokeWidth={1.75} />
            </button>
            <h1 className="truncate text-sm font-semibold text-skin-text sm:text-base">
              {nav === "dashboard" && (data ? "Analysis" : "New analysis")}
              {nav === "model-comparison" && "Model comparison"}
              {nav === "knowledge" && "Data & RAG"}
              {nav === "history" && "History"}
              {nav === "system-health" && "System health"}
            </h1>
          </div>
          <div className="flex items-center gap-2 text-xs text-skin-muted">
            <Zap className="h-3.5 w-3.5 text-skin-accent" />
            <span className="hidden sm:inline">4 branches · one call</span>
          </div>
        </header>

        <main className="scrollbar-thin flex-1 overflow-y-auto px-4 py-5 sm:px-6">
          {showDashboardEmpty ? <EmptyState onPick={(q) => void sendText(q)} /> : null}

          {showDashboardResults ? (
            <div className="mx-auto max-w-6xl space-y-5" ref={resultsAnchorRef}>
              {loading ? (
                <div className="flex items-center gap-3 rounded-2xl border border-skin-border bg-skin-card p-4 text-sm text-skin-muted shadow-card animate-fade-in">
                  <Loader2 className="h-5 w-5 animate-spin text-skin-primary" aria-hidden />
                  Running RAG · non-RAG · ML · LLM in parallel…
                </div>
              ) : null}

              {error ? (
                <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 animate-fade-in">
                  {error}
                </div>
              ) : null}

              {data ? (
                <>
                  <div className="rounded-2xl border border-skin-border bg-skin-card px-4 py-3 text-sm shadow-card animate-fade-in-up">
                    <span className="text-[11px] font-bold uppercase tracking-wider text-skin-muted">You asked</span>
                    <p className="mt-1 text-skin-text">“{data.text}”</p>
                  </div>

                  <section aria-label="Answers">
                    <h2 className="mb-3 text-xs font-bold uppercase tracking-wider text-skin-muted">Answers</h2>
                    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                      <CardShell title="RAG answer" subtitle="LLM with retrieved context" icon={BookOpen}>
                        <p className="scrollbar-thin max-h-64 overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed text-skin-text">
                          {rag?.answer_text}
                        </p>
                        <div className="mt-2 text-xs text-skin-muted">
                          <span className="font-mono">{rag?.latency_ms.toFixed(1)} ms</span>
                          <span className="mx-1">·</span>
                          <span>{rag ? formatCallCostUsd(rag.cost_dollars, false) : ""}</span>
                        </div>
                        {rag ? <RagGroundingAlert contexts={rag.contexts} /> : null}
                      </CardShell>
                      <CardShell title="Non-RAG answer" subtitle="Same LLM without retrieval" icon={MessageSquare}>
                        <p className="scrollbar-thin max-h-64 overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed text-skin-text">
                          {nonRag?.answer_text}
                        </p>
                        <div className="mt-2 text-xs text-skin-muted">
                          <span className="font-mono">{nonRag?.latency_ms.toFixed(1)} ms</span>
                          <span className="mx-1">·</span>
                          <span>{nonRag ? formatCallCostUsd(nonRag.cost_dollars, false) : ""}</span>
                        </div>
                      </CardShell>
                    </div>
                  </section>

                  <section aria-label="Priority predictors">
                    <h2 className="mb-3 text-xs font-bold uppercase tracking-wider text-skin-muted">Priority predictors</h2>
                    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                      <CardShell title="ML priority" subtitle="Trained classifier · $0 API" icon={BarChart3}>
                        {ml ? (
                          <>
                            <PriorityBadge priority={ml.priority} />
                            <ConfidenceGauge value={ml.confidence} />
                            <dl className="mt-3 space-y-1 text-xs text-skin-muted">
                              <div className="flex justify-between">
                                <dt>Latency</dt>
                                <dd className="font-mono text-skin-text">{ml.latency_ms.toFixed(1)} ms</dd>
                              </div>
                              <div className="flex justify-between">
                                <dt>Cost</dt>
                                <dd className="font-mono font-semibold text-skin-text">
                                  {formatCallCostUsd(ml.cost_dollars, true)}
                                </dd>
                              </div>
                            </dl>
                          </>
                        ) : null}
                      </CardShell>
                      <CardShell title="LLM zero-shot" subtitle="Direct triage prompt · token cost" icon={Brain}>
                        {llm ? (
                          <>
                            <PriorityBadge priority={llm.priority} />
                            <LlmCostMeter dollars={llm.cost_dollars} />
                            <dl className="mt-2 space-y-1 text-xs text-skin-muted">
                              <div className="flex justify-between">
                                <dt>Latency</dt>
                                <dd className="font-mono text-skin-text">{llm.latency_ms.toFixed(1)} ms</dd>
                              </div>
                            </dl>
                          </>
                        ) : null}
                      </CardShell>
                    </div>
                  </section>

                  {ml && llm && rag && nonRag ? <FourWayMetricsTable ml={ml} llm={llm} rag={rag} nonRag={nonRag} /> : null}

                  <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                    <RetrievedSourcesPanel contexts={rag?.contexts ?? null} />
                    <DeploymentTable ml={ml} llm={llm} />
                  </div>
                </>
              ) : null}
            </div>
          ) : null}

          {nav === "model-comparison" ? (
            <div className="mx-auto max-w-4xl space-y-5">
              <PipelinePage />
              <DeploymentTable ml={ml} llm={llm} anchorId="deploy-page" />
              {ml && llm && rag && nonRag ? <FourWayMetricsTable ml={ml} llm={llm} rag={rag} nonRag={nonRag} /> : null}
              {!data ? (
                <p className="rounded-2xl border border-dashed border-skin-border bg-white/50 p-5 text-sm text-skin-muted">
                  Switch back to <span className="font-semibold text-skin-text">Dashboard</span> and run a query to
                  populate live metrics here.
                </p>
              ) : null}
            </div>
          ) : null}

          {nav === "knowledge" ? (
            <div className="mx-auto max-w-5xl">
              <DataAndRagPage />
            </div>
          ) : null}

          {nav === "history" ? (
            <div className="mx-auto max-w-5xl space-y-4">
              <HistoryPage
                entries={history}
                onOpen={openHistoryEntry}
                onDelete={deleteHistoryEntry}
                onClear={clearHistory}
              />
            </div>
          ) : null}

          {nav === "system-health" ? (
            <div className="mx-auto max-w-4xl space-y-4">
              <SystemHealthPanel
                loading={healthLoading}
                ok={healthOk}
                latencyMs={healthLatency}
                checkedAt={healthCheckedAt}
                body={healthText}
                historyCount={history.length}
                onRefresh={() => void checkHealth()}
              />
            </div>
          ) : null}
        </main>

        {/* ─────────────────── Composer (DeepSeek-style pill) ─────────────────── */}
        {nav === "dashboard" ? (
          <div className="border-t border-skin-border bg-skin-bg/80 px-4 py-4 backdrop-blur sm:px-6">
            <div className="mx-auto max-w-3xl">
              <div className="group relative flex items-end gap-2 rounded-3xl border border-skin-border bg-white p-2 shadow-card transition-all focus-within:border-skin-primary focus-within:shadow-glow">
                <textarea
                  ref={composerRef}
                  rows={1}
                  maxLength={500}
                  placeholder={`Message ${BRAND_NAME}…`}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      if (!loading && input.trim().length >= 1) runQuery();
                    }
                  }}
                  className="scrollbar-thin max-h-48 min-h-[32px] flex-1 resize-none bg-transparent px-3 py-2 text-sm text-skin-text placeholder:text-skin-muted focus:outline-none"
                  aria-label="User query"
                />
                <button
                  type="button"
                  disabled={loading || input.trim().length < 1}
                  aria-busy={loading}
                  onClick={runQuery}
                  className={cn(
                    "flex h-9 w-9 shrink-0 items-center justify-center rounded-full brand-gradient text-white shadow-sm transition disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none",
                    input.trim().length >= 1 && !loading && "hover:brightness-110 animate-ring-pulse",
                  )}
                  aria-label="Send"
                >
                  {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" strokeWidth={2} />}
                </button>
              </div>
              <div className="mt-2 flex justify-between text-[11px] text-skin-muted">
                <span>Enter to send · Shift+Enter newline</span>
                <span className="font-mono">{input.length} / 500</span>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
