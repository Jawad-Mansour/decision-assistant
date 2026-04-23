import { useCallback, useRef, useState } from "react";

type PriorityBlock = {
  priority: string;
  confidence: number | null;
  latency_ms: number;
  cost_dollars: number;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
};

type Retrieved = {
  id: string;
  text: string;
  distance: number | null;
  similarity_score?: number | null;
};

type AnswerBlock = {
  mode: string;
  answer_text: string;
  contexts: Retrieved[];
  latency_ms: number;
  cost_dollars: number;
};

type QueryResponseBody = {
  text: string;
  ml_priority: PriorityBlock;
  llm_priority: PriorityBlock;
  rag_answer: AnswerBlock;
  non_rag_answer: AnswerBlock;
};

type ChatTurn = { role: "user" | "assistant"; text: string };

function apiBase(): string {
  const raw = import.meta.env.VITE_API_BASE_URL ?? "";
  return raw.replace(/\/$/, "");
}

function queryUrl(): string {
  const base = apiBase();
  return base ? `${base}/query` : "/query";
}

function costPerTenThousand(costPerRequest: number): number {
  return costPerRequest * 10_000;
}

function PriorityCard(props: { title: string; subtitle: string; block: PriorityBlock }) {
  const { title, subtitle, block } = props;
  const urgent = block.priority === "urgent";
  const tokenLine =
    block.prompt_tokens != null || block.completion_tokens != null
      ? `${block.prompt_tokens ?? 0} / ${block.completion_tokens ?? 0}`
      : "Not applicable (on-device ML)";
  return (
    <div className="panel panel-priority">
      <div className="panel-head">
        <h2>{title}</h2>
        <p className="panel-sub">{subtitle}</p>
      </div>
      <div className="panel-body">
        <div className={`priority-pill ${urgent ? "urgent" : "normal"}`}>{block.priority}</div>
        <dl className="metrics">
          {block.confidence != null ? (
            <>
              <dt>Confidence</dt>
              <dd>{(block.confidence * 100).toFixed(1)}%</dd>
            </>
          ) : null}
          <dt>Latency</dt>
          <dd>{block.latency_ms.toFixed(1)} ms</dd>
          <dt>Est. cost (this call)</dt>
          <dd>${block.cost_dollars.toFixed(5)}</dd>
          <dt className="full-width">API tokens (prompt / completion)</dt>
          <dd className="full-width muted">{tokenLine}</dd>
        </dl>
      </div>
    </div>
  );
}

function AnswerCard(props: { title: string; subtitle: string; block: AnswerBlock; showSources: boolean }) {
  const { title, subtitle, block, showSources } = props;
  return (
    <div className="panel panel-answer">
      <div className="panel-head">
        <h2>{title}</h2>
        <p className="panel-sub">{subtitle}</p>
      </div>
      <div className="panel-body">
        <p className="answer-text">{block.answer_text}</p>
        <div className="answer-meta">
          <span>{block.latency_ms.toFixed(1)} ms</span>
          <span>·</span>
          <span>${block.cost_dollars.toFixed(5)}</span>
        </div>
        {showSources && block.contexts.length > 0 ? (
          <div className="sources-block">
            <h3 className="sources-title">Grounding sources</h3>
            <p className="sources-hint">
              Similarity is cosine similarity in [0, 1] from Chroma cosine distance (higher = closer match).
              The answer is constrained to these retrieved conversations.
            </p>
            <ol className="sources-list">
              {block.contexts.map((c, i) => (
                <li key={`${c.id}-${i}`} className="source-item">
                  <div className="source-scores">
                    <span className="score-badge">
                      similarity {(c.similarity_score != null ? c.similarity_score : 0).toFixed(4)}
                    </span>
                    {c.distance != null ? (
                      <span className="distance-muted">distance {c.distance.toFixed(4)}</span>
                    ) : null}
                    <span className="source-id">id {c.id}</span>
                  </div>
                  <div className="source-snippet">{(c.text || "").slice(0, 520)}{(c.text || "").length > 520 ? "…" : ""}</div>
                </li>
              ))}
            </ol>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function ComparisonTable(props: { ml: PriorityBlock; llm: PriorityBlock }) {
  const { ml, llm } = props;
  const rows = [
    {
      name: "ML (Random Forest, on-box)",
      priority: ml.priority,
      confidence: ml.confidence != null ? `${(ml.confidence * 100).toFixed(1)}%` : "—",
      latency: `${ml.latency_ms.toFixed(2)} ms`,
      cost10k: costPerTenThousand(ml.cost_dollars),
      note: "No API tokens; cost is infrastructure only.",
    },
    {
      name: "LLM zero-shot (API)",
      priority: llm.priority,
      confidence: "—",
      latency: `${llm.latency_ms.toFixed(2)} ms`,
      cost10k: costPerTenThousand(llm.cost_dollars),
      note: "Scaled as 10,000 × single-call list-price estimate.",
    },
  ];
  return (
    <section className="compare-section">
      <h2 className="section-title">At scale: ML vs LLM priority (10,000 tickets)</h2>
      <p className="section-lead">
        Throughput at 10,000 tickets/hour is dominated by concurrency and provider rate limits for LLMs.
        This table compares honest per-ticket API metering for the LLM path against effectively free local ML
        inference for the same volume.
      </p>
      <div className="table-wrap">
        <table className="compare-table">
          <thead>
            <tr>
              <th>Predictor</th>
              <th>Priority</th>
              <th>Confidence</th>
              <th>Latency (this run)</th>
              <th>Est. cost / 10k tickets</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name}>
                <td>{r.name}</td>
                <td>
                  <span className={r.priority === "urgent" ? "pill-inline urgent" : "pill-inline normal"}>{r.priority}</span>
                </td>
                <td>{r.confidence}</td>
                <td>{r.latency}</td>
                <td>${r.cost10k.toFixed(2)}</td>
                <td className="muted">{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function App() {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<QueryResponseBody | null>(null);
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const runQuery = useCallback(async () => {
    const trimmed = input.trim();
    if (trimmed.length < 1) return;
    setError(null);
    setLoading(true);
    setHistory((h) => [...h, { role: "user", text: trimmed }]);
    setInput("");
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
      setHistory((h) => [
        ...h,
        {
          role: "assistant",
          text: "Pipeline finished. ML vs LLM triage and RAG vs non-RAG answers are shown in the panels below.",
        },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
      setHistory((h) => [...h, { role: "assistant", text: "Sorry — something went wrong. Check the message below." }]);
    } finally {
      setLoading(false);
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    }
  }, [input]);

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="logo-mark" aria-hidden />
          <div>
            <h1 className="title">Decision Intelligence Assistant</h1>
            <p className="tagline">One question · four parallel analyses · grounded comparison</p>
          </div>
        </div>
      </header>

      <main className="main">
        <section className="chat-card">
          <div className="chat-scroll">
            {history.length === 0 ? (
              <div className="empty-chat">
                <p>
                  Ask a customer-support style question. The backend runs <strong>ML priority</strong>,{" "}
                  <strong>LLM zero-shot priority</strong>, <strong>RAG answer</strong>, and{" "}
                  <strong>non-RAG answer</strong> in parallel on a single <code>/query</code> call.
                </p>
              </div>
            ) : (
              <ul className="messages">
                {history.map((m, i) => (
                  <li key={i} className={`msg msg-${m.role}`}>
                    <span className="msg-label">{m.role === "user" ? "You" : "Assistant"}</span>
                    <div className="msg-bubble">{m.text}</div>
                  </li>
                ))}
              </ul>
            )}
            <div ref={bottomRef} />
          </div>
          <div className="composer">
            <textarea
              rows={2}
              placeholder="Describe the customer issue (1–500 characters)…"
              maxLength={500}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (!loading && input.trim().length >= 1) void runQuery();
                }
              }}
            />
            <button type="button" className="send-btn" disabled={loading || input.trim().length < 1} onClick={() => void runQuery()}>
              {loading ? "Running…" : "Send"}
            </button>
          </div>
        </section>

        {loading ? <div className="banner banner-info">Running full pipeline (ML + LLM + RAG + non-RAG)…</div> : null}
        {error ? <div className="banner banner-error">{error}</div> : null}

        {data ? (
          <>
            <section className="results-intro">
              <h2 className="section-title">Results for your last message</h2>
              <p className="user-echo">“{data.text}”</p>
            </section>

            <div className="four-up">
              <PriorityCard
                title="ML priority"
                subtitle="Trained classifier · no API cost"
                block={data.ml_priority}
              />
              <PriorityCard
                title="LLM zero-shot"
                subtitle="Single triage call · token-metered"
                block={data.llm_priority}
              />
              <AnswerCard
                title="RAG answer"
                subtitle="Retrieval from similar past conversations + LLM"
                block={data.rag_answer}
                showSources
              />
              <AnswerCard title="Non-RAG answer" subtitle="LLM only · no retrieval" block={data.non_rag_answer} showSources={false} />
            </div>

            <ComparisonTable ml={data.ml_priority} llm={data.llm_priority} />
          </>
        ) : null}
      </main>

      <footer className="footer">
        Costs use list-price token rates on actual usage counts from the provider. ML path shows $0 API spend;
        infrastructure is separate.
      </footer>
    </div>
  );
}
