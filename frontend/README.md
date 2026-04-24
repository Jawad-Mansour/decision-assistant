# Frontend — Decision Intelligence Assistant (Zap UI)

React + Vite + TypeScript + TailwindCSS. The UI ships a single-call analysis dashboard plus a local query history, a Data & RAG knowledge page, and a System Health panel. All substantive copy is in `src/App.tsx` — one file is intentional while the surface is small.

See the repo-root [`README.md`](../README.md) for how the frontend slots into the stack, and [`../docs/NOTEBOOKS.md`](../docs/NOTEBOOKS.md) if you want to understand what the backend is serving.

## Quickstart

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173  (Vite proxies /query, /health, /predict, /answer to :8000)
npm run build          # type-check + production build in dist/
npm run preview        # serve the built app locally
```

Optional pre-flight against the backend before building:

```bash
npm run test:backend-first     # runs backend pytest + /health smoke, then build
```

## Environment

Copy `.env.example` → `.env` in this folder only if you need either override:

| Variable | Purpose |
|---|---|
| `VITE_API_BASE_URL` | Full API URL for production builds (skipped when using the Vite dev proxy). |
| `VITE_ML_HOLDOUT_ACCURACY` / `VITE_LLM_HOLDOUT_ACCURACY` | Paste real hold-out numbers from `notebooks/12_ml_training_pipeline.ipynb`. The deployment table renders these only if set — nothing is invented in the UI. |

## File layout

```
frontend/
├── index.html
├── src/
│   ├── main.tsx         # React entry
│   ├── App.tsx          # all screens + components (sidebar, dashboard, history, health, ...)
│   ├── queryApi.ts      # API types + URL helpers + cost helpers
│   ├── evalMetrics.ts   # reads VITE_* hold-out numbers for the deployment table
│   ├── index.css        # Tailwind entry + a few utilities (scrollbar-thin, brand-gradient)
│   └── vite-env.d.ts
├── tailwind.config.js   # skin palette + keyframes (fade, heart-pulse, logo-float, ring-pulse)
├── postcss.config.js
├── tsconfig.json / tsconfig.node.json
├── package.json
├── Dockerfile           # multi-stage: Vite build → Nginx runtime
└── .env.example
```

## How the UI relates to the backend

- The app makes **one** call: `POST /query`. Latency, cost, and similarity scores in the cards come directly from that response.
- `GET /health` is hit only when you open **System Health**; it measures browser-side round-trip with `performance.now()`.
- History is local to the browser (`localStorage` key `triage.history.v1`, capped at 50 entries) — nothing is sent anywhere.

## Renaming the assistant

Change one constant at the top of `src/App.tsx`:

```ts
const BRAND_NAME = "Zap";
const BRAND_TAGLINE = "Decision Intelligence Assistant";
```

This propagates to the sidebar header, About modal, composer placeholder, and heart-button tooltip.
