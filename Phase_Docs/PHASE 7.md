# Phase 7 — Monitoring Dashboard: Next.js + Express + Supabase

**Project:** Communication-Efficient Federated Learning for Diagnostic Imaging under Intermittent Rural Connectivity
**Team:** Pavan D Umesh (23BDS0046) · Aayush Sood (23BDS0177) · Yash Poddar (23BDS0195)
**Phase:** 7 of 8
**Estimated Duration:** 2 weeks
**Depends on:** Phase 6 (populated Supabase schema with real sweep results)

---

## 1. Where This Fits

Every prior phase produced results that lived in the experiment tracker, local plots, and now Supabase (Phase 6). Phase 7 builds the actual **web application** — the piece that makes those results explorable by a human, not just queryable by someone comfortable with SQL or the tracker's UI. This is also the phase that fulfills the project's explicit tech-stack requirement: **Next.js frontend, Node.js + Express backend, Supabase PostgreSQL database**.

This dashboard is a **read/visualization layer over Phase 6's results** — it does not run FL training itself, does not talk to the Python research code directly, and does not need to be "live" during actual training runs unless you choose to pursue the optional real-time extension in Section 8.

| Phase | Name | Status |
|---|---|---|
| 1–6 | Foundation → Experiment Orchestration | ✅ done |
| **7** | **Monitoring Dashboard** | **this file** |
| 8 | Integration, Ablations & Final Report | depends on all |

---

## 2. Objectives of Phase 7

- [ ] A Next.js frontend renders an overview of all experiments/sweeps, an individual run's convergence curves, and the Phase 6 joint-metric heatmap, all pulling from real Supabase data.
- [ ] An Express backend API sits between the frontend and Supabase, exposing clean, purpose-built endpoints rather than having the frontend query Supabase directly with raw SQL.
- [ ] Per-client views let you drill into which simulated clinics struggled, dropped out most, or had the worst connectivity — surfacing Phase 5/6 data at the individual-client level, not just aggregate.
- [ ] The dashboard is presentable as a demo artifact in its own right for Phase 8's final presentation, not just an internal debugging tool.

---

## 3. Architecture Overview

```
[ Supabase PostgreSQL ]
        ▲
        │  (SQL queries via Supabase client library)
        │
[ Express Backend API ]  ← Node.js
        ▲
        │  (REST calls, JSON)
        │
[ Next.js Frontend ]  ← renders charts, tables, heatmaps
        ▲
        │
   [ Browser / Demo ]
```

**Why not have Next.js talk to Supabase directly (which Supabase's client library technically allows)?** Because the project's stated stack explicitly separates frontend (Next.js) from backend (Node.js + Express) from database (Supabase) — keeping the Express layer in place also means query logic, aggregation, and any future auth/rate-limiting lives in one place, and the frontend never needs direct database credentials.

---

## 4. Backend (Node.js + Express) Design

### 4.1 Responsibilities

- Own the Supabase client connection (service-role or scoped key, never exposed to the frontend).
- Expose REST endpoints that map to dashboard views, not raw tables — e.g., an endpoint that returns a run's full convergence curve pre-joined and formatted, rather than making the frontend stitch together multiple raw table queries.
- Perform any aggregation that's easier server-side (e.g., computing mean ± variance across seeds for a given configuration) rather than shipping raw per-seed rows to the frontend and aggregating in the browser.

### 4.2 Endpoints to design (conceptually — no code)

| Endpoint | Returns |
|---|---|
| `GET /experiments` | List of all experiments/sweeps with summary metadata |
| `GET /experiments/:id/runs` | All runs belonging to an experiment, with their configuration and final metrics |
| `GET /runs/:id/rounds` | Full round-by-round convergence data for one run (accuracy, AUROC, F1, communication volume over time) |
| `GET /runs/:id/clients` | Per-client participation/dropout/staleness breakdown for one run |
| `GET /experiments/:id/joint-metric` | Pre-aggregated data shaped for the Phase 6 bandwidth × dropout heatmap |
| `GET /clients/:id` | Static metadata for one simulated clinic (data volume, label distribution, connectivity profile) — cross-referencing Phase 1/5 data |

### 4.3 Non-functional considerations

- Input validation on all route parameters (run/experiment/client IDs) before querying Supabase.
- Sensible error responses (404 for missing IDs, not a raw database error leaking to the frontend).
- CORS configured to allow only the Next.js frontend's origin, not left wide open.
- Environment-based config (Supabase URL/key, allowed origins) via `.env`, never hardcoded — continuing the discipline from Phase 1.

---

## 5. Frontend (Next.js) Design

### 5.1 Pages/views to build

| Page | Purpose |
|---|---|
| **Overview / Home** | List of experiments/sweeps, high-level summary cards (best accuracy achieved, best compression ratio, best operating point from Phase 6) |
| **Experiment detail** | All runs within one experiment, filterable/sortable by compression setting, aggregation strategy, dropout condition |
| **Run detail** | Convergence curves (accuracy/AUROC/F1 vs. round), communication-volume-over-time chart, and a client participation timeline for that specific run |
| **Joint metric heatmap view** | The core Phase 6 visualization — accuracy retained as a function of bandwidth and dropout — rendered interactively (hover to see exact numbers/configuration at each point) |
| **Client explorer** | Browse simulated clinics, see their data volume, label distribution, connectivity profile, and how they performed/participated across runs |

### 5.2 Component/library guidance

- Use a charting library well-suited to both line charts (convergence curves) and heatmaps (joint metric) — evaluate options during setup rather than committing upfront, since the heatmap requirement is somewhat more specialized than typical dashboard line/bar charts.
- Keep data-fetching on the server side (Next.js server components / server-side data fetching) where possible, so the dashboard loads with data already present rather than showing loading spinners for every view — this also matters for a live demo in Phase 8, where flaky client-side fetching during a presentation is a real risk.
- Design for **read-mostly** interaction: filtering, sorting, drilling into a run — this dashboard does not need forms, mutations, or write operations, which simplifies both the frontend and the backend considerably.

### 5.3 Visual design guidance

- Favor clarity over decoration: this is a technical/scientific dashboard for a capstone project, not a marketing site — prioritize legible charts, clear axis labels (with units — kbps, %, MB), and consistent color coding for compression method / aggregation strategy across every chart, so a viewer doesn't have to re-learn the color legend on each page.
- Use the joint-metric heatmap's color scale consistently between the dashboard and any static figures reused in the Phase 8 final report — visual consistency between the two makes the project read as one coherent piece of work rather than two disconnected halves.

---

## 6. Data Flow: Research Code → Dashboard

Reconfirm the pipeline established in Phase 6:

1. Python research code runs experiments (Phases 2–5's logic, orchestrated by Phase 6's runner).
2. Results are batch-ingested into Supabase (Phase 6, Section 5.2) — this remains unchanged in this phase.
3. The Express backend queries Supabase and shapes responses for the frontend.
4. The Next.js frontend renders those responses as charts/tables.

**No new results should be generated in this phase** — Phase 7 is purely about visualizing Phase 6's already-produced data. If a needed metric isn't in Supabase, that's a sign to revisit Phase 6's schema/ingestion, not to compute it fresh inside the web app.

---

## 7. Deployment Considerations

| Component | Suggested approach |
|---|---|
| Next.js frontend | Vercel (native fit for Next.js, free tier sufficient for a college project demo) |
| Express backend | Render, Railway, or similar free/low-cost Node hosting |
| Supabase | Already hosted (Supabase's own managed Postgres) |

For the final Phase 8 demo, confirm the deployed dashboard is reachable and loads real data **before** presentation day — do not rely on running everything locally live during the demo unless a deployed version isn't feasible in time.

---

## 8. Optional Extension: Real-Time View (Time-Permitting Only)

If time allows after the core dashboard (Sections 3–7) is solid, Supabase's real-time subscription feature could let the dashboard show a run's progress live while an experiment is still executing, rather than only after batch ingestion completes. **Treat this as strictly optional** — it adds meaningful complexity (keeping a long-running Python process's writes flowing incrementally rather than in one batch at the end) for a benefit that mostly matters during a live demo, not for the project's actual scientific deliverable. Do not let this extension threaten the core dashboard's completion.

---

## 9. Repository Additions for This Phase

```
web/
├── frontend/                       # Next.js app
│   ├── app/ (or pages/)
│   │   ├── page.tsx                # Overview
│   │   ├── experiments/[id]/       # Experiment detail
│   │   ├── runs/[id]/              # Run detail (convergence, client timeline)
│   │   ├── heatmap/                # Joint metric view
│   │   └── clients/                # Client explorer
│   └── components/                 # Charts, tables, shared UI
└── backend/                         # Express app
    ├── routes/
    │   ├── experiments.js
    │   ├── runs.js
    │   ├── clients.js
    │   └── joint-metric.js
    ├── supabase-client.js           # Centralized Supabase connection
    └── server.js
```

*(Structure and responsibilities only — no code included per your instructions.)*

---

## 10. Deliverables Checklist

- [ ] Express backend implemented with all endpoints from Section 4.2, each validated against real Supabase data from Phase 6.
- [ ] Next.js frontend implemented with all five views from Section 5.1.
- [ ] Joint-metric heatmap rendered interactively in the browser, matching the static version produced in Phase 6.
- [ ] Consistent color coding and units applied across all charts.
- [ ] CORS, environment-based config, and basic error handling in place on the backend.
- [ ] Dashboard deployed (or confirmed reliably runnable) ahead of the Phase 8 demo.
- [ ] Cross-check: every number visible on the dashboard traces back to a real Phase 6 Supabase row — no placeholder or mock data left in by demo time.

---

## 11. Definition of Done

Phase 7 is complete when someone with no prior context can open the deployed dashboard, see the list of experiments, drill into a specific run's convergence curve, view the joint-metric heatmap, and inspect why a specific simulated clinic underperformed — entirely through the UI, without needing you to explain a database query.

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Frontend and backend built in parallel drift out of sync on data shape | Fix the API response shapes (Section 4.2) in a shared reference doc before frontend work starts in earnest |
| Dashboard becomes a second place where bugs in Phase 6's numbers get "discovered" for the first time | Treat any dashboard number that looks wrong as a signal to re-verify Phase 6's underlying data, not to patch the number in the frontend |
| Heatmap library doesn't handle the joint-metric data shape well | Evaluate 1–2 charting libraries early with real (even partial) Phase 6 data before committing frontend layout around one |
| Live demo fails due to a flaky deployed backend/frontend | Confirm the deployed dashboard end-to-end at least a day before the demo, with a local fallback ready as backup |
| Real-time extension (Section 8) eats time meant for the core dashboard | Explicitly time-box it and treat the core dashboard (Sections 3–7) as the non-negotiable deliverable |

---

## 13. Next Phase Preview

**Phase 8 — Integration, Ablations & Final Report** ties every phase together: a full end-to-end system check, targeted ablation studies isolating each component's individual contribution, and the final written report and presentation — using this dashboard as a live artifact and Phase 6's results as the evidentiary backbone.
