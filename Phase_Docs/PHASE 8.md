# Phase 8 — Integration, Ablation Studies & Final Report/Demo

**Project:** Communication-Efficient Federated Learning for Diagnostic Imaging under Intermittent Rural Connectivity
**Team:** Pavan D Umesh (23BDS0046) · Aayush Sood (23BDS0177) · Yash Poddar (23BDS0195)
**Phase:** 8 of 8 (final)
**Estimated Duration:** 2–2.5 weeks
**Depends on:** All prior phases (1–7)

---

## 1. Where This Fits

Every prior phase built and validated one piece. Phase 8 does three things no earlier phase could: **prove the whole system works end to end**, **isolate exactly how much each individual technique contributed** (ablations — the difference between "we combined several methods" and "we can show what each one bought us"), and **package everything into the deliverables that actually get evaluated**: the written report and the live demo.

| Phase | Name | Status |
|---|---|---|
| 1–7 | Foundation → Monitoring Dashboard | ✅ done |
| **8** | **Integration, Ablations & Final Report** | **this file** |

---

## 2. Objectives of Phase 8

- [ ] A full end-to-end run (data → partitioned clients → compressed, async, connectivity-constrained FL training → results in Supabase → visible on the dashboard) is verified to work without manual intervention or hidden manual steps.
- [ ] A targeted ablation study isolates the individual contribution of each major component (compression, async aggregation, staleness weighting, connectivity realism) rather than only reporting the fully-combined system's performance.
- [ ] The final written report is structured to mirror and extend your original literature review — explicitly closing the gap identified in Slides 6–8 with your own results.
- [ ] A rehearsed live demo (using the Phase 7 dashboard) is ready, with a fallback plan if live infrastructure fails during presentation.
- [ ] The full repository is in a reproducible, documented state — someone outside the team could plausibly re-run the core pipeline from the README alone.

---

## 3. Full System Integration Check

Before ablations or writing begin, verify the entire pipeline works as one continuous system, not just as individually-tested phases:

1. **Fresh-environment test:** on a clean checkout (or with a teammate who hasn't touched the code recently), rebuild the environment from `environment.yml`, confirm data paths resolve, and run one small end-to-end experiment (few rounds, few clients) purely to confirm nothing is broken by undocumented local state.
2. **Full pipeline trace:** run one complete configuration from Phase 6's core sweep and manually trace a single number all the way through — e.g., pick one run's final accuracy, confirm it matches what's stored in Supabase, confirm it matches what the dashboard displays. This catches silent mismatches (e.g., dashboard caching stale data, ingestion script rounding differently than the tracker) before they surface during the demo.
3. **Failure-mode check:** deliberately try a couple of edge cases — a run with maximum dropout, a run with maximum compression — and confirm the system degrades gracefully (doesn't crash, produces sensible if poor results) rather than only ever having been tested on "reasonable" settings.

---

## 4. Ablation Study Design

The goal is to answer, with evidence: **"if we removed component X, how much would results change?"** for each of the following, reusing Phase 6's orchestration mechanism with each ablation as a distinct configuration:

| Ablation | Compares | What it isolates |
|---|---|---|
| **IID vs. non-IID partitioning** | Phase 2's FedAvg on Phase 1's non-IID clients vs. the same setup on a simple random/IID split | How much of any observed difficulty is genuinely due to non-IID data, separating "hard because federated" from "hard because non-IID" |
| **Error feedback on vs. off (sparsification)** | Phase 3's top-k sparsification with the residual buffer enabled vs. disabled | Directly validates the claim (Section 4.1 of Phase 3) that error feedback is necessary, not just helpful |
| **Staleness weighting on vs. off (async aggregation)** | Phase 4's buffered aggregation with staleness-aware weighting vs. naive uniform averaging of buffered updates | Isolates whether staleness weighting is actually load-bearing for stability, or if buffering alone would have sufficed |
| **Simple dropout model vs. connectivity-trace-driven model** | Phase 4's probability-based dropout vs. Phase 5's realistic connectivity engine, holding the aggregation strategy fixed | Shows whether the added realism of Phase 5 changes conclusions meaningfully, justifying that phase's complexity — or shows results were robust to the modeling choice either way, which is also a reportable finding |
| **Compression alone vs. compression + async vs. full system** | Three configurations at matched dropout/bandwidth conditions | Shows whether combining techniques gives a genuinely compounded benefit (the project's central claim, per your Slide 7 gap analysis) or whether gains are mostly attributable to one dominant technique |

Each ablation needs its own small, focused sweep (not the full Phase 6 grid) — a handful of seeds at one or two representative operating points is sufficient to support an ablation claim; exhaustive re-sweeping of every ablation across every condition is not necessary and not a good use of remaining time.

---

## 5. Final Report Structure

Mirror your literature review's own structure, since it was built around a problem/gap/direction narrative that your results now complete:

1. **Introduction & Problem Statement** — restate the motivating scenario (rural clinics, intermittent connectivity, diagnostic imaging) from your original title slide.
2. **Related Work** — condensed version of your literature review's Groups A–D tables; this section already exists in near-final form from your original deck.
3. **Research Gap** — reuse Slides 6–8's gap analysis directly; it still accurately describes what was missing before this project.
4. **Proposed Approach & System Design** — summarize Phases 1–5's design decisions (label taxonomy, non-IID partitioning, compression methods chosen, async aggregation mechanism, connectivity modeling) at a level of detail appropriate for a report, not a full re-statement of every implementation file.
5. **Experimental Setup** — datasets, model architecture, hyperparameters, evaluation protocol (drawing from Phases 1–2 and 6).
6. **Results** — the Phase 6 joint-metric heatmap as the centerpiece figure, plus the Phase 8 ablation results, plus per-client findings where relevant (e.g., any disproportionate impact on underperforming clinics noted in Phases 2–5).
7. **Limitations** — be explicit and honest: e.g., simulation-based connectivity (not real deployed clinics), label taxonomy simplifications from Phase 1, compute/time constraints on sweep breadth (Phase 6, Section 3.3). Papers in your own literature review are commended for stating limitations clearly (see the "Limitations" column of your Groups A–D tables) — hold your own report to the same standard.
8. **Conclusion & Future Work** — what the joint results show about combining compression + straggler tolerance + medical imaging FL, and what a next iteration (e.g., real hardware deployment, additional datasets, secure aggregation per BASecAgg) would look like.

---

## 6. Demo Preparation

- **Primary demo path:** walk through the deployed Phase 7 dashboard live — overview → drill into a representative run → show the joint-metric heatmap → point to the recommended operating point from Phase 6.
- **Fallback path:** have static screenshots/exported charts of every dashboard view ready in case live infrastructure (Supabase/Vercel/Render) has an outage or connectivity issue during presentation — precisely the kind of "intermittent connectivity" irony worth avoiding live.
- **Rehearse the narrative, not just the clicks:** the demo should tell the same story as the report (problem → gap → approach → result), not just click through UI features without narrative framing.
- **Anticipate likely questions** from evaluators, e.g.: "why these three datasets," "how is this different from just running FedBuff," "what would change with real hospitals instead of simulation" — prepare direct, evidence-backed answers to these before presentation day, not on the spot.

---

## 7. Reproducibility Packaging

- [ ] Top-level `README.md` updated with: project summary, repo structure overview, environment setup instructions, and how to reproduce at least one headline result end to end.
- [ ] `infra/docs/decisions-log.md` (maintained since Phase 1) reviewed and cleaned up as a single reference of every major design decision and its justification — this doubles as a source document when writing the report's methodology section.
- [ ] All configs used for final reported numbers are present in the repo and named clearly enough to map back to specific report figures/tables.
- [ ] Any dataset access instructions (especially CheXpert's registration requirement, per Phase 1) documented so the pipeline isn't silently unreproducible for someone without existing access.

---

## 8. Deliverables Checklist

- [ ] Full end-to-end integration verified on a fresh environment (Section 3, step 1).
- [ ] Single-number trace-through completed and confirmed consistent across tracker, Supabase, and dashboard (Section 3, step 2).
- [ ] Edge-case failure modes tested and confirmed to degrade gracefully (Section 3, step 3).
- [ ] All five ablation studies from Section 4 completed, with results clearly attributable to the specific component being isolated.
- [ ] Final report drafted following the structure in Section 5, with the Phase 6 heatmap as the centerpiece figure.
- [ ] Limitations section written honestly and specifically, not generically.
- [ ] Demo rehearsed at least once end-to-end on the actual deployed dashboard.
- [ ] Fallback static assets prepared in case of live infrastructure issues during the demo.
- [ ] README and decisions log finalized for reproducibility.

---

## 9. Definition of Done (Project-Level)

The project is complete when: the full pipeline runs end to end from a fresh environment, the ablation results support specific, falsifiable claims about each component's contribution, the report closes the exact gap identified in your original literature review with your own evidence rather than citations to others' work, and the demo can be delivered confidently even if something in the live infrastructure hiccups.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Ablation studies reveal a component contributes less than expected (or hurts) | Report this honestly — a well-explained negative or null ablation result is more credible than results that suspiciously always favor every design choice made |
| Report and dashboard tell slightly inconsistent stories due to last-minute changes in one but not the other | Do the Section 3, step 2 trace-through again after any late change to either the pipeline or the dashboard, right before finalizing the report |
| Live demo fails during presentation | Fallback static assets (Section 6) prepared and tested in advance, not assembled at the last minute |
| Running out of time for full ablation breadth | Prioritize the "error feedback on/off" and "compression alone vs. full system" ablations first — they most directly support the project's central claim; treat the IID-vs-non-IID and dropout-model ablations as valuable but secondary if time is short |
| Report reads as a re-statement of implementation details rather than a scientific narrative | Structure explicitly around the problem → gap → approach → result → limitations arc (Section 5), not a phase-by-phase changelog of what was built |

---

## Project Complete

This closes the 8-phase implementation plan: **Foundation → FL Baseline → Compression → Async Aggregation → Connectivity Engine → Evaluation Orchestration → Dashboard → Integration & Report.** Each phase file in this series is self-contained enough to hand to a teammate independently, while this final phase is where all of them are expected to visibly click together into one coherent, defensible piece of work.
