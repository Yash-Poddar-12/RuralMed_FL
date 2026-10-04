# Phase 5 — Intermittent Connectivity Engine: Trace-Driven Bandwidth & Dropout Simulation

**Project:** Communication-Efficient Federated Learning for Diagnostic Imaging under Intermittent Rural Connectivity
**Team:** Pavan D Umesh (23BDS0046) · Aayush Sood (23BDS0177) · Yash Poddar (23BDS0195)
**Phase:** 5 of 8
**Estimated Duration:** 2 weeks
**Depends on:** Phase 3 (compressed payload sizes), Phase 4 (pluggable straggler-model interface, buffered async aggregation)

---

## 1. Where This Fits

Phase 4 modeled dropout and straggling with simple, hand-picked probability distributions — enough to build and validate the async aggregation mechanism, but not a real stand-in for "intermittent rural connectivity." Phase 5 replaces that placeholder with a **realistic, trace-driven connectivity engine**: each simulated clinic gets a bandwidth and connection-availability profile over time, and whether/how fast a client's update arrives is *derived* from that profile and the actual payload size (Phase 3's compressed bytes), not sampled from an arbitrary distribution.

This is the phase where your project's central framing — "bandwidth-aware, straggler-tolerant FL under intermittent rural connectivity" — stops being a description of intent and becomes an actual simulated condition your model is tested under. Critically, this phase plugs into the **exact interface** Phase 4 built (Section 3 of that file) — the aggregation strategy itself does not change; only what feeds it delay/dropout events changes.

| Phase | Name | Status |
|---|---|---|
| 1 | Foundation: Env, Repo, Data | ✅ done |
| 2 | FL Core Baseline (FedAvg) | ✅ done |
| 3 | Communication-Efficient Compression | ✅ done |
| 4 | Straggler-Tolerant Async Aggregation | ✅ done |
| **5** | **Intermittent Connectivity Engine** | **this file** |
| 6 | Evaluation & Experiment Orchestration | depends on 3–5 |
| 7 | Monitoring Dashboard | depends on Phase 6 |
| 8 | Integration, Ablations & Final Report | depends on all |

---

## 2. Objectives of Phase 5

By the end of this phase you should be able to answer "yes" to all of the following:

- [ ] Each simulated clinic has a connectivity profile (bandwidth + availability over time) rather than a single static dropout probability.
- [ ] Whether a client's update "arrives in time" for a given round/buffer window is computed from **actual transmission time** (payload size ÷ available bandwidth at that moment), not sampled abstractly.
- [ ] Connectivity profiles are heterogeneous across clients in a documented, defensible way (not uniform, not purely random) — mirroring real regional variation in rural connectivity.
- [ ] This engine plugs into Phase 4's pluggable straggler-model interface with no changes required to the aggregation strategy itself.
- [ ] You can now sweep a **real bandwidth constraint** (e.g., "clinics capped at N kbps") as an experimental axis, alongside the dropout-rate axis from Phase 4 — giving you the actual X and Y axes of the project's target metric: "accuracy retained at X% bandwidth, Y% dropout."

---

## 3. Connectivity Modeling Approach

### 3.1 Two viable approaches — pick one, document the choice

| Approach | Description | Tradeoff |
|---|---|---|
| **A. Markov-chain connectivity states** | Each client's connection alternates between discrete states (e.g., Connected-Good, Connected-Poor, Disconnected) with transition probabilities between states over time, plus a bandwidth range associated with each state | Simpler to implement and tune, easy to make interpretable and clearly document the assumptions behind it — **recommended default** given the project's timeline |
| **B. Real-world network trace replay** | Use publicly available cellular/rural-network bandwidth trace datasets (e.g., traces from network measurement studies), replayed per simulated client, possibly with regional/rural-specific traces if available | More externally credible ("we didn't invent the connectivity pattern"), but harder to source traces specifically representative of rural clinic connectivity, and adds a data-acquisition dependency this late in the project |

**Recommendation:** implement Approach A as the primary mechanism (it directly parametrizes the "X% bandwidth, Y% dropout" target metric cleanly), and treat Approach B as an optional enrichment — if a suitable public rural/cellular bandwidth trace dataset can be found quickly, calibrate Approach A's Markov state bandwidth ranges and transition probabilities against it for added realism, rather than fully replacing the simpler mechanism.

### 3.2 Per-client heterogeneity

Do not give every simulated clinic the same connectivity profile — that would defeat the purpose of calling this "rural" simulation. Assign profiles with deliberate heterogeneity, for example:
- A subset of clients with mostly-good connectivity, occasional brief drops (representing better-served clinics).
- A subset with poor baseline bandwidth but stable connection (slow-but-steady clinics).
- A subset with frequent, unpredictable disconnection (the worst-case "intermittent" clinics the project title calls out).

**Decision to record:** whether to correlate connectivity quality with client data volume (Phase 1, Section 7.2) or client label distribution (Phase 1, Section 7.1) — e.g., is it more realistic for larger clinics to have better infrastructure, or is that an assumption you want to explicitly avoid making? Either choice is defensible; document which one you made and why, since it affects how any observed accuracy disparities across clients should be interpreted in the final report.

### 3.3 Bandwidth-to-time conversion

Given a client's current available bandwidth (from its connectivity state) and the actual compressed payload size (bytes, from Phase 3's accounting), compute the transmission time required. If that time exceeds the round's time budget (synchronous case) or the client remains disconnected long enough to miss the current buffer window (asynchronous case), treat it as a straggler event or dropout event respectively — feeding directly into Phase 4's existing aggregation logic through the same interface as before.

---

## 4. Interfacing with Phase 4

Phase 4 defined a pluggable straggler-model interface (dropout sampler + delay sampler) that the aggregation strategy consumes without knowing the internals. This phase's job is to build a **new implementation behind that same interface** — not a new aggregation mechanism. Concretely:
- The connectivity engine replaces `dropout_sampler.py` and `delay_sampler.py` from Phase 4's structure with connectivity-trace-driven equivalents.
- The FedBuff-style buffered aggregation strategy, staleness weighting, and naive synchronous-with-dropout baseline from Phase 4 remain **completely unchanged** — if you find yourself editing the aggregation logic in this phase, that's a sign the interface boundary wasn't respected in Phase 4 and needs revisiting before proceeding.

---

## 5. Experimental Axes This Phase Unlocks

| Axis | What it controls | Where it plugs in |
|---|---|---|
| **Bandwidth cap** (e.g., kbps available to a given connectivity state) | Determines transmission time for a given payload — directly interacts with Phase 3's compression ratio (higher compression = same bandwidth goes further) | Feeds delay computation (Section 3.3) |
| **Connection-loss/disconnection frequency** | Determines how often a client is unreachable at all, regardless of bandwidth | Feeds dropout events, consumed by Phase 4's buffered aggregation |
| **Compression setting** (carried over from Phase 3) | Reduces payload size, reducing transmission time for the same bandwidth | Interacts multiplicatively with bandwidth cap — worth an explicit joint sweep |

This is the first phase where **bandwidth and compression jointly determine outcomes**, rather than compression ratio being evaluated in isolation (Phase 3) or dropout being evaluated in isolation (Phase 4) — a client with heavy compression but very low bandwidth may behave similarly to a client with light compression but decent bandwidth, and demonstrating this interaction is a genuinely novel contribution relative to the literature you reviewed, where (per your own Slide 6/7 gap analysis) compression and connectivity-tolerance research have stayed on separate tracks.

---

## 6. Metrics to Track

In addition to everything carried over from Phases 2–4:

| Metric | Why it matters here |
|---|---|
| Per-client effective bandwidth over time (trace visualization) | Sanity check that the connectivity engine is producing plausible-looking, heterogeneous traces before trusting downstream results |
| Per-client "uptime %" (fraction of time in a connected state) | Summary statistic tying back to the per-client heterogeneity design (Section 3.2) |
| Round/buffer-window completion time under realistic connectivity vs. Phase 4's simple probability model | Confirms the new engine produces meaningfully different (and more realistic) timing behavior, not just a relabeled version of Phase 4's model |
| Accuracy retained as a function of **average network bandwidth cap** (not just compression ratio) | This is the literal "X% bandwidth" axis from your proposed direction slide, now measured for real |

---

## 7. Repository Additions for This Phase

```
research/
├── src/
│   └── simulation/
│       ├── connectivity_states.py     # Markov-chain state definitions + transition probabilities (Approach A)
│       ├── client_profiles.py         # Per-client heterogeneous profile assignment (Section 3.2)
│       ├── bandwidth_delay.py         # Bandwidth-to-transmission-time conversion (Section 3.3), implements Phase 4's delay-sampler interface
│       └── trace_replay.py            # Optional: real-world trace loading/replay (Approach B, if pursued)
├── configs/
│   ├── phase5_connectivity_profiles.yaml
│   └── phase5_bandwidth_sweep.yaml
└── results/
    └── phase5/                         # Trace visualizations, uptime stats, accuracy-vs-bandwidth curves
```

*(Structure and responsibilities only — no code included per your instructions.)*

---

## 8. Deliverables Checklist

- [ ] Markov-chain (or chosen alternative) connectivity state model implemented with documented states and transition probabilities.
- [ ] Heterogeneous per-client connectivity profiles assigned, with the correlation decision from Section 3.2 explicitly recorded.
- [ ] Bandwidth-to-transmission-time conversion implemented and validated against Phase 3's actual compressed-payload byte counts.
- [ ] New connectivity-trace-driven implementation plugged into Phase 4's existing straggler-model interface, with **zero changes** to the aggregation strategy itself confirmed.
- [ ] Trace visualizations produced for a sample of clients, visually reviewed for plausibility before running full experiments on top of them.
- [ ] Joint bandwidth × compression sweep run, demonstrating the interaction effect described in Section 5.
- [ ] Accuracy-vs-bandwidth-cap curve produced, parallel in spirit to Phase 3's accuracy-vs-compression-ratio curve.
- [ ] Per-client uptime % reported and cross-referenced against Phase 1's partition characteristics and Phase 4's per-client participation metrics.

---

## 9. Definition of Done

Phase 5 is complete when you can state, with numbers: *"At an average client bandwidth of B kbps combined with [compression setting], the system retains Z% of the Phase 2 accuracy ceiling, and this held even though C% of clients spent more than half their time disconnected."* If bandwidth is only referenced as an abstract compression ratio rather than an actual simulated kbps constraint feeding real transmission-time computation, the phase isn't done.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Connectivity engine produces implausible traces (e.g., everyone disconnected constantly, or no meaningful variation) | Visually inspect trace plots (Section 6, first metric) before running any full experiment on top of them — this is a fast, cheap check that prevents wasted compute on broken traces |
| Temptation to hand-tune the aggregation strategy while building this phase | Resist — if the interface from Phase 4 was built correctly, no aggregation-side changes should be necessary; treat any such urge as a signal to revisit the interface boundary instead |
| Bandwidth-to-time conversion ignores that Phase 3's compression settings change payload size per client differently | Make sure the conversion always uses the actual per-run compressed payload size, not a fixed assumed size, so the bandwidth × compression interaction (Section 5) is measured correctly rather than approximated |
| Real-world trace sourcing (Approach B) consumes time without adding proportional value | Time-box any attempt at Approach B; fall back cleanly to the documented Approach A if a suitable trace dataset isn't found quickly |
| Joint bandwidth × compression × dropout sweep becomes combinatorially large | Defer the full 3-way systematic sweep to Phase 6's dedicated experiment orchestration; this phase only needs to demonstrate the engine works and show one clear joint interaction example |

---

## 11. Next Phase Preview

**Phase 6 — Evaluation & Experiment Orchestration** will take all the pieces built so far (compression from Phase 3, async aggregation from Phase 4, connectivity simulation from Phase 5) and run them together as a **systematic, orchestrated sweep** across compression settings, dropout rates, and bandwidth caps, producing the final joint metric your proposed direction slide names directly: accuracy retained at X% bandwidth, Y% dropout — organized, logged, and ready to feed both the final report and Phase 7's dashboard.
