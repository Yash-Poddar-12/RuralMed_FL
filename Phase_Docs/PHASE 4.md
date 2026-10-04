# Phase 4 — Straggler-Tolerant Asynchronous Aggregation & Dropout Modeling

**Project:** Communication-Efficient Federated Learning for Diagnostic Imaging under Intermittent Rural Connectivity
**Team:** Pavan D Umesh (23BDS0046) · Aayush Sood (23BDS0177) · Yash Poddar (23BDS0195)
**Phase:** 4 of 8
**Estimated Duration:** 2–2.5 weeks
**Depends on:** Phase 2 (FedAvg baseline), Phase 3 (compression, optional but should remain composable)

---

## 1. Where This Fits

Phases 2–3 assumed every simulated clinic participates fully, every round, with no delays. That assumption is exactly what your own Group C literature (FedBuff, BASecAgg, Xu et al.) identifies as unrealistic for real cross-device FL — and it's the opposite of "intermittent rural connectivity," the exact scenario this project is named after. Phase 4 removes that assumption: clients can now be **slow (stragglers)** or **absent entirely (dropout)**, and synchronous FedAvg — which must wait for every selected client every round — is replaced with a **buffered asynchronous aggregation** strategy that keeps making progress despite this.

This phase deliberately does **not** yet simulate realistic bandwidth/connectivity traces — that's Phase 5. Here, straggler delay and dropout are modeled with simple, tunable probability distributions, so the aggregation mechanism itself can be built and validated before layering more realistic connectivity modeling on top of it.

| Phase | Name | Status |
|---|---|---|
| 1 | Foundation: Env, Repo, Data | ✅ done |
| 2 | FL Core Baseline (FedAvg) | ✅ done |
| 3 | Communication-Efficient Compression | ✅ done |
| **4** | **Straggler-Tolerant Async Aggregation** | **this file** |
| 5 | Intermittent Connectivity Engine | depends on Phase 4 |
| 6 | Evaluation & Experiment Orchestration | depends on 3–5 |
| 7 | Monitoring Dashboard | depends on Phase 6 |
| 8 | Integration, Ablations & Final Report | depends on all |

---

## 2. Objectives of Phase 4

By the end of this phase you should be able to answer "yes" to all of the following:

- [ ] A buffered asynchronous aggregation strategy (FedBuff-style) is implemented as an alternative to Phase 2's synchronous FedAvg, switchable via config.
- [ ] Client staleness (how out-of-date a client's local model was when it started training, relative to the current global model) is tracked and used to weight that client's contribution during aggregation.
- [ ] A configurable straggler/dropout model exists, independent of the aggregation strategy itself, so it can later be swapped for Phase 5's more realistic connectivity traces without rewriting the aggregation logic.
- [ ] You can directly compare synchronous vs. asynchronous aggregation on **wall-clock training efficiency** under the same dropout/straggler conditions — reproducing, on your own data, the kind of efficiency claim FedBuff makes in the literature (3.3x over sync FL).
- [ ] Accuracy under async aggregation with dropout is compared against both the Phase 2 no-dropout ceiling and a naive synchronous-FedAvg-with-dropout baseline, so the actual benefit of asynchrony (not just "having compression" or "having non-IID data") is isolated.

---

## 3. Straggler & Dropout Model Design

Build this as a **separate, pluggable component** from the aggregation strategy — Phase 5 will replace this simple model with connectivity-trace-driven behavior, and the aggregation logic in Section 4 should not need to change when that happens.

| Parameter | Role |
|---|---|
| Per-round dropout probability | Probability a selected client fails to return an update at all this round (models a clinic losing connectivity entirely mid-round) |
| Per-client local-training/communication delay distribution | Models straggling — some clients take much longer to finish a round than others (long-tailed distribution recommended, e.g. lognormal, rather than uniform, since a few very slow clients is more realistic than mild uniform variance) |
| Correlation with Phase 1's client data-volume imbalance | Clients with larger data partitions (Phase 1, Section 7.2) plausibly take longer per local epoch — consider linking delay distribution parameters to partition size rather than assigning delay fully at random, since this is a more defensible simulation choice for the final report |
| Dropout rate as a sweep parameter | This becomes the "Y% dropout" axis of your target metric from the proposed direction slide — must be easy to vary via config across multiple values (e.g., 0%, 10%, 25%, 40%, 60%) |

**Decision to record:** whether dropout, once it happens to a client in a given round, means that client's partial local progress is fully discarded (simpler, matches most cited papers' assumption) or preserved for a future round. Discarding is the safer, more literature-consistent default — note it explicitly.

---

## 4. Buffered Asynchronous Aggregation Design (FedBuff-style)

### 4.1 Core mechanism

Instead of the server waiting for a fixed round of selected clients to all finish (as in Phase 2), the server now:
1. Continuously dispatches the current global model to available clients as they become free.
2. Accepts client updates **as they arrive**, regardless of which "round" they logically started from.
3. Accumulates incoming updates into a **buffer**; once the buffer reaches a configured size *K* (a tunable concurrency parameter), the server aggregates the buffered updates into the global model and clears the buffer, immediately continuing to accept new updates.
4. Never blocks waiting on a specific straggling client — a slow or dropped client simply contributes later (or not at all, if it drops), without stalling everyone else's progress.

This directly targets the problem statement in your Group C literature: "stragglers slow synchronous FL; full asynchronous FL breaks secure aggregation" — buffering is the middle ground FedBuff proposes, and it's the mechanism to reproduce here.

### 4.2 Staleness weighting

Each buffered update was computed starting from some earlier version of the global model, which may have since been updated by other clients' contributions. The number of global updates that happened *since* that client started is its **staleness**. Weight each client's contribution during aggregation inversely by its staleness (a more stale update is trusted less), rather than treating all buffered updates identically — this is the detail that keeps buffered-async aggregation stable rather than degrading into noisy averaging of wildly out-of-date updates.

### 4.3 Buffer size (K) as a key tunable

| Buffer size | Effect |
|---|---|
| Small K | Faster global model updates (less waiting), but each aggregation step is based on fewer clients — potentially noisier |
| Large K | Smoother aggregation, closer in spirit to synchronous FedAvg, but slower to update and more exposed to staleness if stragglers are numerous |

Sweep at least 3 buffer sizes and report the accuracy/wall-clock tradeoff — this becomes another figure for the final report, parallel in spirit to Phase 3's compression-ratio sweep.

### 4.4 Compatibility with Phase 3's compression

Design the client update path so compression (Phase 3) and asynchronous buffering (this phase) are **independent, composable options**, not alternatives — a client's update can be both compressed and arrive asynchronously. Validate this combination works (doesn't crash, still converges) even if the full accuracy-vs-compression-vs-dropout 3-way sweep is deferred to Phase 6's more systematic experiment orchestration.

---

## 5. Baselines to Compare Against

This phase's evaluation needs **three** points of comparison, not just "async with dropout":

1. **Phase 2 ceiling:** synchronous FedAvg, 100% participation, no dropout (already have this).
2. **Naive degraded baseline:** synchronous FedAvg, but now with the same dropout/straggler model applied — i.e., what happens if you *don't* switch to async aggregation and just let dropout hurt the existing synchronous approach (rounds either wait indefinitely for stragglers, or you must decide a synchronous timeout policy — document whichever fallback you choose here, since this baseline needs to be well-defined to be a fair comparison).
3. **This phase's buffered async aggregation:** same dropout/straggler conditions as #2, but using the FedBuff-style strategy from Section 4.

The entire point of this phase is demonstrating #3 outperforms #2 (in wall-clock efficiency and/or final accuracy) under identical dropout conditions, while ideally approaching #1's accuracy despite the dropout.

---

## 6. Metrics to Track

In addition to the accuracy/AUROC/F1 and communication-volume metrics carried over from Phases 2–3:

| Metric | Why it matters here |
|---|---|
| Wall-clock time to reach a target accuracy threshold | The efficiency claim (async vs. sync) is fundamentally a wall-clock comparison, not just a final-accuracy comparison |
| Staleness distribution across aggregated updates | Diagnostic — reveals whether your buffer size and dropout settings are producing reasonable staleness levels or pathologically stale ones |
| Effective client participation rate over the full run | Under dropout, some clients may end up contributing far fewer times than others over the whole training run — worth reporting, especially cross-referenced with Phase 1's per-client data volumes |
| Accuracy retained relative to Phase 2 ceiling, at each dropout rate | This is your project's core "Y% dropout" target metric axis, now with a real, measured curve |

---

## 7. Repository Additions for This Phase

```
research/
├── src/
│   ├── fl_core/
│   │   ├── aggregation/
│   │   │   ├── fedbuff_strategy.py     # Buffered async aggregation + staleness weighting
│   │   │   └── sync_with_dropout.py    # Naive synchronous-with-dropout baseline (Section 5, #2)
│   │   └── straggler_model/
│   │       ├── dropout_sampler.py      # Per-round dropout probability model
│   │       └── delay_sampler.py        # Per-client straggler delay distribution (pluggable — Phase 5 replaces this)
├── configs/
│   ├── phase4_async_bufferK_sweep.yaml
│   └── phase4_dropout_rate_sweep.yaml
└── results/
    └── phase4/                          # Wall-clock comparisons, staleness plots, dropout-rate accuracy curves
```

*(Structure and responsibilities only — no code included per your instructions.)*

---

## 8. Deliverables Checklist

- [ ] Dropout and straggler-delay models implemented as pluggable, config-driven components, independent of aggregation logic.
- [ ] Buffered asynchronous aggregation strategy implemented, including staleness-weighted contribution.
- [ ] Naive synchronous-with-dropout baseline implemented and clearly documented (including its timeout/fallback policy).
- [ ] Buffer size (K) swept across at least 3 values, with accuracy/wall-clock tradeoff reported.
- [ ] Dropout rate swept across at least 4–5 values (e.g., 0/10/25/40/60%).
- [ ] All three comparison points from Section 5 run under matched dropout conditions.
- [ ] Confirmed compatibility (no crashes, still converges) when Phase 3 compression is enabled simultaneously with async aggregation.
- [ ] Wall-clock efficiency of async vs. sync-with-dropout reported as a ratio (mirroring FedBuff's own "3.3x more efficient" framing, but with your own measured number).
- [ ] Staleness distribution and effective per-client participation rate logged and reviewed.

---

## 9. Definition of Done

Phase 4 is complete when you can state, with numbers: *"Under Y% dropout, buffered async aggregation reaches [target accuracy] in [wall-clock time], versus [wall-clock time] for synchronous FedAvg under the same dropout — a Z× efficiency gain — while retaining W% of the no-dropout accuracy ceiling."* If dropout is only "handled" in the sense that the code doesn't crash, but no efficiency/accuracy comparison exists, the phase isn't done.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Async aggregation converges but to a worse accuracy than expected | Check the staleness-weighting mechanism first — unweighted or incorrectly weighted stale updates are the most common cause of async FL instability |
| Naive synchronous-with-dropout baseline is unfairly weak or unfairly strong depending on your chosen timeout policy | Document the timeout/fallback policy explicitly (Section 5) so the comparison's fairness is auditable, not implicit |
| Dropout model and straggler-delay model get tangled with the aggregation logic, making Phase 5's connectivity engine hard to plug in | Enforce the pluggable-component boundary from Section 3 strictly — the aggregation strategy should call the straggler model as an interface, not contain its logic inline |
| Combining async aggregation with Phase 3 compression breaks silently | Explicitly test this combination, even briefly, in this phase rather than discovering incompatibility during Phase 6's full sweep |
| Buffer size sweep and dropout-rate sweep multiply into too many runs for the time available | Prioritize the dropout-rate sweep at a single reasonable buffer size first (it directly serves the project's core target metric); treat the full buffer-size sweep as secondary if time is tight |

---

## 11. Next Phase Preview

**Phase 5 — Intermittent Connectivity Engine** will replace this phase's simple probability-based dropout/delay model with **realistic, trace-driven connectivity simulation** (bandwidth caps, latency, and connection-loss patterns modeled after real rural/intermittent network behavior), plugged into the exact same pluggable straggler-model interface built in Section 3 — so the aggregation strategy from this phase does not need to change, only the thing feeding it delay and dropout events.
