# Phase 3 — Communication-Efficient Compression: Quantization & Sparsification

**Project:** Communication-Efficient Federated Learning for Diagnostic Imaging under Intermittent Rural Connectivity
**Team:** Pavan D Umesh (23BDS0046) · Aayush Sood (23BDS0177) · Yash Poddar (23BDS0195)
**Phase:** 3 of 8
**Estimated Duration:** 2 weeks
**Depends on:** Phase 2 (working FedAvg baseline, communication-volume instrumentation, tracked baseline numbers)

---

## 1. Where This Fits

Phase 3 takes the exact FedAvg loop validated in Phase 2 and inserts a **compression step** into the client→server (and optionally server→client) update path. This is where the project starts to directly build on your Group A and Group B literature — QSGD, signSGD, FedPAQ (quantization) and Deep Gradient Compression / Top-k sparsification — rather than just citing them.

The uplink from a rural clinic is the realistic bottleneck (clinics sending large model updates over poor connections), so **client→server compression is the priority**; server→client (broadcast) compression is a secondary, optional extension once the primary direction works.

| Phase | Name | Status |
|---|---|---|
| 1 | Foundation: Env, Repo, Data | ✅ done |
| 2 | FL Core Baseline (FedAvg) | ✅ done |
| **3** | **Communication-Efficient Compression** | **this file** |
| 4 | Straggler-Tolerant Async Aggregation | depends on Phase 2 |
| 5 | Intermittent Connectivity Engine | depends on Phase 4 |
| 6 | Evaluation & Experiment Orchestration | depends on 3–5 |
| 7 | Monitoring Dashboard | depends on Phase 6 |
| 8 | Integration, Ablations & Final Report | depends on all |

---

## 2. Objectives of Phase 3

By the end of this phase you should be able to answer "yes" to all of the following:

- [ ] At least one quantization method and one sparsification method are implemented and independently switchable via config, without touching the FedAvg loop's core logic.
- [ ] For every compressed run, you can report an exact compression ratio against the Phase 2 uncompressed baseline (bytes-per-round, not an estimate).
- [ ] For every compression setting swept, you can report the accuracy/AUROC/F1 cost relative to the Phase 2 baseline — i.e., you have real accuracy-vs-bandwidth tradeoff curves, not just compression working "in principle."
- [ ] Sparsification includes error feedback (residual accumulation), matching the DGC approach from your Group B literature — omitting this is a known cause of divergence, not an optional nicety.
- [ ] Results are logged with full provenance, comparable directly against Phase 2's tracked baseline numbers.

---

## 3. Which Techniques to Implement — and in What Order

| Priority | Technique | Source in your literature review | Why this order |
|---|---|---|---|
| 1st | **Top-k sparsification with error feedback** (Deep Gradient Compression style) | Lin et al., ICLR 2018 (Group B) | Highest compression ratios reported in your own review (up to 600x); most directly motivated by "mobile devices face low bandwidth and intermittent connections" — the closest existing method to your project's actual framing |
| 2nd | **Stochastic quantization** (QSGD-style, tunable bit-width) | Alistarh et al., NeurIPS 2017 (Group A) | Simpler to implement correctly than sparsification (no residual buffer needed), useful as a second, independently comparable compression axis |
| 3rd (optional, time-permitting) | **1-bit sign-based compression** (signSGD-style) | Bernstein et al., ICML 2018 (Group A) | Cited in your review as extreme compression with fault tolerance — worth having as an extreme data point on your accuracy-vs-bandwidth curve, but only after the two above are solid |

Implement and validate #1 fully before starting #2 — trying to build both compression paths simultaneously makes it hard to tell which one introduced a bug if convergence breaks.

---

## 4. Sparsification Design (Top-k + Error Feedback)

### 4.1 Core mechanism

Each client, instead of sending its full model update (all weight/gradient values), sends only the **top-k largest-magnitude entries** (by absolute value) of its update, plus their indices. Everything not sent is **not discarded** — it is accumulated locally into a **residual buffer** and added back into the *next* round's update before top-k selection happens again. This residual/error-feedback step is what the DGC paper (and the follow-up "Rethinking Gradient Sparsification" paper in your Group B) identifies as essential — without it, small-but-consistent update directions never accumulate enough magnitude to ever get selected, and training silently stalls in those directions.

### 4.2 Parameters to expose and sweep

| Parameter | Role |
|---|---|
| Sparsification ratio (e.g., keep top 1%, 5%, 10% of values) | Primary knob controlling the accuracy-vs-bandwidth tradeoff — this becomes the "X% bandwidth" axis of your target metric from the proposed direction slide |
| Warmup rounds at lower sparsification (higher density) | DGC's own paper flags this as needed for early-training stability; decide whether your simulated setting can even assume "a few stable early rounds" — note that this assumption is explicitly listed as a *limitation* in your own Group B literature table, so treat warmup as a design tension to discuss in the final report, not a free assumption |
| Momentum correction (local momentum accumulated alongside the residual) | Improves convergence stability at high sparsification ratios; implement after the basic residual mechanism works, not simultaneously |
| Per-layer vs. global top-k selection | Per-layer top-k (selecting the top-k within each layer separately) is simpler to reason about and is the more common implementation choice — document whichever you pick |

### 4.3 Server-side handling

The server must reconstruct each client's sparse update (indices + values) into the correct positions in a full-size tensor before applying FedAvg's weighted averaging — averaging must correctly handle positions where *some* clients sent a value and others didn't (typically: treat missing positions as zero contribution from that client for that round, not as "ignore this client entirely").

---

## 5. Quantization Design (QSGD-style)

### 5.1 Core mechanism

Instead of sending full 32-bit floating-point values, each client stochastically quantizes its update to a lower bit-width (e.g., 8-bit, 4-bit, or fewer) per value, using a scale factor (typically the update vector's norm) plus a randomized rounding scheme so that the quantized value is an **unbiased estimator** of the original — this unbiasedness is exactly what gives QSGD its convergence guarantee in the original paper, so don't substitute naive (deterministic) rounding without noting that the theoretical guarantee no longer applies.

### 5.2 Parameters to expose and sweep

| Parameter | Role |
|---|---|
| Bit-width (e.g., 8, 4, 2 bits) | Primary compression-ratio knob for this technique, directly comparable to the sparsification ratio's role above |
| Quantization granularity (per-tensor vs. per-layer scale factor) | Per-layer scaling generally preserves accuracy better at aggressive bit-widths; note whichever is chosen |

### 5.3 Server-side handling

The server dequantizes each incoming update using the transmitted scale factor before performing FedAvg averaging — the scale factor itself must also be transmitted (a small number, negligible added communication) since the server cannot dequantize without it.

---

## 6. Communication Cost Accounting — Extending Phase 2's Instrumentation

Reuse the exact instrumentation built in Phase 2 (Section 5 of that file), but now compute the **actual transmitted size** of the compressed payload, not the original model size:

- For sparsification: bytes = (number of kept indices × index size) + (number of kept values × value size), compared directly against Phase 2's full-model byte count for the same round.
- For quantization: bytes = (number of values × bit-width, packed) + scale-factor overhead, compared the same way.
- Compute and log **compression ratio = Phase 2 baseline bytes / Phase 3 compressed bytes** for every run — this ratio is the headline number for this phase and feeds directly into the "accuracy retained at X% bandwidth" target metric from your proposed direction slide.

---

## 7. Evaluation Protocol — Accuracy-vs-Bandwidth Tradeoff

This phase's primary deliverable is a curve, not a single number:

1. Fix everything from the Phase 2 baseline config (rounds, local epochs, learning rate, seeds, client sampling — unchanged, 100% participation, no dropout yet — that's Phase 4/5's job).
2. Sweep sparsification ratio across at least 4–5 settings (e.g., 50%, 20%, 10%, 5%, 1% density).
3. Sweep quantization bit-width across at least 3–4 settings (e.g., 16-bit, 8-bit, 4-bit, 2-bit).
4. For every setting, record: final/converged test accuracy, macro-AUROC/F1, compression ratio, and per-round communication volume.
5. Plot **accuracy (y-axis) vs. compression ratio or bandwidth reduction (x-axis)** for both techniques on the same chart — this is the figure your literature review's Slide 6 "Comparison" table was building toward, now with your own results instead of just citing others'.
6. Identify the "knee" of the curve — the point past which accuracy degrades sharply — since this is exactly the practically useful operating point to highlight in the final report.

Also re-run the per-client evaluation breakdown from Phase 2, Section 6 at a couple of representative compression settings — check whether compression disproportionately hurts already-underperforming clinics (a plausible and reportable finding given your Group D literature's non-IID sensitivity concerns).

---

## 8. Repository Additions for This Phase

```
research/
├── src/
│   ├── fl_core/
│   │   ├── compression/
│   │   │   ├── sparsification.py    # Top-k selection + residual/error-feedback buffer management
│   │   │   ├── quantization.py      # Stochastic quantization + dequantization
│   │   │   └── comms_accounting.py  # Compressed-payload byte accounting (extends Phase 2's tracker)
│   │   ├── client.py                # Updated: applies compression before returning update
│   │   └── server.py                # Updated: reconstructs/dequantizes before FedAvg averaging
├── configs/
│   ├── phase3_sparsification_sweep.yaml
│   └── phase3_quantization_sweep.yaml
└── results/
    └── phase3/                       # Per-setting run outputs, accuracy-vs-bandwidth plots
```

*(Structure and responsibilities only — no code included per your instructions.)*

---

## 9. Deliverables Checklist

- [ ] Top-k sparsification with error feedback implemented, switchable via config, validated against a known-good low-sparsification setting first (should closely match Phase 2's baseline accuracy when almost nothing is dropped).
- [ ] Stochastic quantization implemented, switchable via config, similarly validated at a high bit-width first.
- [ ] Compressed-payload byte accounting implemented and validated (spot-check the reported bytes against a manual calculation for at least one run).
- [ ] Full sparsification-ratio sweep completed and logged.
- [ ] Full quantization-bit-width sweep completed and logged.
- [ ] Accuracy-vs-compression-ratio plot produced for both techniques on one chart.
- [ ] "Knee point" of acceptable compression identified and written down with a number.
- [ ] Per-client breakdown re-examined at representative compression settings for disproportionate impact on any clinic.
- [ ] All results traceable to config files and comparable directly against Phase 2's tracked baseline numbers.

---

## 10. Definition of Done

Phase 3 is complete when you can state, with numbers: *"At Y% sparsification / N-bit quantization, we retain Z% of baseline accuracy while reducing communication volume by W×, and beyond that point accuracy degrades sharply."* If you can only say compression "works" without a quantified tradeoff curve, the phase isn't done.

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Sparsification diverges or stalls entirely | Almost always means the error-feedback/residual buffer is missing or implemented incorrectly — validate it in isolation (confirm residuals are actually accumulating and being reincorporated) before blaming the sparsification ratio itself |
| Quantization silently biases updates, hurting convergence | Confirm the quantization scheme is stochastic/unbiased, not naive rounding, per Section 5.1 |
| Compression ratio numbers don't match theoretical expectations | Validate byte accounting manually on one small run before trusting the full sweep |
| High-compression settings look fine on global accuracy but hide per-client damage | Always cross-check the per-client breakdown (Section 7, step 6), not just the global metric |
| Sweep takes too long given Phase 2's per-run time budget | Prioritize the sparsification sweep (Priority 1 technique) fully before spending equal time on quantization; treat signSGD (Priority 3) as optional and cuttable if time runs short |

---

## 12. Next Phase Preview

**Phase 4 — Straggler-Tolerant Async Aggregation** will keep this phase's compression techniques available as an option, but shift focus to the *timing* dimension: some simulated clinics will now be slow or drop out entirely mid-round, and the FedAvg loop will be replaced with a buffered asynchronous aggregation strategy (drawing on FedBuff and the Xu et al. non-IID + dropout paper from your Group C literature) that can tolerate this without stalling.
