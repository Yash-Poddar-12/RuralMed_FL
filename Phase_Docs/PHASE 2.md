# Phase 2 — FL Core Baseline: FedAvg Simulation on Partitioned Clinics

**Project:** Communication-Efficient Federated Learning for Diagnostic Imaging under Intermittent Rural Connectivity
**Team:** Pavan D Umesh (23BDS0046) · Aayush Sood (23BDS0177) · Yash Poddar (23BDS0195)
**Phase:** 2 of 8
**Estimated Duration:** 1.5–2 weeks
**Depends on:** Phase 1 (processed datasets, non-IID client partitions, environment, experiment tracking)

---

## 1. Where This Fits

Phase 2 has one job: build the **plain FedAvg baseline** — no compression, no async aggregation, no dropout, no bandwidth constraints — trained across the simulated clinics from Phase 1. This is the McMahan et al. (2017) baseline from your literature review's Group A, applied to your own non-IID medical-imaging partitions instead of the toy benchmarks it was originally tested on.

Everything from Phase 3 onward (quantization, sparsification, async aggregation, connectivity simulation) is measured **against this baseline**. If Phase 2 is sloppy — flaky convergence, unlogged hyperparameters, no reproducibility — every later comparison in the final report becomes unreliable. This phase is where rigor gets established, not where corners get cut because "it's just the baseline."

| Phase | Name | Status |
|---|---|---|
| 1 | Foundation: Env, Repo, Data | ✅ done |
| **2** | **FL Core Baseline (FedAvg)** | **this file** |
| 3 | Communication-Efficient Compression | depends on Phase 2 |
| 4 | Straggler-Tolerant Async Aggregation | depends on Phase 2 |
| 5 | Intermittent Connectivity Engine | depends on Phase 4 |
| 6 | Evaluation & Experiment Orchestration | depends on 3–5 |
| 7 | Monitoring Dashboard | depends on Phase 6 |
| 8 | Integration, Ablations & Final Report | depends on all |

---

## 2. Objectives of Phase 2

By the end of this phase you should be able to answer "yes" to all of the following:

- [ ] A working Flower simulation trains a shared global model across all 8–15 simulated clinics from Phase 1, using unmodified FedAvg.
- [ ] The global model's accuracy/AUROC on the centralized held-out test set is tracked round-by-round and converges to a sensible, literature-consistent number (not near-random, not suspiciously perfect).
- [ ] Every run's hyperparameters, seed, and results are logged automatically — nothing lives only in someone's terminal history.
- [ ] Total communication cost (bytes transferred per round, uncompressed) is measured and logged — this number becomes the baseline that Phase 3's compression ratios are computed against.
- [ ] The simulation runs end-to-end without manual babysitting (i.e., you can kick off a run and walk away).

---

## 3. Model Architecture Decision

| Decision | Recommendation | Why |
|---|---|---|
| Backbone | ResNet-18 or DenseNet-121, ImageNet-pretrained | Both are the standard backbones used across the medical-imaging FL papers in your Group D literature (COVID CXR study, radiograph FL study) — using the same family keeps your results comparable to cited work. DenseNet-121 is closer to CheXpert's own original baseline; ResNet-18 is lighter and trains faster under repeated federated rounds — pick ResNet-18 first for iteration speed, and treat DenseNet-121 as an optional Phase 8 ablation if time allows. |
| Output layer | 3-class softmax (Normal / Pneumonia / COVID-19) | Matches the unified taxonomy fixed in Phase 1, Section 6.1 |
| Pretraining | ImageNet weights, fine-tuned end-to-end | Federated training is round- and sample-constrained; starting from ImageNet weights meaningfully speeds convergence versus training from scratch, which matters when every round costs simulated "bandwidth" |
| Loss function | Weighted cross-entropy | Class imbalance is expected and intentional (non-IID clinics) — use per-client or global class weights, and record which one was chosen in the decisions log, since it affects every later comparison |

**Decision to record:** freeze vs. fine-tune backbone layers. Recommendation: fine-tune the full network — freezing early layers reduces communication cost (smaller model = less to send in later compression phases isn't the point here, since Phase 2 has no compression yet) but tends to underperform on domain-shifted medical images versus natural ImageNet images.

---

## 4. Flower Simulation Architecture

### 4.1 Simulation vs. real distributed deployment

Use Flower's **simulation mode** (virtual clients running in-process/in-process-pool, not real separate machines) for the entire project. This is the correct choice, not a shortcut — real distributed deployment would require actual rural-clinic-like hardware you don't have, and simulation is exactly how the FedBuff, BASecAgg, and radiograph-FL papers you reviewed validated their methods too.

### 4.2 Components to design (conceptually, before writing any training code)

| Component | Responsibility |
|---|---|
| **ClientApp definition** | Wraps one simulated clinic: loads only that client's partition (from Phase 1's index files), runs local training for a fixed number of local epochs, returns updated model weights + local metrics + local sample count |
| **ServerApp / Strategy** | Implements FedAvg: selects a fraction of clients each round, waits for their updates, performs weighted averaging (weighted by each client's local sample count — critical given Phase 1's deliberately imbalanced client sizes), and broadcasts the new global model |
| **Client resource allocation** | Since dozens of virtual clients share the same physical GPU/CPU during simulation, explicitly configure how much compute each virtual client is allowed (Flower's resource-per-client settings) so runs don't silently thrash or OOM |
| **Round-level evaluation hook** | After each round's aggregation, evaluate the new global model on the **centralized test set held out in Phase 1** (never on client data) and log the result |

### 4.3 Federated Training Loop — Parameters to Fix and Record

| Parameter | Starting recommendation | Note |
|---|---|---|
| Total rounds | 50–100 | Enough to observe a convergence curve; adjust after the first pilot run |
| Clients sampled per round | 100% of clients for the pure baseline | Later phases (4–5) will vary this to model dropout — Phase 2 should establish the "everyone always participates" ceiling first |
| Local epochs per round | 1–3 | Too many local epochs on non-IID data causes client drift; 1 epoch is the closest match to the original FedAvg paper's most stable setting |
| Batch size | 16–32 (constrained by image resolution and available GPU memory) | Document actual value used — it affects both convergence and (in later phases) communication timing assumptions |
| Optimizer | SGD with momentum, or Adam | Record learning rate and any decay schedule explicitly — federated convergence is sensitive to this |
| Random seed | Fixed, and swept over ≥3 seeds for the final reported baseline number | A single-seed FedAvg run is not a reportable result — average across seeds before treating this as "the baseline" |

---

## 5. Communication Cost Instrumentation (Important — Do Not Skip)

Phase 2 has no compression, but it must still **measure** the uncompressed communication cost every round: total bytes of model parameters sent client→server and server→client. This number is the denominator for every "communication efficiency" claim your later phases and final report will make (e.g., "Phase 3's quantization reduced communication by 8x relative to the Phase 2 baseline"). If this isn't measured now, Phase 3 has nothing to compare against.

Log per round:
- Model size in bytes (uncompressed, full precision)
- Number of participating clients × model size = total round communication volume
- Cumulative communication volume across all rounds so far

---

## 6. Evaluation Protocol

- **Global metric:** Accuracy and macro-averaged AUROC/F1 on the centralized test set (macro-averaging matters because of class imbalance — plain accuracy can look good while the minority class, likely COVID-19, is barely learned).
- **Per-client diagnostic:** Optionally evaluate the global model separately on each client's *local* held-out data too — this reveals whether some simulated clinics benefit far less from the shared model than others, which is a finding worth carrying into the final report given the literature's noted radiograph-FL failure case (FL trailing local training under severe heterogeneity).
- **Convergence sanity check:** Before trusting the full non-IID run, run one pilot with an **IID random split** of the same data (quick, throwaway partitioning, not the Phase 1 non-IID one) to confirm the training pipeline itself is correct and FedAvg converges as expected in the easy case. If it doesn't converge even under IID conditions, the bug is in your training/aggregation code, not in non-IID difficulty — debug there first.

---

## 7. Experiment Tracking Integration

Every run in this phase must log automatically (continuing the discipline set up in Phase 1, Section 8):
- Full hyperparameter set (rounds, local epochs, learning rate, batch size, seed, model architecture)
- Per-round global test accuracy/AUROC/F1
- Per-round communication volume (Section 5)
- Wall-clock time per round (useful later for comparing against async aggregation's efficiency claims in Phase 4)

Tag every run with a config hash or run name that ties back to its YAML config file in `research/configs/`, so any result in the eventual report can be traced back to an exact, rerunnable configuration.

---

## 8. Repository Additions for This Phase

```
research/
├── src/
│   ├── fl_core/
│   │   ├── models/            # Model architecture definitions (ResNet-18/DenseNet-121 wrappers)
│   │   ├── client.py           # ClientApp: local training + evaluation logic
│   │   ├── server.py           # ServerApp: FedAvg strategy, aggregation, round orchestration
│   │   └── comms_tracker.py    # Communication-volume instrumentation (Section 5)
│   └── eval/
│       └── metrics.py          # Accuracy/AUROC/F1 computation, macro-averaging
├── configs/
│   └── phase2_fedavg_baseline.yaml
└── results/
    └── phase2/                 # Logged run outputs, convergence plots, per-seed results
```

*(File names above describe responsibilities only — no code is included per your instructions; this is the structure to build against.)*

---

## 9. Deliverables Checklist

- [ ] Flower ClientApp and ServerApp implementing unmodified FedAvg, running against Phase 1's non-IID partitions.
- [ ] IID pilot run completed and confirmed to converge (sanity check, Section 6).
- [ ] Full non-IID baseline run completed across ≥3 random seeds.
- [ ] Convergence curves (global accuracy/AUROC vs. round) plotted and saved.
- [ ] Per-client evaluation breakdown produced and reviewed for any clinic performing far worse than the global average.
- [ ] Per-round and cumulative communication cost logged for every run.
- [ ] All runs tracked in the experiment tracker with full config provenance.
- [ ] Baseline numbers (accuracy, AUROC, F1, total communication volume) written into `infra/docs/decisions-log.md` as the reference point for every later phase's comparisons.

---

## 10. Definition of Done

Phase 2 is complete when you can state, with a number and a citation to a specific tracked run: *"Our FedAvg baseline reaches X% test accuracy / Y macro-AUROC after Z rounds, at a total communication cost of W GB uncompressed, averaged over 3 seeds."* If that sentence can't be filled in with real, reproducible numbers yet, the phase isn't done.

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Model doesn't converge under non-IID partitioning | Confirm first via the IID pilot run (Section 6) to isolate whether the issue is the pipeline or genuine non-IID difficulty |
| Simulated clients thrash shared GPU memory | Explicitly cap per-client resource allocation in the Flower simulation config rather than letting all virtual clients contend freely |
| Baseline numbers change every time someone reruns it | Fix and record random seeds; treat any un-seeded run as exploratory, not reportable |
| Communication-volume tracking gets added as an afterthought later | Build it into the server/client loop now (Section 5) — retrofitting it after Phase 3's compression exists makes "before vs. after" comparisons unreliable |
| Class imbalance quietly collapses the minority (COVID-19) class | Watch per-class recall/F1, not just accuracy, from the very first pilot run — this mirrors exactly the failure mode called out in your Group D literature (radiograph FL, FedBB) |

---

## 12. Next Phase Preview

**Phase 3 — Communication-Efficient Compression** will take this exact FedAvg loop and insert a compression step (quantization and/or top-k sparsification, drawn from your Group A/B literature) into the client→server update path, measuring how much the communication volume from Section 5 shrinks and how much (if any) accuracy is traded away for that reduction.
