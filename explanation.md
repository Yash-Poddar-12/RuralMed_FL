# RuralMed-FL: Complete Project Explanation

## 1. The project in one paragraph

This project studies **communication-efficient federated learning (FL) for chest X-ray diagnosis when rural clinics have unreliable, slow internet connections**.

The machine-learning task is three-class image classification:

- `Normal`
- `Pneumonia`
- `COVID-19`

Instead of collecting every clinic's X-rays in one central database, the project simulates ten rural clinics. Each clinic keeps its own training images, trains a copy of the model locally, and sends only a model update to a coordinating server. The server combines the updates into a better global model. The project then asks:

1. How accurate is federated training on heterogeneous clinic data?
2. How much communication can be saved by compressing model updates?
3. Can training continue when clinics are slow or disconnected?
4. What happens when bandwidth, compression, dropout, and asynchronous aggregation interact?

The repository implements this research pipeline in Python, stores normalized experiment results for Supabase, and provides a Next.js/Express dashboard for exploring those results.

The project name used in several artifacts is **RuralMed-FL**; the root README describes the same work as **Communication-Efficient Federated Learning for Diagnostic Imaging**.

---

## 2. Why this problem matters

Medical imaging is data-heavy. A modern image-classification model may contain millions of parameters, so one training update can be tens of megabytes. A rural clinic may have:

- a smaller and different patient population;
- a different mix of diseases;
- fewer images than a large hospital;
- slower or shared cellular connectivity;
- temporary outages or power/network failures.

Centralizing every image creates privacy, governance, storage, and bandwidth problems. Federated learning is attractive because the images remain at the clinic while model knowledge is shared.

However, ordinary federated learning often assumes that clients can all participate reliably and quickly. That assumption is exactly what this project challenges. A method that works on a fast laboratory network may fail when a clinic cannot upload an 86 MB update before the round deadline.

The central research idea is therefore:

> Combine update compression, dropout-tolerant aggregation, and realistic intermittent-connectivity simulation, then measure diagnostic performance and communication cost together.

---

## 3. Federated learning in beginner terms

### Centralized machine learning

In ordinary centralized training:

1. Images and labels are collected in one place.
2. A model makes predictions.
3. The loss function measures mistakes.
4. Backpropagation computes how the weights should change.
5. An optimizer updates the weights.
6. This repeats for many batches and epochs.

### Federated learning

In this project, the images are divided among simulated clinics:

1. The server sends the current global model to selected clinics.
2. Each clinic trains locally on its own images.
3. Each clinic returns its updated model parameters.
4. The server averages the returned models, weighted by the number of local training images.
5. The new global model is sent out for the next round.

If client $i$ has $n_i$ training examples and returns model parameters $w_i$, standard FedAvg computes approximately:

$$
w_{global} = \sum_i \frac{n_i}{\sum_j n_j} w_i.
$$

The weight matters because the repository deliberately gives clinics unequal amounts of data. A clinic with 20,000 training images should not have exactly the same influence as one with 1,000 images.

Federated learning does **not** mean that the model learns without data. It means that the data stays local and the parameter updates are exchanged instead.

---

## 4. End-to-end architecture

```text
Raw public datasets
        |
        v
Data acquisition and verification
        |
        v
Label harmonization -> resize/normalize -> patient-safe split
        |
        v
Non-IID clinic partitions (10 simulated rural clinics)
        |
        v
DenseNet-121 global model
        |
        +--> local client training
        |        |
        |        +--> optional Top-K or QSGD compression
        |        +--> bandwidth/connectivity delay and dropout
        |
        v
FedAvg or FedBuff aggregation
        |
        v
Centralized held-out test evaluation
        |
        v
MLflow/local JSON -> Phase 6 summaries -> Supabase
                                      |
                                      v
                              Express API -> Next.js dashboard
```

The main implementation areas are:

| Area | Location | Responsibility |
|---|---|---|
| Data preparation | `research/src/data_prep/` | Acquire, validate, harmonize, preprocess, split, and load images |
| Client partitioning | `research/src/partitioning/partition.py` | Create heterogeneous clinic index files |
| Model and FL core | `research/src/fl_core/` | DenseNet, clients, server, aggregation, communication tracking |
| Compression | `research/src/fl_core/compression/` | Top-K sparsification, error feedback, QSGD, byte accounting |
| Connectivity simulation | `research/src/simulation/` | Markov states, client profiles, bandwidth and dropout decisions |
| Evaluation | `research/src/eval/` | Sweeps, summaries, heatmaps, operating point, Supabase export |
| Experiment entry points | `research/scripts/` | Run training, plots, preparation, and Phase 6 sweeps |
| Database contract | `infra/supabase/` and `infra/docs/` | Schema, decisions, reports, and handoff documentation |
| Dashboard | `web/backend/` and `web/frontend/` | Read-only result API and browser UI |

---

## 5. Data: what is being learned from

### Sources

The planned/implemented public sources are:

- **NIH ChestX-ray14**: a large chest X-ray collection with many findings.
- **CheXpert-small**: a chest X-ray dataset with overlapping findings and uncertainty labels. Access requires approval; the repository treats missing approved files explicitly rather than silently pretending they exist.
- **COVID-QU-Ex**: images organized around Normal, Non-COVID pneumonia, and COVID-19.

Raw and processed medical data are ignored by Git. The repository instructions intentionally keep full acquisition and heavy processing on a teammate's larger drive. This laptop is intended to use an ignored approximately 300-image development sample.

### Label harmonization

The original datasets do not use one common label system. The preprocessing layer maps them into the project's three-class task:

- `No Finding` becomes `Normal`.
- Pneumonia and selected related findings become `Pneumonia`.
- COVID-19 remains `COVID-19`.
- Other findings are excluded from the primary three-class task but can remain in provenance metadata.
- CheXpert uncertainty is handled conservatively; uncertain labels are not silently converted into positive labels.

This is a research simplification. It makes the datasets usable together, but it is not equivalent to a clinical-grade diagnostic labeling process.

### Preprocessing

Images are converted to grayscale, resized to $224 \times 224$, replicated to three channels, converted to floating-point tensors, and normalized with ImageNet-style channel statistics. DenseNet expects three channels even though the original radiographs are grayscale.

The `ImagingDataset` loader resolves image IDs through `primary_manifest.csv`, checks that index files contain valid IDs, prevents image paths from escaping the processed-data directory, and returns a normalized tensor plus an integer class label.

### Leakage prevention

The split code groups related images using patient IDs and exact pixel hashes. It also removes duplicate pixels and excludes label conflicts. Train, validation, and centralized test sets are kept group-isolated where the metadata supports it.

The test set is **not** distributed to clients. It remains centralized and is used to evaluate the global model after aggregation. This is important: the reported global test metrics are not simply averages of the clinics' local training metrics.

### Non-IID clinic partitions

Each simulated clinic is intentionally different. The partitioner combines:

- **source skew**: a clinic prefers one dataset source;
- **label skew**: a Dirichlet draw gives each clinic a different class mixture;
- **volume imbalance**: lognormal volume targets create small and large clinics;
- **whole-group allocation**: patient/duplicate groups are kept together.

The repository validates that every train/validation image is assigned once, no test image is assigned to a clinic, and patient groups do not cross splits or clients. The representative full-data partition has ten clients with train sizes ranging from about 1,150 to 20,722 images, an approximately 18x range.

This non-IID design is essential. If every clinic had an identical random sample, the project would not really test the rural-clinic problem.

---

## 6. The model

### DenseNet-121 backbone

The classifier is a **DenseNet-121** model, chosen because it is a recognized chest-radiograph backbone and is related to the CheXNet/CheXpert ecosystem.

The original ImageNet classifier predicts 1,000 object categories. This project replaces it with:

```text
DenseNet-121 feature extractor (1024 features)
    -> Dropout(p=0.2)
    -> Linear(1024, 3)
```

The final three outputs are converted into class probabilities with softmax during evaluation. All layers are fine-tuned rather than freezing the backbone, because medical radiographs differ substantially from ordinary ImageNet photographs.

The model has roughly 6.96 million parameters and an uncompressed float32 model size of about 27.83 MB in the checked-in Phase 6 trace. The larger payload values used in Phase 5 interaction reports refer to update-payload accounting configurations, so payload size should always be read together with the specific artifact and run configuration.

### Loss and optimization

Training uses weighted cross-entropy. The checked-in Phase 2 configuration uses global class weights approximately:

```text
Normal:     0.555
Pneumonia:  1.116
COVID-19:   3.299
```

The larger COVID-19 weight compensates for class imbalance. The standard training configuration uses one local epoch per round, batch size 16, Adam with learning rate $10^{-4}$, weight decay $10^{-4}$, and automatic mixed precision when CUDA is available.

The client creates a fresh Adam optimizer each round. This is a deliberate stateless-client choice: optimizer momentum is not carried between rounds.

### GPU requirement

The current `XRayClient` resolves the device at import time and raises an error if CUDA is unavailable. DenseNet-121 training is considered too slow for CPU-only full experiments. The data-preparation and unit-test paths are designed to work on CPU, but full FL training expects a suitable GPU environment.

---

## 7. What happens in one training round

For the standard client/server path:

1. Flower selects clients according to the strategy configuration.
2. Each client receives the current global parameters.
3. The client loads its own `train.csv` partition and trains locally.
4. The client computes local loss and accuracy for diagnostics.
5. The client forms an update relative to the parameters it received.
6. Optional compression transforms that update into a smaller payload.
7. The connectivity adapter decides whether the payload can arrive within the time budget.
8. The server aggregates surviving updates.
9. A separate evaluation model scores the new global parameters on the centralized test split.
10. Accuracy, macro-AUROC, macro-F1, class sensitivity, communication, and round information are logged.

The server's centralized evaluator reports per-class diagnostics as well as overall metrics. That matters because accuracy can hide failure on a minority class.

---

## 8. Metrics and how to interpret them

### Accuracy

The fraction of test images classified correctly. It is easy to understand but can be misleading when classes are imbalanced.

### Precision, recall, and sensitivity

For a class, **sensitivity/recall** asks: of the truly positive cases, how many did the model detect? COVID-19 sensitivity is particularly important in this project because a model can have a respectable overall accuracy while missing many COVID-19 cases.

### F1 and macro-F1

F1 balances precision and recall. **Macro-F1** calculates F1 separately for each class and averages the classes equally. It prevents the majority class from dominating the summary.

### AUROC and macro-AUROC

AUROC measures ranking/discrimination across decision thresholds. A score of 0.5 is roughly random ranking; 1.0 is perfect ranking. Macro-AUROC averages one-vs-rest AUROC across the three classes, so each class receives equal importance.

### Communication volume

Communication is measured in bytes, separately for upload and download. The baseline is:

$$
\text{round bytes} = 2 \times \text{participating clients} \times \text{model bytes}.
$$

The compression ratio is:

$$
\text{compression ratio} = \frac{\text{uncompressed baseline bytes}}{\text{actual transmitted bytes}}.
$$

A ratio of $7\times$ means the compressed transfer is approximately one-seventh of the baseline transfer, not that accuracy is seven times higher.

### Uptime and dropout

Uptime is the fraction of simulated connectivity steps in which a clinic's update is usable. Dropout means the client did not contribute that round, either because it was disconnected or because transmission exceeded the round budget.

### Staleness

In buffered asynchronous aggregation, staleness is how many global updates occurred after a client started from an older global model. Higher staleness means the update may be less aligned with the current model.

---

## 9. Compression methods

### Top-K sparsification with error feedback

The client computes a dense update vector. Top-K keeps only the largest-magnitude entries and sends each kept value plus its index. For example, 1% Top-K keeps roughly 1% of the coordinates.

The unsent values are stored in a client-specific residual:

$$
r_{t+1} = (\Delta_t + r_t) - \text{TopK}(\Delta_t + r_t).
$$

On the next round, the residual is added back before selecting the next Top-K entries. This prevents small but persistent update directions from disappearing forever. Residuals can be persisted to disk because Flower simulation actors may be recreated.

The implementation preserves BatchNorm-related state from sparsification rather than sparsifying it. This is intended to avoid corrupting running statistics and numerical instability.

### Stochastic QSGD-style quantization

Quantization keeps the vector dense but uses fewer bits per coordinate. The implementation sends quantized integer levels plus an L2-norm scale factor. Randomized rounding aims to make the reconstructed update an unbiased estimator of the original update.

Configured options include 8, 4, 2, and 1 bit. Four-bit quantization sends about half a byte per coordinate, plus small metadata overhead.

### Exact byte accounting

The accounting module includes:

- full float32 bytes;
- sparse index bytes plus value bytes plus headers;
- packed quantized bits plus scale and headers.

The unit tests check these formulas analytically. The practical reported communication also includes preserved uncompressed BatchNorm state where applicable.

---

## 10. Aggregation and unreliable clients

### FedAvg baseline

Phase 2 uses Flower's sample-count-weighted FedAvg. This is the reference point: no compression, no connectivity penalty, and a regular synchronous aggregation loop.

The baseline matters because later claims need a denominator. Without knowing the accuracy and bytes of the plain system, “7x savings” or “accuracy retained” would not be meaningful.

### FedBuff-style buffered aggregation

The Phase 4 strategy collects client updates in a buffer of size $K$. Once $K$ updates are available, it aggregates them rather than requiring one fixed group of clients to all finish together.

The update weight is:

$$
w_i = n_i(1 + \tau_i)^{-\alpha},
$$

where $n_i$ is the client's sample count, $\tau_i$ is staleness, and $\alpha$ is configured as 0.5 in the main Phase 4/5 setup.

- Small $K$ updates the global model quickly but may be noisy.
- Large $K$ gives a smoother update but waits for more contributions.

The code supports a `buffer_size` sweep, with $K \in \{1,3,5\}$ represented in Phase 4 artifacts.

### Important implementation qualification

The repository uses Flower's simulation backend. The `FedBuffStrategy` documentation calls this **virtual asynchrony**: clients are selected and dropout is simulated within Flower's round-oriented execution, while the strategy buffers surviving updates. It is not a real multi-machine event-driven server receiving network packets at arbitrary wall-clock times. This is a valid simulation abstraction, but it should be described accurately in a report.

### Synchronous dropout comparison

`SyncWithDropout` provides a comparison strategy. It filters dropped clients and averages whichever survivors returned, without waiting for a full cohort. This fallback policy is explicitly documented because the quality of the comparison depends on what “synchronous under dropout” means.

---

## 11. Connectivity simulation

Phase 4 uses simple dropout probabilities. Phase 5 replaces that abstraction with a three-state Markov model:

| State | Meaning | Raw bandwidth range |
|---|---|---:|
| GOOD | Stable rural broadband/4G | 1,024 to 5,120 kbps |
| POOR | Congested or weak link | 64 to 512 kbps |
| DISCONNECTED | No usable connection | 0 kbps |

Each clinic has a transition matrix that defines how likely it is to move between states from one round to the next. The three profile tiers are:

- `stable`: large clinic, reliable link;
- `intermittent`: mid-sized clinic, fluctuating link;
- `unreliable`: small remote sub-centre, frequent outages.

The current design correlates larger data volume with better infrastructure. This is an explicit modeling assumption, not an observed fact about real hospitals.

For a payload of $B$ bytes and bandwidth $R$ bits/second:

$$
t_{transmit} = \frac{8B}{R}.
$$

If the state is disconnected, transmission time is infinite. If transmission time exceeds the configured round budget, the client is marked unavailable for that round. This makes compression directly affect connectivity outcomes: a smaller payload is more likely to finish before the deadline.

---

## 12. Phase-by-phase explanation

### Phase 1: Foundation, environment, and data

**Purpose:** build a reproducible base before attempting FL.

It establishes the repository layout, Python dependencies, data-root configuration, source adapters, label mapping, image preprocessing, patient-safe train/validation/test splitting, and non-IID clinic partitions. It also starts the Supabase schema planning and decisions log.

**Why it matters:** bad labels, data leakage, or irreproducible partitions would invalidate every later result. Phase 1 is also where the project protects raw medical data from Git and supports an offline dev sample.

**Evidence in this repository:** preparation tests, split and partition validation, `research/data_prep/`, `research/src/partitioning/`, Phase 1 configs, and Phase 1 reports.

### Phase 2: Plain FedAvg baseline

**Purpose:** establish the accuracy and communication ceiling/floor for later comparisons.

It introduces the DenseNet-121 client, Flower server, centralized evaluation, MLflow logging, and communication tracking. The checked-in Phase 2 configuration uses ten clients, one local epoch, Adam, mixed precision, and up to 100 rounds.

**Why it matters:** every efficiency claim needs to answer “compared with what?” Phase 2 supplies the uncompressed reference.

**Representative result:** the Phase 6 integration trace records an uncompressed 30-round run with test accuracy `0.7667`, macro-AUROC `0.9051`, macro-F1 `0.7680`, and total communication `16.70 GB`. Other reports contain 100-round baseline figures, so numbers must always be tied to their run artifact and round count.

### Phase 3: Compression

**Purpose:** reduce the number of bytes sent by clinics.

It adds Top-K sparsification with error feedback, QSGD-style stochastic quantization, warmup support, BatchNorm preservation, and exact wire-byte calculations. Compression is designed to be a client-side option, so it can later be combined with async aggregation and connectivity simulation.

**Why it matters:** on rural links, reducing the update size can be the difference between an update arriving and being discarded as a timeout.

**Reported Phase 3 benchmark:** the phase report describes Top-5% DGC-style compression with approximately 7.01x uplink compression, 85.7% uplink savings, final macro-AUROC increasing from 90.52% to 91.72%, final accuracy changing from 81.67% to 80.72%, identical final macro-F1 at 83.76%, and zero NaNs in that run. This is a specific benchmark, not proof that every compression ratio behaves identically.

### Phase 4: Buffered async aggregation and dropout

**Purpose:** prevent slow or absent clinics from stopping progress.

It adds FedBuff-style buffering, staleness-weighted aggregation, a pluggable dropout model, a synchronous-with-dropout comparator, and buffer-size/dropout sweeps.

**Why it matters:** standard synchronous FL is governed by the slowest required participant. The phase tests whether buffering can improve resilience and time-to-progress under the same client failures.

**Reported evidence:** generated figures include dropout retention, FedBuff-versus-sync comparison, buffer-size comparison, and COVID-19 sensitivity versus dropout. These figures are supporting results; they do not automatically constitute every planned controlled ablation.

### Phase 5: Intermittent-connectivity engine

**Purpose:** replace abstract dropout probabilities with bandwidth- and state-driven outcomes.

It assigns each clinic a Markov connectivity profile, samples bandwidth, computes transmission time from actual payload size, applies a bandwidth cap, and exposes the result through the Phase 4 dropout interface.

**Why it matters:** it connects communication engineering to model behavior. Compression is no longer only a byte-count exercise; it changes whether remote clinics can participate.

**Strong reported interaction result at 256 kbps:**

| Configuration | Macro-AUROC | Macro-F1 | COVID sensitivity | Effective uptime |
|---|---:|---:|---:|---:|
| Top-K 1%, 11.53 MB | 0.9256 | 0.8432 | 99.00% | 55.6% |
| Top-K 10%, 23.07 MB | 0.9241 | 0.8433 | 99.16% | 38.4% |
| Uncompressed, 86.00 MB | 0.9082 | 0.7842 | 76.08% | 30.4% |

Interpretation: under this simulated condition, the uncompressed update often misses the five-minute budget. Fewer clinics contribute, uptime falls, and COVID-19 sensitivity drops sharply. The result supports the project's main systems argument, but it remains a simulation result based on modeled connectivity.

### Phase 6: Joint experiment orchestration

**Purpose:** combine compression, aggregation, bandwidth, dropout, and seeds into reproducible runs.

`experiment_runner.py` expands a YAML grid into subprocess commands, runs the Phase 5 entry point, captures logs and raw JSON, and writes a manifest. `joint_metric_plots.py` groups runs, computes means and standard deviations, creates heatmaps, and selects a recommended operating point. `supabase_writer.py` converts raw results into normalized experiment, run, round, client, and client-event rows.

**Why it matters:** isolated phase results are not enough. The project's central metric is performance under a joint bandwidth/dropout condition.

**Planned sweep:** the YAML config describes 4 compression settings x 4 connectivity conditions x 3 seeds, plus a three-seed synchronous comparison: 51 planned runs in total.

**Actual checked-in evidence:** `results/phase6/raw/` contains eight result JSON files, all with seed 42. Therefore the checked-in summary is a representative sweep, not the complete three-seed statistical study described by the config. The summary's standard deviations are zero because there is one seed per populated point.

### Phase 7: Monitoring dashboard

**Purpose:** make stored results inspectable by humans.

The Express backend reads Supabase using environment variables and provides endpoints for experiments, runs, round metrics, client events, the joint metric, and client details. The Next.js frontend provides:

- an overview of experiments and completed runs;
- experiment/run tables;
- convergence charts;
- a bandwidth-by-dropout macro-AUROC heatmap;
- a clinic explorer and client detail view.

The dashboard is a read-only visualization layer. It does not execute training and should not invent results. If a number is missing, the correct place to fix it is the Phase 6 export/schema pipeline.

### Phase 8: Integration, ablations, and final packaging

**Purpose:** verify the complete pipeline, isolate component contributions, and package the report/demo.

The planned ablations are:

1. IID versus non-IID partitions;
2. error feedback on versus off;
3. staleness weighting on versus off;
4. probability dropout versus connectivity-trace dropout;
5. compression alone versus compression plus async versus the full system.

**Current evidence status:** the Phase 8 evidence matrix correctly says that the checked-in repository does not contain result files for the IID, error-feedback-off, or staleness-weighting-off counterfactuals. The simple-dropout versus connectivity comparison is supporting evidence rather than a perfectly matched controlled pair. These should be described as limitations, not overclaimed as causal ablations.

---

## 13. Results currently in the repository

### Phase 6 summary table

The checked-in `results/phase6/summary.csv` contains these eight populated points:

| Strategy | Compression | Bandwidth | Dropout | Accuracy | Macro-AUROC | Macro-F1 | Seeds |
|---|---|---:|---:|---:|---:|---:|---:|
| FedBuff | none | 2048 kbps | 0% | 0.7667 | 0.9051 | 0.7680 | 1 |
| FedBuff | none | 256 kbps | 50% | 0.7699 | 0.9082 | 0.7842 | 1 |
| FedBuff | Top-K 10% | 256 kbps | 50% | 0.8181 | 0.9241 | 0.8433 | 1 |
| FedBuff | Top-K 1% | 2048 kbps | 0% | 0.7873 | 0.9025 | 0.8168 | 1 |
| FedBuff | Top-K 1% | 512 kbps | 25% | 0.7981 | 0.9131 | 0.8126 | 1 |
| FedBuff | Top-K 1% | 256 kbps | 50% | 0.8167 | 0.9256 | 0.8432 | 1 |
| FedBuff | Top-K 1% | 64 kbps | 75% | 0.7828 | 0.9254 | 0.8216 | 1 |
| Sync + dropout | Top-K 1% | 512 kbps | 25% | 0.8080 | 0.9304 | 0.8402 | 1 |

These rows are not a monotonic curve. That is normal for stochastic training and changing client participation, but it also means a single point should not be interpreted as a universal law.

### Current recommended operating point

The repository selects:

```text
Strategy:       FedBuff
Compression:    Top-K 1%
Bandwidth cap:  256 kbps
Dropout:        50%
Accuracy:       0.8167
Macro-AUROC:    0.9256
Macro-F1:       0.8432
Seeds:          1
```

The selector prioritizes the highest macro-AUROC among compressed FedBuff points. It is a practical operating point in the current artifact, not a clinically validated deployment recommendation.

### Async versus sync comparison

At the matched checked-in Phase 6 point of Top-K 1%, 512 kbps, and 25% dropout:

| Strategy | Accuracy | Macro-AUROC | Macro-F1 |
|---|---:|---:|---:|
| FedBuff | 0.7981 | 0.9131 | 0.8126 |
| Sync + dropout | 0.8080 | 0.9304 | 0.8402 |

The measured macro-AUROC difference is `-0.0173` for FedBuff relative to sync in this one-seed comparison. That does **not** show FedBuff always loses; it shows that the current matched point does not support a blanket claim that async is better. Phase 4 contains additional generated comparison figures under different configurations.

### What the results mean overall

The most defensible project-level interpretation is:

- Top-K compression substantially reduces payload size.
- Smaller payloads can keep more clinics within a bandwidth/time budget.
- Modeled connectivity can turn payload size into participation differences.
- FedBuff provides a configurable way to buffer surviving updates and discount stale ones.
- The current evidence supports the value of combining compression with connectivity-aware evaluation.
- The repository does not yet support strong causal claims for every planned ablation or a three-seed statistical conclusion at the Phase 6 points.

---

## 14. Database and dashboard data flow

The Phase 6 result JSON is the research-side source. The export layer normalizes it into:

- `experiments`: named sweeps;
- `runs`: one configuration/seed execution;
- `rounds`: convergence and communication metrics per round;
- `clients`: clinic metadata;
- `client_round_events`: participation, dropout, staleness, bandwidth, and uptime events.

The batch file `results/phase6/supabase_batch.json` is the portable ingestion artifact. Supabase receives rows through the REST API using environment-provided credentials. The Express service queries Supabase and returns UI-shaped JSON. The Next.js pages fetch those endpoints.

This separation is intentional:

```text
Python research code -> JSON/batch rows -> Supabase -> Express -> Next.js
```

The browser should never need a Supabase service-role key.

---

## 15. Testing and quality controls

The repository includes tests under `research/tests/` covering:

- label harmonization;
- preprocessing and image loading;
- duplicate and patient-group isolation;
- deterministic partitioning;
- data-root portability;
- archive safety and download-resume behavior;
- offline dev-sample behavior;
- Top-K selection and residual accumulation;
- QSGD stochastic behavior;
- flatten/unflatten correctness;
- exact communication byte formulas;
- DenseNet compression boundaries and BatchNorm preservation;
- Phase 6 sweep expansion, summaries, recommendations, and Supabase row contracts.

The project also uses configuration hashes, acquisition receipts, manifest files, MLflow artifacts, raw JSON results, and generated reports. These are provenance mechanisms: they make it easier to answer which data, configuration, seed, and code path produced a number.

---

## 16. Important limitations and caveats

1. **Simulation, not deployment:** clinics, network conditions, and outages are simulated. No real hospital network was measured.
2. **Simplified label taxonomy:** three harmonized classes are useful for a capstone experiment but do not replace clinical annotation or validation.
3. **CheXpert access/data availability:** the repository explicitly handles pending approved files; full data preparation depends on the external data environment.
4. **Single-seed checked-in Phase 6 results:** eight raw result files are present, all seed 42, despite the larger planned sweep.
5. **Missing controlled ablations:** no checked-in IID counterfactual, error-feedback-off result, or staleness-weighting-off result is available.
6. **Virtual asynchrony:** FedBuff is implemented inside Flower simulation rather than as a real distributed asynchronous network service.
7. **Payload/config distinctions:** some reports use approximately 86 MB uncompressed payloads while the checked-in model-byte trace reports approximately 27.83 MB. Always cite the exact run and configuration when discussing bytes.
8. **Synthetic infrastructure correlation:** the assumption that larger clinics have better connectivity may make the modeled system easier or harder in particular ways. It should be stated, not hidden.
9. **No privacy/security mechanism:** federated learning keeps raw images local in the simulation, but secure aggregation, differential privacy, and adversarial protections are not the focus of this implementation.
10. **Dashboard status:** the dashboard is a read-only result browser. Phase 8 records that a complete live API-to-dashboard trace still requires manual confirmation.

These limitations do not make the project useless. They define what its results can legitimately support: an engineered, reproducible simulation study of communication-aware federated learning, not a clinical deployment claim.

---

## 17. How to read the repository

For a beginner, this is the most useful reading order:

1. Read this file and the root `README.md`.
2. Read `Phase_Docs/PHASE 1.md` through `PHASE 8.md` to see the intended research plan.
3. Read `research/src/data_prep/` and `research/src/partitioning/partition.py` to understand the data contract.
4. Read `research/src/fl_core/models/densenet.py`, `client.py`, and `server.py` to understand one FL round.
5. Read `research/src/fl_core/compression/` for update compression.
6. Read `research/src/fl_core/aggregation/fedbuff_strategy.py` and `sync_with_dropout.py` for resilience behavior.
7. Read `research/src/simulation/` for network behavior.
8. Read `research/src/eval/experiment_runner.py`, `joint_metric_plots.py`, and `supabase_writer.py` for orchestration and reporting.
9. Inspect `results/phase6/` and `results/phase8/` to distinguish measured evidence from planned work.
10. Read `web/backend/server.js` and the frontend `app/` routes to understand the dashboard.
11. Use `research/tests/` as executable examples of the data and compression contracts.

---

## 18. Final takeaway

This is not merely an image classifier and not merely a dashboard. It is a staged systems-and-ML experiment:

```text
medical-image classification
    + federated data ownership
    + non-IID rural-clinic simulation
    + update compression
    + dropout-tolerant aggregation
    + bandwidth-aware connectivity
    + reproducible experiment orchestration
    + result storage and visualization
```

The project asks whether a global diagnostic model can remain useful when its clients do not look alike and cannot communicate reliably. Its strongest current evidence is the Phase 5/6 interaction showing that Top-K compression can preserve diagnostic metrics and substantially improve simulated clinic participation under a 256 kbps constraint. Its strongest scientific responsibility is to state that this evidence is simulated and currently supported by one seed per checked-in Phase 6 point, with several planned ablations still awaiting controlled result files.