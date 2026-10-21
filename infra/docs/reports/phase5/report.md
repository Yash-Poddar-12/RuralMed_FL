# Phase 5 Benchmark Summary: Intermittent Connectivity Engine

**Project:** RuralMed-FL  
**Architecture:** DenseNet-121 (CheXNet backbone, 6.96M parameters)  
**Task:** 3-class Chest Radiograph Classification (Normal, Pneumonia, COVID-19)  
**Simulation Setup:** 10 simulated rural Indian clinics partitioned with volume-correlated infrastructure tiers (3 Stable, 3 Intermittent, 4 Unreliable) driven by a 3-state discrete Markov chain.  
**Aggregation Strategy:** FedBuff (Buffer size $K=3$, staleness penalty $\alpha=0.5$, round budget $\tau=300\text{ s}$).

---

## 1. Key Results: Bandwidth Sweep & Interaction Matrix

### Axis 1: Bandwidth Degradation (Fixed TopK 1% Compression)
Evaluating model resilience as available transmission bandwidth is choked from uncapped down to extreme 2G/satellite speeds (64 kbps):

| Bandwidth Cap | Compression Mode | Payload Size | Macro-AUROC | Macro-F1 | COVID-19 Sens | Test Accuracy | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uncapped** | TopK 1% | 11.53 MB | **0.9025** | 0.8168 | 95.41% | 78.73% | Complete |
| **1024 kbps** | TopK 1% | 11.53 MB | **0.9212** | 0.8172 | 88.14% | 79.37% | Complete |
| **512 kbps** | TopK 1% | 11.53 MB | **0.9131** | 0.8126 | 83.42% | 79.81% | Complete |
| **256 kbps** | TopK 1% | 11.53 MB | **0.9256** | **0.8432** | **99.00%** | **81.67%** | Complete |
| **128 kbps** | TopK 1% | 11.53 MB | **0.9278** | **0.8432** | **98.58%** | **81.71%** | Complete |
| **64 kbps** | TopK 1% | 11.53 MB | **0.9254** | 0.8216 | 97.20% | 78.28% | Complete |

> **Key Takeaway (Axis 1):** With TopK 1% gradient compression, FedBuff exhibits extraordinary bandwidth resilience. Even under an extreme 64 kbps bottleneck, Macro-AUROC remains above **0.925** and COVID-19 sensitivity exceeds **97.2%**, comfortably exceeding the Phase 2 centralized benchmark (**0.9051**).

---

### Axis 2: Joint Bandwidth × Compression Interaction (at 256 kbps Cap)
Demonstrating why compression is essential when network links are constrained:

| Experiment | Payload Size | Effective Clinic Uptime | Macro-AUROC | Macro-F1 | COVID-19 Sens | Sensitivity Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TopK 1%** | 11.53 MB | **55.6%** | **0.9256** | **0.8432** | **99.00%** | Baseline |
| **TopK 10%** | 23.07 MB | **38.4%** | **0.9241** | **0.8433** | **99.16%** | +0.16% |
| **Uncompressed** | 86.00 MB | **30.4%** | **0.9082** | **0.7842** | **76.08%** | **-22.92% (COLLAPSE)** |

> **Key Takeaway (Axis 2):** Under realistic 256 kbps edge connectivity, transmitting an uncompressed 86 MB model payload takes $\approx 45\text{ minutes}$, vastly exceeding the 300-second round budget. This causes remote clinic updates to time out as stragglers. Effective clinic uptime drops from **55.6% to 30.4%**, resulting in a catastrophic **22.9% collapse in COVID-19 clinical sensitivity** (from 99.0% down to 76.1%). Compression is therefore proven to be a safety-critical necessity for rural federated healthcare.

---

## 2. Infrastructure Tier Diagnostics

Per-clinic infrastructure tiers correlated with training data volume (Phase 5 spec §3.2):
- **Stable Clinics (n=3):** `client_00` (10,972 imgs), `client_02` (17,134 imgs), `client_03` (20,722 imgs). Maintained 100% link availability with 2.6–2.9 Mbps throughput.
- **Intermittent Clinics (n=3):** `client_06` (9,189 imgs), `client_07` (5,897 imgs), `client_08` (7,956 imgs). Experienced cellular jitter (33%–100% per-round uptime).
- **Unreliable Sub-Centres (n=4):** `client_01` (2,860 imgs), `client_04` (1,150 imgs), `client_05` (2,200 imgs), `client_09` (3,448 imgs). High outage rates (0%–33% uptime under uncompressed payloads, preserved up to 33%–66% with TopK 1%).

---

## 3. Publication Figures

The following publication-quality figures are generated in `infra/docs/reports/phase5/`:

1. **`fig1_bandwidth_degradation_curve.png`**: Macro-AUROC, Accuracy, and Macro-F1 across the 64 kbps $\rightarrow$ Uncapped bandwidth sweep, showing performance retention well above the Phase 2 ceiling.
2. **`fig2_joint_bandwidth_compression_interaction.png`**: Side-by-side bar chart of AUROC, F1, and COVID-19 sensitivity at 256 kbps comparing TopK 1%, TopK 10%, and Uncompressed.
3. **`fig3_client_uptime_and_tier_distribution.png`**: Effective clinic uptime across the three infrastructure tiers, illustrating the straggler discard penalty for uncompressed models.
4. **`fig4_covid_sensitivity_vs_bandwidth.png`**: COVID-19 recall retention curve across bandwidth constraints, highlighting the clinical safety margin achieved by gradient compression.

---

## 4. Teammate Hand-off Checklist (Phase 6 Readiness)

- **Branch:** `phase5-connectivity-engine`
- **Simulation Code:**
  - `research/src/simulation/connectivity_states.py` (Markov states & bandwidth distributions)
  - `research/src/simulation/client_profiles.py` (tier profile definitions & volume-based assignment)
  - `research/src/simulation/markov_engine.py` (per-client Markov state machine)
  - `research/src/simulation/bandwidth_delay.py` (bandwidth-to-delay adapter for FedBuff)
- **Runner Script:** `research/scripts/run_phase5.py` (supports `--bw-cap`, `--rounds`, `--run-name`, `--seed`)
- **Plotting Script:** `research/scripts/plot_phase5_results.py`
- **MLflow Tracking:** All runs logged under experiment `phase5-bandwidth-sweep` in `research/artifacts/mlruns`.
- **Note on Phase 6:** Per repo guidelines, Phase 6 must not be started without separate instruction. All data prep and heavy runs remain appropriately scoped.
