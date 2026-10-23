# Faculty Review Brief

## Project claim

This project evaluates communication-efficient federated learning for
three-class chest-radiograph classification across ten heterogeneous rural
clinic simulations. It combines Top-K update compression, buffered
asynchronous aggregation, and intermittent bandwidth/dropout conditions.

## What was built

1. Phase 1 creates centralized evaluation data and heterogeneous client
   partitions with class and source skew.
2. Phase 2 establishes the uncompressed FedAvg baseline.
3. Phase 3 measures sparse-update communication savings using residual
   error-feedback compression.
4. Phase 4 introduces FedBuff, dropout handling, staleness, and buffer-size
   comparisons.
5. Phase 5 adds a Markov connectivity model with bandwidth caps and client
   uptime diagnostics.
6. Phase 6 combines compression and connectivity into a joint sweep and
   produces heatmaps and an operating-point recommendation.
7. Phase 7 exposes the stored results through the Next.js and Express
   dashboard.
8. Phase 8 packages the evidence, integration trace, limitations, and demo.

## Main measured result

At the current recommended point, FedBuff with Top-K 1% compression at a
256 kbps cap and 50% dropout reports:

- Test accuracy: `0.8167`
- Macro-AUROC: `0.9256`
- Macro-F1: `0.8432`
- COVID-19 sensitivity: `0.9900`

The Phase 5 matched interaction report gives the strongest compression
comparison at 256 kbps:

| Configuration | Macro-AUROC | Macro-F1 | COVID sensitivity |
|---|---:|---:|---:|
| Top-K 1% | 0.9256 | 0.8432 | 99.0% |
| Top-K 10% | 0.9241 | 0.8433 | 99.2% |
| Uncompressed | 0.9082 | 0.7842 | 76.1% |

## What to say about ablations

The attached repository contains supporting comparisons, not all five
controlled Phase 8 ablations. Specifically, there are no attached IID
counterfactual, error-feedback-off, or staleness-weighting-off result files.
The presentation should state this as a limitation rather than label the
existing Phase 3-6 comparisons as controlled ablations.

## Recommended demo path

1. Show the Phase 1 client partition and class/source skew.
2. Show the Phase 3 compression trade-off figure.
3. Show the Phase 4 FedBuff versus synchronous and buffer-size figures.
4. Show the Phase 5 bandwidth/compression interaction and client uptime.
5. Show the Phase 6 retained-accuracy and macro-AUROC heatmaps.
6. Open the Phase 7 dashboard and drill into the same Phase 6 operating point.
7. Close with the Phase 8 evidence matrix and limitations.

## Limitations to disclose

- The current Phase 6 attached sweep has one seed per condition.
- Connectivity is simulated rather than measured from deployed clinics.
- The task uses a three-class harmonized label taxonomy and proxy pneumonia
  labels from public datasets.
- The dashboard is a read-only visualization layer; it does not execute FL.
- The available evidence does not support causal claims for every planned
  ablation.