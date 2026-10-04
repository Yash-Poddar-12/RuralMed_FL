# Communication-Efficient Federated Learning for Diagnostic Imaging

University capstone project investigating diagnostic-imaging federated learning under intermittent rural connectivity.

## Team

- Yash Poddar (23BDS0195)
- Pavan D Umesh (23BDS0046)
- Aayush Sood (23BDS0177)

## Repository layout

- `research/`: Python data preparation, partitioning, and later ML experiments.
- `web/`: Phase 7 dashboard frontend and backend placeholders.
- `infra/`: shared decisions, contracts, and Supabase planning.
- `Phase_Docs/`: project phase specifications supplied by the team.

Phase 1 is limited to repository, environment, and data foundations. No federated-learning implementation belongs here yet.

Start with [Phase 1 status and GPU handoff](infra/docs/phase1-handoff.md), [research commands](research/README.md), and the [decisions log](infra/docs/decisions-log.md). The full COVID dataset plus an explicitly marked NIH pilot are processed and partitioned; remaining acquisition and pinned-environment checks continue in a local background worker. Raw/processed data and credentials remain ignored by Git.
