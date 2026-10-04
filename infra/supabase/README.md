# Supabase project shell

Phase 1 keeps a schema plan; actual migrations and dashboard integration belong to Phase 7.
Hosted project creation needs the team's Supabase account and organization. No project or real credentials are claimed to exist yet.

Copy `.env.example` to `.env` and fill the project URL and keys after creation. Keep `.env` ignored; the service role key is server-only and must never enter frontend code or Git.

| Planned table | Planned identity and relationships | Purpose |
|---|---|---|
| experiments | UUID id, config_hash, config JSONB, status, created_at | Immutable configuration and experiment provenance |
| clients | UUID id, experiment_id FK, client_index, dataset_summary JSONB | Clinic identity and partition metadata |
| training_rounds | UUID id, experiment_id FK, round_number, started_at, finished_at | Round lifecycle |
| metrics | UUID id, experiment_id FK, round_id nullable FK, client_id nullable FK, name, value, step, recorded_at | Global and client observations |

Later migrations should enforce `(experiment_id, client_index)` and `(experiment_id, round_number)` uniqueness. Configure read access and authenticated backend writes with row-level security once the dashboard's actual access model exists. Store dataset statistics, never medical images, in Supabase.
