-- Phase 6 normalized result store.
-- The external IDs keep batch ingestion deterministic before Supabase UUIDs
-- are needed by the Phase 7 API.
create extension if not exists pgcrypto;

create table if not exists experiments (
  id uuid primary key default gen_random_uuid(),
  name text not null unique,
  description text not null default '',
  config jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists runs (
  id uuid primary key default gen_random_uuid(),
  external_id text not null unique,
  experiment_id uuid references experiments(id) on delete cascade,
  experiment_external_name text not null,
  compression_setting text not null,
  aggregation_strategy text not null,
  dropout_condition text not null,
  bandwidth_kbps double precision,
  seed integer not null,
  status text not null check (status in ('planned', 'running', 'completed', 'failed')),
  parameters jsonb not null default '{}'::jsonb,
  final_metrics jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists rounds (
  id uuid primary key default gen_random_uuid(),
  run_id uuid references runs(id) on delete cascade,
  run_external_id text not null,
  round_number integer not null check (round_number >= 0),
  recorded_at timestamptz not null default now(),
  global_accuracy double precision,
  global_auroc double precision,
  global_f1 double precision,
  communication_bytes bigint,
  unique (run_external_id, round_number)
);

create table if not exists clients (
  id uuid primary key default gen_random_uuid(),
  external_id text not null unique,
  experiment_id uuid references experiments(id) on delete cascade,
  experiment_external_name text not null,
  client_index integer not null,
  data_volume integer,
  label_distribution_summary jsonb not null default '{}'::jsonb,
  connectivity_profile_id text,
  unique (experiment_external_name, client_index)
);

create table if not exists client_round_events (
  id uuid primary key default gen_random_uuid(),
  run_id uuid references runs(id) on delete cascade,
  run_external_id text not null,
  client_id uuid references clients(id) on delete cascade,
  client_external_id text not null,
  round_number integer,
  status text not null,
  staleness double precision,
  effective_bandwidth double precision,
  uptime double precision
);

create index if not exists rounds_run_round_idx
  on rounds (run_external_id, round_number);
create index if not exists client_events_run_idx
  on client_round_events (run_external_id, client_external_id);

alter table experiments enable row level security;
alter table runs enable row level security;
alter table rounds enable row level security;
alter table clients enable row level security;
alter table client_round_events enable row level security;

create policy "public experiment reads" on experiments for select using (true);
create policy "public run reads" on runs for select using (true);
create policy "public round reads" on rounds for select using (true);
create policy "public client reads" on clients for select using (true);
create policy "public event reads" on client_round_events for select using (true);
