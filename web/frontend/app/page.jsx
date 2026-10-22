"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

function Metric({ label, value, suffix = "" }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value == null ? "—" : `${value}${suffix}`}</strong>
    </div>
  );
}

export default function HomePage() {
  const [experiments, setExperiments] = useState([]);
  const [clients, setClients] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([
      fetch(`${apiUrl}/experiments`).then((response) => response.json()),
      fetch(`${apiUrl}/clients`).then((response) => response.json()),
    ])
      .then(([experimentResponse, clientResponse]) => {
        if (experimentResponse.error) throw new Error(experimentResponse.error);
        setExperiments(experimentResponse.data || []);
        setClients(clientResponse.data || []);
      })
      .catch((reason) => setError(reason.message));
  }, []);

  const best = experiments
    .map((experiment) => experiment.bestMacroAuroc)
    .filter((value) => value != null);

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">RURALMED / PHASE 07</p>
          <h1>Federated learning observatory</h1>
        </div>
        <span className="status-pill"><i /> Supabase connected</span>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">Communication-efficient diagnostic imaging</p>
          <h2>See where the network changes the model.</h2>
          <p className="lede">
            Explore Phase 6 sweeps across compression, aggregation, bandwidth,
            and intermittent rural connectivity.
          </p>
        </div>
        <Link className="primary-button" href="/heatmap">Open joint metric <span>↗</span></Link>
      </section>

      {error && <div className="alert">{error}. Start the API and verify Supabase credentials.</div>}

      <section className="metrics-grid">
        <Metric label="Experiments" value={experiments.length} />
        <Metric label="Completed runs" value={experiments.reduce((sum, item) => sum + (item.completedCount || 0), 0)} />
        <Metric label="Best macro-AUROC" value={best.length ? Math.max(...best).toFixed(4) : null} />
        <Metric label="Simulated clinics" value={clients.length || null} />
      </section>

      <section className="section-heading">
        <div>
          <p className="eyebrow">Experiment registry</p>
          <h3>Phase 6 sweeps</h3>
        </div>
        <Link href="/clients" className="text-link">Explore clinics →</Link>
      </section>

      <section className="experiment-list">
        {experiments.map((experiment) => (
          <Link className="experiment-card" href={`/experiments/${encodeURIComponent(experiment.name)}`} key={experiment.id || experiment.name}>
            <div className="card-topline">
              <span className="tag">SWEEP</span>
              <span>{new Date(experiment.created_at).toLocaleDateString()}</span>
            </div>
            <h4>{experiment.name}</h4>
            <p>{experiment.description || "Joint compression and connectivity evaluation."}</p>
            <div className="card-footer">
              <span>{experiment.completedCount}/{experiment.runCount} runs complete</span>
              <strong>{experiment.bestMacroAuroc == null ? "—" : experiment.bestMacroAuroc.toFixed(4)} AUROC</strong>
            </div>
          </Link>
        ))}
        {!experiments.length && !error && <div className="empty">No experiments have been ingested yet.</div>}
      </section>
    </main>
  );
}
