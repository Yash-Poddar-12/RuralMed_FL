"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";

const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

function LineChart({ rows }) {
  if (!rows.length) return <div className="empty">No round metrics available.</div>;
  const width = 760;
  const height = 260;
  const points = (key) => rows.map((row, index) => {
    const x = 32 + (index / Math.max(rows.length - 1, 1)) * (width - 56);
    const y = height - 28 - Number(row[key] || 0) * (height - 58);
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Convergence metrics">
      <line x1="32" y1="20" x2="32" y2="232" className="axis" />
      <line x1="32" y1="232" x2="744" y2="232" className="axis" />
      {[0, 0.5, 1].map((value) => <line key={value} x1="32" y1={204 - value * 184} x2="744" y2={204 - value * 184} className="gridline" />)}
      <polyline points={points("global_accuracy")} className="line blue" />
      <polyline points={points("global_auroc")} className="line green" />
      <polyline points={points("global_f1")} className="line orange" />
      <text x="42" y="18" className="chart-label">score</text>
      <text x="700" y="252" className="chart-label">round</text>
    </svg>
  );
}

export default function RunPage() {
  const { id } = useParams();
  const run = decodeURIComponent(id);
  const [rounds, setRounds] = useState([]);
  const [clients, setClients] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([
      fetch(`${apiUrl}/runs/${encodeURIComponent(run)}/rounds`).then((response) => response.json()),
      fetch(`${apiUrl}/runs/${encodeURIComponent(run)}/clients`).then((response) => response.json()),
    ]).then(([roundResponse, clientResponse]) => {
      if (roundResponse.error) throw new Error(roundResponse.error);
      setRounds(roundResponse.data || []);
      setClients(clientResponse.data || []);
    }).catch((reason) => setError(reason.message));
  }, [run]);

  const participation = clients.filter((item) => item.status === "participated").length;
  const dropped = clients.filter((item) => item.status === "dropped").length;

  return (
    <main className="shell">
      <Link href="/" className="back-link">← Overview</Link>
      <header className="page-heading">
        <p className="eyebrow">Run detail</p>
        <h1>{run}</h1>
        <p className="lede">Round-by-round convergence and clinic connectivity events.</p>
      </header>
      {error && <div className="alert">{error}</div>}
      <section className="panel">
        <div className="panel-heading"><h3>Global convergence</h3><span className="legend"><i className="blue-dot" /> Accuracy <i className="green-dot" /> AUROC <i className="orange-dot" /> F1</span></div>
        <LineChart rows={rounds} />
      </section>
      <section className="metrics-grid compact">
        <div className="metric"><span>Rounds recorded</span><strong>{rounds.length}</strong></div>
        <div className="metric"><span>Participated events</span><strong>{participation}</strong></div>
        <div className="metric"><span>Dropped events</span><strong>{dropped}</strong></div>
        <div className="metric"><span>Communication</span><strong>{rounds.reduce((sum, row) => sum + Number(row.communication_bytes || 0), 0) / 1e9 < 0.01 ? `${(rounds.reduce((sum, row) => sum + Number(row.communication_bytes || 0), 0) / 1e6).toFixed(1)} MB` : `${(rounds.reduce((sum, row) => sum + Number(row.communication_bytes || 0), 0) / 1e9).toFixed(2)} GB`}</strong></div>
      </section>
      <section className="panel">
        <div className="panel-heading"><h3>Client participation timeline</h3><span className="muted">Each row is a recorded event</span></div>
        <div className="event-list">{clients.slice(0, 100).map((event, index) => <div className="event-row" key={`${event.client_external_id}-${event.round_number}-${index}`}><span>{event.client_external_id}</span><span>Round {event.round_number ?? "summary"}</span><b className={event.status === "dropped" ? "bad" : "good"}>{event.status}</b><span>{event.effective_bandwidth == null ? "—" : `${Number(event.effective_bandwidth).toFixed(0)} kbps`}</span></div>)}</div>
      </section>
    </main>
  );
}
