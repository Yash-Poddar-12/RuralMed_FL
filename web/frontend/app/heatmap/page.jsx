"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

export default function HeatmapPage() {
  const [experiments, setExperiments] = useState([]);
  const [experiment, setExperiment] = useState("");
  const [points, setPoints] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${apiUrl}/experiments`).then((response) => response.json()).then((payload) => {
      setExperiments(payload.data || []);
      if (payload.data?.[0]) setExperiment(payload.data[0].name);
    }).catch((reason) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (!experiment) return;
    fetch(`${apiUrl}/experiments/${encodeURIComponent(experiment)}/joint-metric`).then((response) => response.json()).then((payload) => {
      if (payload.error) throw new Error(payload.error);
      setPoints(payload.data || []);
    }).catch((reason) => setError(reason.message));
  }, [experiment]);

  const bandwidths = [...new Set(points.map((point) => point.bandwidthKbps == null ? "uncapped" : point.bandwidthKbps))];
  const dropouts = [...new Set(points.map((point) => point.dropoutCondition))];
  const value = (dropout, bandwidth) => points.find((point) => point.dropoutCondition === dropout && (point.bandwidthKbps == null ? "uncapped" : point.bandwidthKbps) === bandwidth)?.macroAurocMean;

  return (
    <main className="shell">
      <Link href="/" className="back-link">← Overview</Link>
      <header className="page-heading split-heading"><div><p className="eyebrow">Joint metric</p><h1>Accuracy under constraint</h1><p className="lede">Macro-AUROC retained across bandwidth and dropout conditions.</p></div><select value={experiment} onChange={(event) => setExperiment(event.target.value)}>{experiments.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}</select></header>
      {error && <div className="alert">{error}</div>}
      <section className="panel">
        <div className="heatmap">
          <div className="heatmap-corner">Dropout<br />Bandwidth →</div>
          {bandwidths.map((bandwidth) => <div className="heatmap-header" key={bandwidth}>{bandwidth === "uncapped" ? "Uncapped" : `${bandwidth}k`}</div>)}
          {dropouts.map((dropout) => <div className="heatmap-row" key={dropout}><div className="heatmap-label">{dropout}</div>{bandwidths.map((bandwidth) => { const score = value(dropout, bandwidth); return <div className="heat-cell" title={`${dropout}, ${bandwidth}: ${score == null ? "no data" : score.toFixed(4)}`} style={{ background: score == null ? "#e7e9e4" : `rgba(22, 110, 91, ${Math.max(0.12, Math.min(1, (score - 0.5) * 2))})` }} key={bandwidth}>{score == null ? "—" : score.toFixed(3)}</div>; })}</div>)}
        </div>
        <p className="muted heatmap-note">Hover a cell for the exact condition. Values are means across the available seeds.</p>
      </section>
    </main>
  );
}
