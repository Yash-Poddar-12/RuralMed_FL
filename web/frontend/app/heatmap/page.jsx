"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

const conditionLabels = {
  no_dropout: "No dropout",
  moderate_dropout: "Moderate",
  severe_dropout: "Severe",
  extreme_dropout: "Extreme",
};

function formatCondition(value) {
  return conditionLabels[value] || value.replaceAll("_", " ");
}

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

  const bandwidths = [...new Set(points.map((point) => point.bandwidthKbps == null ? "uncapped" : point.bandwidthKbps))]
    .sort((left, right) => left === "uncapped" ? 1 : right === "uncapped" ? -1 : left - right);
  const dropouts = [...new Set(points.map((point) => point.dropoutCondition))]
    .sort((left, right) => {
      const leftPoint = points.find((point) => point.dropoutCondition === left);
      const rightPoint = points.find((point) => point.dropoutCondition === right);
      return (leftPoint?.bandwidthKbps ?? 0) - (rightPoint?.bandwidthKbps ?? 0);
    });
  const value = (dropout, bandwidth) => points.find((point) => point.dropoutCondition === dropout && (point.bandwidthKbps == null ? "uncapped" : point.bandwidthKbps) === bandwidth)?.macroAurocMean;

  return (
    <main className="shell">
      <Link href="/" className="back-link">← Overview</Link>
      <header className="page-heading split-heading"><div><p className="eyebrow">Joint metric</p><h1>Accuracy under constraint</h1><p className="lede">Macro-AUROC across bandwidth caps and dropout conditions. Each cell is the mean of the available runs.</p></div><label className="select-label">Experiment<select value={experiment} onChange={(event) => setExperiment(event.target.value)}>{experiments.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}</select></label></header>
      {error && <div className="alert">{error}</div>}
      <section className="panel">
        <div className="panel-heading heatmap-heading"><div><h3>Macro-AUROC by network condition</h3><span className="muted">Higher values indicate stronger diagnostic discrimination.</span></div><span className="heatmap-scale"><i /> Lower <b /> Higher</span></div>
        <div className="heatmap-wrap">
          <div className="heatmap-axis-title">Dropout</div>
          <div className="heatmap">
            <div className="heatmap-corner">Condition</div>
            {bandwidths.map((bandwidth) => <div className="heatmap-header" key={bandwidth}>{bandwidth === "uncapped" ? "Uncapped" : `${bandwidth} kbps`}</div>)}
            {dropouts.map((dropout) => <div className="heatmap-row" key={dropout}><div className="heatmap-label"><strong>{formatCondition(dropout)}</strong><small>{dropout.replace("_dropout", "")} dropout</small></div>{bandwidths.map((bandwidth) => { const score = value(dropout, bandwidth); const intensity = score == null ? 0 : Math.max(0.12, Math.min(1, (score - 0.5) * 2)); return <div className="heat-cell" title={`${formatCondition(dropout)}, ${bandwidth}: ${score == null ? "no data" : score.toFixed(4)}`} style={{ background: score == null ? "#eef2f0" : `rgba(20, 125, 115, ${intensity})`, color: score == null || intensity < 0.55 ? "#234e52" : "#ffffff" }} key={bandwidth}>{score == null ? "No data" : score.toFixed(3)}</div>; })}</div>)}
          </div>
        </div>
        <p className="muted heatmap-note">Hover a cell for the exact value. Blank cells mean that configuration has not been run.</p>
      </section>
    </main>
  );
}
