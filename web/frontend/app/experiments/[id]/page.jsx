"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";

const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

export default function ExperimentPage() {
  const { id } = useParams();
  const experiment = decodeURIComponent(id);
  const [runs, setRuns] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${apiUrl}/experiments/${encodeURIComponent(experiment)}/runs`)
      .then((response) => response.json())
      .then((payload) => {
        if (payload.error) throw new Error(payload.error);
        setRuns(payload.data || []);
      })
      .catch((reason) => setError(reason.message));
  }, [experiment]);

  return (
    <main className="shell">
      <Link href="/" className="back-link">← Overview</Link>
      <header className="page-heading">
        <p className="eyebrow">Experiment detail</p>
        <h1>{experiment}</h1>
        <p className="lede">Compare every Phase 6 configuration and seed.</p>
      </header>
      {error && <div className="alert">{error}</div>}
      <div className="table-wrap">
        <table>
          <thead><tr><th>Run</th><th>Compression</th><th>Strategy</th><th>Connectivity</th><th>Bandwidth</th><th>Seed</th><th>AUROC</th></tr></thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.external_id}>
                <td><Link href={`/runs/${encodeURIComponent(run.external_id)}`} className="table-link">{run.external_id}</Link></td>
                <td><span className="tag">{run.compression_setting}</span></td>
                <td>{run.aggregation_strategy}</td>
                <td>{run.dropout_condition}</td>
                <td>{run.bandwidth_kbps == null ? "Uncapped" : `${run.bandwidth_kbps} kbps`}</td>
                <td>{run.seed}</td>
                <td>{run.final_metrics?.test_macro_auroc == null ? "—" : Number(run.final_metrics.test_macro_auroc).toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  );
}
