"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";

const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

export default function ClientDetailPage() {
  const { id } = useParams();
  const [client, setClient] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    fetch(`${apiUrl}/clients/${encodeURIComponent(decodeURIComponent(id))}`).then((response) => response.json()).then((payload) => {
      if (payload.error) throw new Error(payload.error);
      setClient(payload.data);
    }).catch((reason) => setError(reason.message));
  }, [id]);
  return <main className="shell"><Link href="/clients" className="back-link">← Clinics</Link>{error && <div className="alert">{error}</div>}{client && <><header className="page-heading"><p className="eyebrow">Clinic detail</p><h1>{client.external_id}</h1><p className="lede">Static metadata from the Phase 1 partition and Phase 5 connectivity profile.</p></header><section className="metrics-grid compact"><div className="metric"><span>Client index</span><strong>{client.client_index}</strong></div><div className="metric"><span>Data volume</span><strong>{client.data_volume == null ? "—" : client.data_volume.toLocaleString()}</strong></div><div className="metric"><span>Connectivity</span><strong>{client.connectivity_profile_id || "—"}</strong></div></section><section className="panel"><h3>Label distribution</h3><pre className="json">{JSON.stringify(client.label_distribution_summary || {}, null, 2)}</pre></section></>}</main>;
}
