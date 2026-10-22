"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

export default function ClientsPage() {
  const [clients, setClients] = useState([]);
  const [error, setError] = useState("");
  useEffect(() => {
    fetch(`${apiUrl}/clients`).then((response) => response.json()).then((payload) => {
      if (payload.error) throw new Error(payload.error);
      setClients(payload.data || []);
    }).catch((reason) => setError(reason.message));
  }, []);
  return (
    <main className="shell">
      <Link href="/" className="back-link">← Overview</Link>
      <header className="page-heading"><p className="eyebrow">Client explorer</p><h1>Clinic network</h1><p className="lede">Inspect the simulated rural clinics behind each experiment.</p></header>
      {error && <div className="alert">{error}</div>}
      <div className="client-grid">{clients.map((client) => <Link className="client-card" href={`/clients/${encodeURIComponent(client.external_id)}`} key={client.external_id}><span className="client-index">CLINIC {String(client.client_index).padStart(2, "0")}</span><h3>{client.external_id}</h3><p>{client.connectivity_profile_id || "Connectivity profile pending"}</p><strong>{client.data_volume == null ? "—" : client.data_volume.toLocaleString()} images</strong></Link>)}</div>
    </main>
  );
}
