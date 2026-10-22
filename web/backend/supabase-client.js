const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY;

export function isConfigured() {
  return Boolean(SUPABASE_URL && SUPABASE_KEY);
}

export async function supabaseGet(table, params = {}) {
  if (!isConfigured()) {
    const error = new Error("Supabase is not configured");
    error.status = 503;
    throw error;
  }

  const search = new URLSearchParams(params);
  const response = await fetch(`${SUPABASE_URL}/rest/v1/${table}?${search}`, {
    headers: {
      apikey: SUPABASE_KEY,
      Authorization: `Bearer ${SUPABASE_KEY}`,
      Accept: "application/json",
    },
  });
  if (!response.ok) {
    const detail = await response.text();
    const error = new Error(`Supabase query failed for ${table}: ${detail}`);
    error.status = response.status >= 500 ? 502 : 400;
    throw error;
  }
  return response.json();
}
