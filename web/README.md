# Phase 7 dashboard

Exact Next.js, React and Express versions are registered in the existing frontend/backend directories. Install the Node LTS version pinned in `.nvmrc` before `npm ci` in either directory. Package lockfiles preserve dependency resolution. The dashboard is a read-only visualization layer over the Phase 6 Supabase
results. The Express API owns the Supabase service-role key; the Next.js app
only talks to the API.

## Local development

Install the pinned Node version from `.nvmrc` (Node 24), then install both
applications:

```bash
cd web/backend
npm ci
cp .env.example .env
# Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env
npm run dev
```

In a second terminal:

```bash
cd web/frontend
npm ci
cp .env.local.example .env.local
npm run dev
```

Open <http://localhost:3000>. The API health check is
<http://localhost:4000/health>.

The backend endpoints are:

- `GET /experiments`
- `GET /experiments/:id/runs`
- `GET /runs/:id/rounds`
- `GET /runs/:id/clients`
- `GET /experiments/:id/joint-metric`
- `GET /clients`
- `GET /clients/:id`
