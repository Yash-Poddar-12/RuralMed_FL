import express from "express";
import { isConfigured, supabaseGet } from "./supabase-client.js";

const app = express();
const port = Number(process.env.PORT || 4000);
const frontendOrigin = process.env.FRONTEND_ORIGIN || "http://localhost:3000";

app.use((request, response, next) => {
  response.setHeader("Access-Control-Allow-Origin", frontendOrigin);
  response.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  response.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (request.method === "OPTIONS") return response.sendStatus(204);
  next();
});
app.use(express.json());

const sendError = (response, error) => {
  const status = Number(error.status) || 500;
  response.status(status).json({
    error: status === 500 ? "Internal server error" : error.message,
  });
};

const requireValue = (request, name) => {
  const value = request.params[name];
  if (!value || value.length > 200 || /[<>]/.test(value)) {
    const error = new Error(`Invalid ${name}`);
    error.status = 400;
    throw error;
  }
  return value;
};

app.get("/health", (_request, response) => {
  response.json({ status: "ok", supabaseConfigured: isConfigured() });
});

app.get("/experiments", async (_request, response) => {
  try {
    const experiments = await supabaseGet("experiments", {
      select: "id,name,description,config,created_at",
      order: "created_at.desc",
    });
    const runs = await supabaseGet("runs", {
      select: "experiment_external_name,status,final_metrics",
      limit: "10000",
    });
    const summaries = experiments.map((experiment) => {
      const experimentRuns = runs.filter(
        (run) => run.experiment_external_name === experiment.name,
      );
      const metrics = experimentRuns
        .map((run) => run.final_metrics || {})
        .filter((value) => Number.isFinite(Number(value.test_macro_auroc)));
      return {
        ...experiment,
        runCount: experimentRuns.length,
        completedCount: experimentRuns.filter((run) => run.status === "completed").length,
        bestMacroAuroc: metrics.length
          ? Math.max(...metrics.map((value) => Number(value.test_macro_auroc)))
          : null,
      };
    });
    response.json({ data: summaries });
  } catch (error) {
    sendError(response, error);
  }
});

app.get("/experiments/:id/runs", async (request, response) => {
  try {
    const experiment = requireValue(request, "id");
    const runs = await supabaseGet("runs", {
      select: "id,external_id,experiment_external_name,compression_setting,aggregation_strategy,dropout_condition,bandwidth_kbps,seed,status,parameters,final_metrics,created_at",
      experiment_external_name: `eq.${experiment}`,
      order: "created_at.desc",
      limit: "10000",
    });
    response.json({ data: runs });
  } catch (error) {
    sendError(response, error);
  }
});

app.get("/runs/:id/rounds", async (request, response) => {
  try {
    const run = requireValue(request, "id");
    const rounds = await supabaseGet("rounds", {
      select: "id,run_external_id,round_number,recorded_at,global_accuracy,global_auroc,global_f1,communication_bytes",
      run_external_id: `eq.${run}`,
      order: "round_number.asc",
      limit: "10000",
    });
    response.json({ data: rounds });
  } catch (error) {
    sendError(response, error);
  }
});

app.get("/runs/:id/clients", async (request, response) => {
  try {
    const run = requireValue(request, "id");
    const events = await supabaseGet("client_round_events", {
      select: "client_external_id,round_number,status,staleness,effective_bandwidth,uptime",
      run_external_id: `eq.${run}`,
      order: "round_number.asc",
      limit: "10000",
    });
    response.json({ data: events });
  } catch (error) {
    sendError(response, error);
  }
});

app.get("/experiments/:id/joint-metric", async (request, response) => {
  try {
    const experiment = requireValue(request, "id");
    const runs = await supabaseGet("runs", {
      select: "external_id,compression_setting,aggregation_strategy,dropout_condition,bandwidth_kbps,seed,final_metrics",
      experiment_external_name: `eq.${experiment}`,
      status: "eq.completed",
      limit: "10000",
    });
    const groups = new Map();
    for (const run of runs) {
      const key = [
        run.aggregation_strategy,
        run.compression_setting,
        run.dropout_condition,
        run.bandwidth_kbps ?? "uncapped",
      ].join("|");
      const group = groups.get(key) || { ...run, values: [], seeds: [] };
      const metric = Number(run.final_metrics?.test_macro_auroc);
      if (Number.isFinite(metric)) group.values.push(metric);
      group.seeds.push(run.seed);
      groups.set(key, group);
    }
    const data = [...groups.values()].map((group) => ({
      strategy: group.aggregation_strategy,
      compression: group.compression_setting,
      dropoutCondition: group.dropout_condition,
      bandwidthKbps: group.bandwidth_kbps,
      seeds: group.seeds,
      macroAurocMean: group.values.length
        ? group.values.reduce((sum, value) => sum + value, 0) / group.values.length
        : null,
      macroAurocStd: group.values.length > 1
        ? Math.sqrt(group.values.reduce((sum, value) => sum + (value - group.values.reduce((a, b) => a + b, 0) / group.values.length) ** 2, 0) / (group.values.length - 1))
        : 0,
    }));
    response.json({ data });
  } catch (error) {
    sendError(response, error);
  }
});

app.get("/clients/:id", async (request, response) => {
  try {
    const client = requireValue(request, "id");
    const rows = await supabaseGet("clients", {
      select: "id,external_id,experiment_external_name,client_index,data_volume,label_distribution_summary,connectivity_profile_id",
      external_id: `eq.${client}`,
      limit: "1",
    });
    if (!rows.length) return response.status(404).json({ error: "Client not found" });
    response.json({ data: rows[0] });
  } catch (error) {
    sendError(response, error);
  }
});

app.get("/clients", async (_request, response) => {
  try {
    const clients = await supabaseGet("clients", {
      select: "id,external_id,experiment_external_name,client_index,data_volume,label_distribution_summary,connectivity_profile_id",
      order: "client_index.asc",
      limit: "10000",
    });
    response.json({ data: clients });
  } catch (error) {
    sendError(response, error);
  }
});

app.use((_request, response) => response.status(404).json({ error: "Route not found" }));
app.use((error, _request, response, _next) => sendError(response, error));

app.listen(port, () => {
  console.log(`RuralMed dashboard API listening on http://localhost:${port}`);
});
