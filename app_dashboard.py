from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from graph_core import MODEL_PRICES_BRL_PER_1M, run_graph

import html
import json
import time

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
except Exception:  # pragma: no cover
    Counter = Histogram = None
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain"


app = FastAPI(title="Maritaca Hybrid Graph Agentic RAG")


if Counter and Histogram:
    REQUESTS_TOTAL = Counter("mhga_requests_total", "Total graph requests", ["model"])
    GRAPH_LATENCY = Histogram("mhga_graph_latency_seconds", "Graph latency", ["model"])
    JUDGE_TOTAL = Counter("mhga_judge_total", "Judge decisions", ["model", "is_enough"])
    TOOL_SELECTED_TOTAL = Counter("mhga_tool_selected_total", "Selected tools", ["model", "tool"])
else:
    REQUESTS_TOTAL = GRAPH_LATENCY = JUDGE_TOTAL = TOOL_SELECTED_TOTAL = None


class RunRequest(BaseModel):
    question: str
    model: str = "sabiazinho-4"
    max_retries: int = 3


class CompareRequest(BaseModel):
    question: str
    models: list[str]
    max_retries: int = 1


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return DASHBOARD_HTML


@app.post("/api/run")
def api_run(payload: RunRequest) -> dict:
    start = time.time()
    state = run_graph(
        question=payload.question,
        model=payload.model,
        max_retries=payload.max_retries,
    )

    if REQUESTS_TOTAL:
        REQUESTS_TOTAL.labels(payload.model).inc()
        GRAPH_LATENCY.labels(payload.model).observe(time.time() - start)
        if state.get("judge"):
            JUDGE_TOTAL.labels(payload.model, str(state["judge"].get("is_enough"))).inc()
        if state.get("plan"):
            for tool in state["plan"].get("selected_tools", []):
                TOOL_SELECTED_TOTAL.labels(payload.model, tool).inc()

    return state


@app.post("/api/compare")
def api_compare(payload: CompareRequest) -> dict:
    results = []
    for model in payload.models:
        state = run_graph(
            question=payload.question,
            model=model,
            max_retries=payload.max_retries,
        )
        results.append(
            {
                "model": model,
                "answer": state.get("final_answer"),
                "plan": state.get("plan"),
                "judge": state.get("judge"),
                "total_elapsed": state.get("total_elapsed"),
                "total_tokens": state.get("total_tokens"),
                "total_estimated_cost_brl": state.get("total_estimated_cost_brl"),
            }
        )
    return {"question": payload.question, "results": results}


@app.get("/api/prices")
def api_prices() -> dict:
    return MODEL_PRICES_BRL_PER_1M


@app.get("/metrics")
def metrics() -> Response:
    if not generate_latest:
        return Response("prometheus_client not installed\n", media_type="text/plain")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Maritaca Hybrid Graph Agentic RAG</title>
  <style>
    :root {
      --bg: #0d0f0e;
      --panel: #151917;
      --panel-2: #101311;
      --line: #2c352f;
      --text: #f4f7f2;
      --muted: #9ca89f;
      --green: #48d26d;
      --red: #ff6f75;
      --yellow: #e9c46a;
      --blue: #77b7ff;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top, #172018 0, var(--bg) 36rem);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    main {
      width: min(1380px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }

    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 24px;
    }

    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 54px);
      line-height: 0.95;
      letter-spacing: 0;
    }

    .tagline {
      margin-top: 10px;
      color: var(--green);
      font-size: 18px;
      font-weight: 650;
    }

    .top-actions {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    a, a:visited { color: var(--green); }

    .grid {
      display: grid;
      grid-template-columns: minmax(360px, 0.9fr) minmax(500px, 1.25fr);
      gap: 18px;
      align-items: start;
    }

    .panel {
      background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015));
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 18px 55px rgba(0,0,0,0.25);
    }

    .panel h2, .panel h3 {
      margin: 0 0 12px;
      font-size: 15px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    textarea, select, input {
      width: 100%;
      background: #0a0c0b;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      font: inherit;
      outline: none;
    }

    textarea {
      min-height: 132px;
      resize: vertical;
      line-height: 1.45;
    }

    label {
      display: block;
      margin: 14px 0 7px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }

    button {
      border: 0;
      border-radius: 6px;
      padding: 12px 14px;
      color: #061008;
      background: var(--green);
      font-weight: 800;
      cursor: pointer;
      min-height: 44px;
    }

    button.secondary {
      background: #222a24;
      color: var(--text);
      border: 1px solid var(--line);
    }

    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }

    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 14px;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }

    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 88px;
    }

    .metric .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .metric .value {
      margin-top: 8px;
      font-size: 24px;
      font-weight: 800;
    }

    .answer {
      white-space: pre-wrap;
      font-size: 18px;
      line-height: 1.55;
      padding: 18px;
      background: #0a0c0b;
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 128px;
    }

    .trace {
      display: grid;
      gap: 10px;
    }

    .trace-item {
      display: grid;
      grid-template-columns: 170px 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 12px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 8px;
    }

    .node {
      font-weight: 800;
    }

    .elapsed {
      color: var(--yellow);
      font-variant-numeric: tabular-nums;
      font-weight: 800;
    }

    pre {
      overflow: auto;
      max-height: 380px;
      background: #090b0a;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      color: #d9e7dc;
      line-height: 1.45;
      white-space: pre-wrap;
    }

    .chips {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .chip {
      padding: 6px 9px;
      border-radius: 999px;
      background: #203024;
      color: var(--green);
      border: 1px solid #31533a;
      font-size: 13px;
      font-weight: 750;
    }

    .status-enough { color: var(--green); }
    .status-not { color: var(--red); }

    .section-gap { margin-top: 18px; }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }

    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px;
      vertical-align: top;
      text-align: left;
    }

    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.07em; }

    .model-price-table {
      font-size: 13px;
    }

    .model-price-table td,
    .model-price-table th {
      padding: 8px;
    }

    .model-price-table strong {
      color: var(--green);
    }

    .flow-shell {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.8fr);
      gap: 14px;
      align-items: stretch;
    }

    .graph-stage {
      min-height: 520px;
      background:
        linear-gradient(rgba(72, 210, 109, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(72, 210, 109, 0.03) 1px, transparent 1px),
        #080a09;
      background-size: 28px 28px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 10px;
    }

    #mermaidGraph {
      width: 100%;
      min-height: 500px;
    }

    #mermaidGraph svg {
      width: 100%;
      height: auto;
      max-height: 560px;
    }

    .live-trace {
      background: #090b0a;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 520px;
      max-height: 620px;
      overflow: auto;
    }

    .live-step {
      display: grid;
      grid-template-columns: 28px 1fr;
      gap: 10px;
      padding: 10px 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      color: var(--muted);
    }

    .live-step:last-child {
      border-bottom: 0;
    }

    .live-step .badge {
      width: 24px;
      height: 24px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      font-size: 12px;
      font-weight: 900;
      background: #202720;
      color: var(--muted);
      border: 1px solid var(--line);
    }

    .live-step.running {
      color: var(--yellow);
    }

    .live-step.running .badge {
      color: #101008;
      background: var(--yellow);
      box-shadow: 0 0 18px rgba(233, 196, 106, 0.45);
    }

    .live-step.success {
      color: var(--text);
    }

    .live-step.success .badge {
      color: #061008;
      background: var(--green);
      box-shadow: 0 0 18px rgba(72, 210, 109, 0.45);
    }

    .live-message {
      line-height: 1.35;
      font-size: 14px;
    }

    .live-time {
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }

    @media (max-width: 980px) {
      .grid, .metrics, .flow-shell { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
      .trace-item { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Maritaca Hybrid<br/>Graph Agentic RAG</h1>
        <div class="tagline">Built with Brazilian LLMs, for Brazilian AI engineering.</div>
      </div>
      <div class="top-actions">
        <a href="/metrics" target="_blank">Prometheus /metrics</a>
      </div>
    </header>

    <div class="grid">
      <section class="panel">
        <h2>Run Graph</h2>
        <label for="question">Question</label>
        <textarea id="question">Qual o horario de suporte?</textarea>

        <label for="model">Model</label>
        <select id="model">
          <option value="sabiazinho-4">sabiazinho-4</option>
          <option value="sabia-4">sabia-4</option>
          <option value="sabia-4-thinking">sabia-4-thinking</option>
        </select>

        <label for="retries">Max retries</label>
        <input id="retries" type="number" min="0" max="3" value="3" />

        <div class="button-row">
          <button id="runBtn" onclick="runGraph()">Run Graph</button>
          <button id="compareBtn" class="secondary" onclick="compareModels()">Compare Models</button>
        </div>
        <div class="button-row">
          <button class="secondary" onclick="replayMockTrace()">Replay Route</button>
          <button class="secondary" onclick="resetFlow()">Reset Flow</button>
        </div>

        <div class="section-gap">
          <h3>Fixed Model Cost Table</h3>
          <table class="model-price-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Input / 1M</th>
                <th>Output / 1M</th>
                <th>Use case</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>sabiazinho-4</strong></td>
                <td>R$ 1.00</td>
                <td>R$ 4.00</td>
                <td>fast router, judge, cheap graph runs</td>
              </tr>
              <tr>
                <td><strong>sabia-4</strong></td>
                <td>R$ 5.00</td>
                <td>R$ 20.00</td>
                <td>better final answers and synthesis</td>
              </tr>
              <tr>
                <td><strong>sabia-4-thinking</strong></td>
                <td>R$ 5.00</td>
                <td>R$ 40.00</td>
                <td>harder reasoning, highest answer cost</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="section-gap">
          <h3>Selected Tools</h3>
          <div class="chips" id="tools"></div>
        </div>

        <div class="section-gap">
          <h3>Planner Output</h3>
          <pre id="plan">{}</pre>
        </div>

        <div class="section-gap">
          <h3>Judge Output</h3>
          <pre id="judge">{}</pre>
        </div>
      </section>

      <section>
        <div class="metrics">
          <div class="metric"><div class="label">Latency</div><div class="value" id="latency">-</div></div>
          <div class="metric"><div class="label">Tokens</div><div class="value" id="tokens">-</div></div>
          <div class="metric"><div class="label">Cost BRL</div><div class="value" id="cost">-</div></div>
          <div class="metric"><div class="label">Judge</div><div class="value" id="judgeStatus">-</div></div>
        </div>

        <div class="panel">
          <h2>Live Agentic Route</h2>
          <div class="flow-shell">
            <div class="graph-stage">
              <div id="mermaidGraph"></div>
            </div>
            <div class="live-trace" id="liveTrace"></div>
          </div>
        </div>

        <div class="panel">
          <h2>Final Answer</h2>
          <div class="answer" id="answer">Run the graph to generate a grounded answer.</div>
        </div>

        <div class="panel section-gap">
          <h2>Execution Trace</h2>
          <div class="trace" id="trace"></div>
        </div>

        <div class="panel section-gap">
          <h2>Model Comparison</h2>
          <div id="comparison">Run comparison to benchmark models on the same question.</div>
        </div>

        <div class="panel section-gap">
          <h2>Aggregated Context</h2>
          <pre id="context"></pre>
        </div>
      </section>
    </div>
  </main>

  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

    mermaid.initialize({
      startOnLoad: false,
      theme: "base",
      securityLevel: "loose",
      flowchart: { curve: "basis", htmlLabels: true },
      themeVariables: {
        background: "#080a09",
        primaryColor: "#151917",
        primaryTextColor: "#f4f7f2",
        primaryBorderColor: "#2c352f",
        lineColor: "#526057",
        fontFamily: "Inter, ui-sans-serif, system-ui",
      },
    });

    const $ = (id) => document.getElementById(id);
    let animationToken = 0;

    const NODE_IDS = [
      "user_query",
      "planner",
      "executor",
      "cag_memory",
      "rag_memory",
      "aggregator",
      "judge",
      "final_answer",
      "final_response",
      "retry_context",
      "loop_planner",
    ];

    const EDGE_IDS = [
      "user_query->planner",
      "planner->executor",
      "executor->cag_memory",
      "executor->rag_memory",
      "cag_memory->aggregator",
      "rag_memory->aggregator",
      "aggregator->judge",
      "judge->final_answer",
      "final_answer->final_response",
      "judge->retry_context",
      "retry_context->loop_planner",
      "loop_planner->planner",
    ];

    const SUCCESS_TRACE = [
      {
        node_id: "user_query",
        edge_id: null,
        message: "User Query received",
      },
      {
        node_id: "planner",
        edge_id: "user_query->planner",
        message: "Planner selected CAG + RAG and assigned value/cost scores",
      },
      {
        node_id: "executor",
        edge_id: "planner->executor",
        message: "Parallel Graph Executor started selected tools",
      },
      {
        node_id: "cag_memory",
        edge_id: "executor->cag_memory",
        message: "CAG Memory retrieved cached business knowledge",
      },
      {
        node_id: "rag_memory",
        edge_id: "executor->rag_memory",
        message: "RAG Memory retrieved evidence chunks from ChromaDB",
      },
      {
        node_id: "aggregator",
        edge_id: "cag_memory->aggregator",
        extra_edge_id: "rag_memory->aggregator",
        message: "Context Aggregator merged outputs and preserved source labels",
      },
      {
        node_id: "judge",
        edge_id: "aggregator->judge",
        message: "Judge Agent passed sufficiency check",
      },
      {
        node_id: "final_answer",
        edge_id: "judge->final_answer",
        message: "Final Answer Agent generated grounded answer",
      },
      {
        node_id: "final_response",
        edge_id: "final_answer->final_response",
        message: "Final Response completed",
      },
    ];

    function pretty(obj) {
      return JSON.stringify(obj ?? {}, null, 2);
    }

    function setBusy(isBusy) {
      $("runBtn").disabled = isBusy;
      $("compareBtn").disabled = isBusy;
      $("runBtn").textContent = isBusy ? "Running..." : "Run Graph";
    }

    function sleep(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }

    function graphDefinition(nodeStatuses = {}, edgeStatuses = {}) {
      const nodeClass = (id) => nodeStatuses[id] || "idle";
      const edgeClass = (id) => edgeStatuses[id] || "idleEdge";
      const classLine = (status) => {
        const ids = NODE_IDS.filter(id => nodeClass(id) === status);
        return ids.length ? `    class ${ids.join(",")} ${status};` : "";
      };

      return `
flowchart TD
    user_query["User Query"] --> planner["Planner Agent<br/><small>Maritaca / Sabiazinho-4<br/>selected_tools + scores</small>"]
    planner --> executor["Parallel Graph Executor<br/><small>runs selected tools</small>"]

    executor --> cag_memory[("CAG Memory<br/><small>cache.json<br/>FAQs / prices / policies</small>")]
    executor --> rag_memory[("RAG Memory<br/><small>ChromaDB<br/>embeddings / docs / evidence</small>")]

    cag_memory --> aggregator["Context Aggregator<br/><small>merge + source labels</small>"]
    rag_memory --> aggregator

    aggregator --> judge["Judge Agent<br/><small>sufficiency check</small>"]

    judge --> final_answer["Final Answer Agent<br/><small>grounded answer</small>"]
    final_answer --> final_response["Final Response"]

    judge -.-> retry_context["Retry Context<br/><small>missing_info + updated query</small>"]
    retry_context -.-> loop_planner["Loop to Planner<br/><small>max retries = 3</small>"]
    loop_planner -.-> planner

    classDef idle fill:#151917,stroke:#526057,color:#f4f7f2,stroke-width:1px;
    classDef inactive fill:#0b0d0c,stroke:#29312c,color:#6f7a72,stroke-width:1px;
    classDef running fill:#32270d,stroke:#e9c46a,color:#fff3bf,stroke-width:3px;
    classDef success fill:#102417,stroke:#48d26d,color:#d9ffe2,stroke-width:3px;
    classDef retry fill:#2d1d0b,stroke:#ff9f1c,color:#ffe4b0,stroke-width:3px;

${classLine("idle")}
${classLine("inactive")}
${classLine("running")}
${classLine("success")}
${classLine("retry")}

    linkStyle 0 stroke:${edgeColor(edgeClass("user_query->planner"))},stroke-width:${edgeWidth(edgeClass("user_query->planner"))};
    linkStyle 1 stroke:${edgeColor(edgeClass("planner->executor"))},stroke-width:${edgeWidth(edgeClass("planner->executor"))};
    linkStyle 2 stroke:${edgeColor(edgeClass("executor->cag_memory"))},stroke-width:${edgeWidth(edgeClass("executor->cag_memory"))};
    linkStyle 3 stroke:${edgeColor(edgeClass("executor->rag_memory"))},stroke-width:${edgeWidth(edgeClass("executor->rag_memory"))};
    linkStyle 4 stroke:${edgeColor(edgeClass("cag_memory->aggregator"))},stroke-width:${edgeWidth(edgeClass("cag_memory->aggregator"))};
    linkStyle 5 stroke:${edgeColor(edgeClass("rag_memory->aggregator"))},stroke-width:${edgeWidth(edgeClass("rag_memory->aggregator"))};
    linkStyle 6 stroke:${edgeColor(edgeClass("aggregator->judge"))},stroke-width:${edgeWidth(edgeClass("aggregator->judge"))};
    linkStyle 7 stroke:${edgeColor(edgeClass("judge->final_answer"))},stroke-width:${edgeWidth(edgeClass("judge->final_answer"))};
    linkStyle 8 stroke:${edgeColor(edgeClass("final_answer->final_response"))},stroke-width:${edgeWidth(edgeClass("final_answer->final_response"))};
    linkStyle 9 stroke:${edgeColor(edgeClass("judge->retry_context"))},stroke-width:${edgeWidth(edgeClass("judge->retry_context"))};
    linkStyle 10 stroke:${edgeColor(edgeClass("retry_context->loop_planner"))},stroke-width:${edgeWidth(edgeClass("retry_context->loop_planner"))};
    linkStyle 11 stroke:${edgeColor(edgeClass("loop_planner->planner"))},stroke-width:${edgeWidth(edgeClass("loop_planner->planner"))};
`;
    }

    function edgeColor(status) {
      if (status === "success") return "#48d26d";
      if (status === "running") return "#e9c46a";
      if (status === "retry") return "#ff9f1c";
      if (status === "inactive") return "#29312c";
      return "#526057";
    }

    function edgeWidth(status) {
      return status === "success" || status === "running" || status === "retry" ? "3px" : "1.5px";
    }

    async function renderGraph(nodeStatuses = {}, edgeStatuses = {}) {
      const graph = graphDefinition(nodeStatuses, edgeStatuses);
      try {
        const { svg } = await mermaid.render(`mhga-${Date.now()}-${Math.random().toString(16).slice(2)}`, graph);
        $("mermaidGraph").innerHTML = svg;
      } catch (error) {
        $("mermaidGraph").innerHTML = `<pre>Mermaid render error:\n${htmlEscape(error.message || error)}</pre>`;
        console.error("Mermaid graph definition:", graph);
        console.error(error);
      }
    }

    function renderLiveTrace(trace, activeIndex = -1, completedCount = 0) {
      $("liveTrace").innerHTML = trace.map((step, index) => {
        const cls = index === activeIndex ? "running" : index < completedCount ? "success" : "";
        const icon = index < completedCount ? "✓" : index === activeIndex ? "…" : index + 1;
        return `
          <div class="live-step ${cls}">
            <div class="badge">${icon}</div>
            <div>
              <div class="live-message">${htmlEscape(step.message)}</div>
              <div class="live-time">${cls === "success" ? "completed" : cls === "running" ? "running" : "queued"}</div>
            </div>
          </div>
        `;
      }).join("");
    }

    function buildStatusesFromTrace(trace, activeIndex, completedCount) {
      const nodeStatuses = {};
      const edgeStatuses = {};
      NODE_IDS.forEach(id => nodeStatuses[id] = ["retry_context", "loop_planner"].includes(id) ? "inactive" : "idle");
      EDGE_IDS.forEach(id => edgeStatuses[id] = id.includes("retry") || id.includes("loop") || id === "judge->retry_context" ? "inactive" : "idle");

      for (let i = 0; i < completedCount; i++) {
        const step = trace[i];
        if (step.node_id) nodeStatuses[step.node_id] = "success";
        if (step.edge_id) edgeStatuses[step.edge_id] = "success";
        if (step.extra_edge_id) edgeStatuses[step.extra_edge_id] = "success";
      }

      if (activeIndex >= 0 && trace[activeIndex]) {
        const step = trace[activeIndex];
        if (step.node_id) nodeStatuses[step.node_id] = "running";
        if (step.edge_id) edgeStatuses[step.edge_id] = "running";
        if (step.extra_edge_id) edgeStatuses[step.extra_edge_id] = "running";
      }

      return { nodeStatuses, edgeStatuses };
    }

    async function replayTrace(trace = SUCCESS_TRACE, delay = 520) {
      const token = ++animationToken;
      renderLiveTrace(trace, -1, 0);
      await renderGraph();

      for (let i = 0; i < trace.length; i++) {
        if (token !== animationToken) return;
        const active = buildStatusesFromTrace(trace, i, i);
        renderLiveTrace(trace, i, i);
        await renderGraph(active.nodeStatuses, active.edgeStatuses);
        await sleep(delay);

        if (token !== animationToken) return;
        const complete = buildStatusesFromTrace(trace, -1, i + 1);
        renderLiveTrace(trace, -1, i + 1);
        await renderGraph(complete.nodeStatuses, complete.edgeStatuses);
        await sleep(120);
      }
    }

    function traceFromState(state) {
      const tools = state.plan?.selected_tools || [];
      const trace = [
        { node_id: "user_query", edge_id: null, message: "User Query received" },
        {
          node_id: "planner",
          edge_id: "user_query->planner",
          message: `Planner selected ${tools.length ? tools.join(" + ").toUpperCase() : "no tools"}`,
        },
        { node_id: "executor", edge_id: "planner->executor", message: "Parallel Graph Executor started selected tools" },
      ];

      if (tools.includes("cag")) {
        trace.push({ node_id: "cag_memory", edge_id: "executor->cag_memory", message: "CAG Memory retrieved cached records" });
      }
      if (tools.includes("rag")) {
        trace.push({ node_id: "rag_memory", edge_id: "executor->rag_memory", message: "RAG Memory retrieved evidence chunks" });
      }

      trace.push({
        node_id: "aggregator",
        edge_id: tools.includes("cag") ? "cag_memory->aggregator" : "rag_memory->aggregator",
        extra_edge_id: tools.includes("cag") && tools.includes("rag") ? "rag_memory->aggregator" : null,
        message: "Context Aggregator merged sources",
      });
      trace.push({ node_id: "judge", edge_id: "aggregator->judge", message: state.judge?.is_enough ? "Judge Agent passed sufficiency check" : "Judge Agent requested retry context" });

      if (state.judge?.is_enough) {
        trace.push({ node_id: "final_answer", edge_id: "judge->final_answer", message: "Final Answer Agent generated grounded answer" });
        trace.push({ node_id: "final_response", edge_id: "final_answer->final_response", message: "Final Response completed" });
      } else {
        trace.push({ node_id: "retry_context", edge_id: "judge->retry_context", message: "Retry Context received missing_info" });
        trace.push({ node_id: "loop_planner", edge_id: "retry_context->loop_planner", extra_edge_id: "loop_planner->planner", message: "Loop to Planner Agent prepared next step" });
      }

      return trace;
    }

    async function replayMockTrace() {
      await replayTrace(SUCCESS_TRACE);
    }

    async function resetFlow() {
      animationToken++;
      renderLiveTrace(SUCCESS_TRACE, -1, 0);
      await renderGraph();
    }

    function renderState(state) {
      $("answer").textContent = state.final_answer || "";
      $("latency").textContent = `${state.total_elapsed ?? "-"}s`;
      $("tokens").textContent = state.total_tokens ?? "-";
      $("cost").textContent = `R$ ${(state.total_estimated_cost_brl ?? 0).toFixed(6)}`;

      const enough = state.judge?.is_enough;
      $("judgeStatus").textContent = enough === true ? "Enough" : enough === false ? "Not enough" : "-";
      $("judgeStatus").className = `value ${enough ? "status-enough" : "status-not"}`;

      $("plan").textContent = pretty(state.plan);
      $("judge").textContent = pretty(state.judge);
      $("context").textContent = state.aggregated_context || "";

      const tools = state.plan?.selected_tools || [];
      $("tools").innerHTML = tools.map(t => `<span class="chip">${t}</span>`).join("");

      const logs = state.logs || [];
      $("trace").innerHTML = logs.map(log => `
        <div class="trace-item">
          <div class="node">${htmlEscape(log.node)}</div>
          <div>${log.usage?.total_tokens ? `${log.usage.total_tokens} tokens` : "no token usage"}</div>
          <div class="elapsed">${log.elapsed}s</div>
        </div>
      `).join("");
    }

    function htmlEscape(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function runGraph() {
      setBusy(true);
      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            question: $("question").value,
            model: $("model").value,
            max_retries: Number($("retries").value || 0),
          })
        });

        if (!response.ok) throw new Error(await response.text());
        const state = await response.json();
        renderState(state);
        await replayTrace(traceFromState(state), 360);
      } catch (error) {
        $("answer").textContent = `Error: ${error.message}`;
      } finally {
        setBusy(false);
      }
    }

    async function compareModels() {
      setBusy(true);
      $("comparison").textContent = "Comparing models...";
      try {
        const response = await fetch("/api/compare", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            question: $("question").value,
            models: ["sabiazinho-4", "sabia-4", "sabia-4-thinking"],
            max_retries: 1,
          })
        });

        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        $("comparison").innerHTML = `
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Latency</th>
                <th>Tokens</th>
                <th>Cost</th>
                <th>Tools</th>
                <th>Answer</th>
              </tr>
            </thead>
            <tbody>
              ${data.results.map(row => `
                <tr>
                  <td><strong>${htmlEscape(row.model)}</strong></td>
                  <td>${row.total_elapsed}s</td>
                  <td>${row.total_tokens}</td>
                  <td>R$ ${(row.total_estimated_cost_brl ?? 0).toFixed(6)}</td>
                  <td>${htmlEscape((row.plan?.selected_tools || []).join(", "))}</td>
                  <td>${htmlEscape(row.answer || "")}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        `;
      } catch (error) {
        $("comparison").textContent = `Error: ${error.message}`;
      } finally {
        setBusy(false);
      }
    }

    window.runGraph = runGraph;
    window.compareModels = compareModels;
    window.replayMockTrace = replayMockTrace;
    window.resetFlow = resetFlow;

    resetFlow();
  </script>
</body>
</html>
"""
