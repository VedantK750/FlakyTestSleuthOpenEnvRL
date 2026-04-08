from __future__ import annotations


def render_home_page() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FlakyGym Control Center</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg-top: #edf7f1;
      --bg-bottom: #d2ead8;
      --ink: #17211d;
      --muted: #4c6359;
      --accent: #0f8b63;
      --accent-2: #e5783b;
      --panel: rgba(255, 255, 255, 0.86);
      --border: rgba(15, 139, 99, 0.22);
      --card-shadow: 0 22px 46px rgba(14, 51, 37, 0.16);
      --display: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      --mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      color: var(--ink);
      font-family: var(--display);
      min-height: 100vh;
      background:
        radial-gradient(circle at 12% 16%, rgba(230, 120, 59, 0.16), transparent 42%),
        radial-gradient(circle at 86% 12%, rgba(15, 139, 99, 0.2), transparent 40%),
        linear-gradient(164deg, var(--bg-top), var(--bg-bottom));
      animation: backdropFade 700ms ease-out;
    }

    @keyframes backdropFade {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .shell {
      max-width: 1100px;
      margin: 24px auto;
      padding: 0 16px 24px;
      display: grid;
      gap: 16px;
    }

    .hero {
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 20px;
      box-shadow: var(--card-shadow);
      padding: 22px 22px 18px;
      animation: slideIn 500ms ease-out;
    }

    @keyframes slideIn {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 6px rgba(15, 139, 99, 0.15);
    }

    h1 {
      margin: 10px 0 8px;
      font-size: clamp(1.6rem, 2.6vw, 2.35rem);
      line-height: 1.1;
      letter-spacing: -0.02em;
    }

    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 760px;
      line-height: 1.5;
    }

    .panel-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }

    .panel {
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 20px;
      box-shadow: var(--card-shadow);
      padding: 18px;
      animation: slideIn 560ms ease-out;
    }

    .panel h2 {
      margin: 0 0 12px;
      font-size: 1.1rem;
      letter-spacing: -0.01em;
    }

    .brief-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .brief-card {
      border: 1px solid rgba(15, 139, 99, 0.22);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.84);
      padding: 10px;
      display: grid;
      gap: 8px;
    }

    .brief-card h3 {
      margin: 0;
      font-size: 0.95rem;
      letter-spacing: -0.01em;
    }

    .brief-card p {
      margin: 0;
      font-size: 12px;
      color: #36564a;
      line-height: 1.45;
    }

    .brief-list {
      margin: 0;
      padding-left: 16px;
      font-size: 12px;
      color: #2f4f43;
      line-height: 1.45;
      display: grid;
      gap: 5px;
    }

    .header-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .header-chips code {
      background: #eef8f3;
      border: 1px solid rgba(15, 139, 99, 0.26);
      border-radius: 999px;
      padding: 4px 8px;
      font: 500 11px/1.1 var(--mono);
      color: #20443a;
      white-space: nowrap;
    }

    .form-grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .field {
      display: grid;
      gap: 6px;
    }

    .field.span-2 {
      grid-column: span 2;
    }

    label {
      font-size: 13px;
      color: var(--muted);
    }

    input, select {
      width: 100%;
      border: 1px solid rgba(18, 88, 63, 0.22);
      border-radius: 10px;
      padding: 10px 11px;
      font: 500 14px/1.2 var(--mono);
      color: var(--ink);
      background: rgba(255, 255, 255, 0.92);
      transition: border-color 180ms ease, box-shadow 180ms ease;
    }

    input:focus, select:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 4px rgba(15, 139, 99, 0.14);
    }

    .task-picker {
      display: grid;
      gap: 8px;
      padding: 10px;
      border-radius: 12px;
      border: 1px solid rgba(15, 139, 99, 0.22);
      background: rgba(255, 255, 255, 0.85);
    }

    .task-picker-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
    }

    .btn-add-task {
      background: #1c6f50;
      color: #fff;
      padding: 0 14px;
    }

    .task-chips {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      min-height: 32px;
      align-items: center;
    }

    .task-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid rgba(20, 67, 50, 0.32);
      border-radius: 999px;
      background: #fff8dc;
      color: #2a362f;
      padding: 6px 10px;
      font: 600 12px/1 var(--display);
    }

    .task-chip button {
      border: 0;
      background: transparent;
      color: #4a2a12;
      padding: 0;
      border-radius: 0;
      min-width: unset;
      font: 700 12px/1 var(--display);
      line-height: 1;
      cursor: pointer;
    }

    .task-chip button:hover {
      transform: none;
      filter: brightness(0.92);
    }

    .task-empty {
      font: 500 12px/1.2 var(--mono);
      color: #52695f;
    }

    .field-note {
      margin: 0;
      color: #4e655b;
      font-size: 12px;
      line-height: 1.4;
    }

    .slider-wrap {
      display: grid;
      gap: 8px;
    }

    .slider-value-row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font: 500 12px/1.3 var(--mono);
      color: #3c5950;
    }

    input[type="range"] {
      width: 100%;
      padding: 0;
      accent-color: var(--accent);
      cursor: pointer;
    }

    .eta-box {
      border: 1px solid rgba(15, 139, 99, 0.22);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.85);
      padding: 10px;
      display: grid;
      gap: 6px;
    }

    .eta-value {
      font: 700 17px/1 var(--display);
      letter-spacing: -0.01em;
    }

    .eta-warning {
      font: 500 12px/1.45 var(--mono);
      border-radius: 8px;
      padding: 6px 8px;
      display: none;
    }

    .eta-warning.warn {
      display: block;
      color: #752f1b;
      background: #fff0d7;
      border: 1px solid rgba(213, 114, 65, 0.45);
    }

    .eta-warning.ok {
      display: block;
      color: #20443a;
      background: #e8f7ef;
      border: 1px solid rgba(15, 139, 99, 0.28);
    }

    .actions {
      margin-top: 6px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    button {
      border: 0;
      border-radius: 11px;
      font: 600 14px/1 var(--display);
      padding: 11px 14px;
      cursor: pointer;
      transition: transform 180ms ease, opacity 180ms ease, filter 180ms ease;
    }

    button:hover {
      transform: translateY(-1px);
    }

    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
      transform: none;
    }

    .btn-run {
      background: var(--accent);
      color: #fff;
    }

    .btn-stop {
      background: #f2b38f;
      color: #431d05;
    }

    .btn-docs {
      background: #dce8e1;
      color: #234437;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      border-radius: 11px;
      padding: 11px 14px;
      font: 600 14px/1 var(--display);
    }

    .status-row {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      border-radius: 999px;
      padding: 6px 11px;
      font: 600 13px/1 var(--display);
      background: #e1ece6;
      color: #2d4b3e;
    }

    .pill .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #60756a;
    }

    .pill.running .status-dot {
      background: var(--accent);
      box-shadow: 0 0 0 7px rgba(15, 139, 99, 0.12);
    }

    .pill.failed .status-dot {
      background: #af4020;
    }

    .pill.completed .status-dot {
      background: #1d724e;
    }

    .pill.stopped .status-dot {
      background: var(--accent-2);
    }

    .meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      font: 500 12px/1.4 var(--mono);
      color: #2f4f43;
    }

    .meta strong {
      color: #193428;
    }

    .log-wrap {
      margin-top: 8px;
      border-radius: 14px;
      border: 1px solid rgba(20, 66, 50, 0.2);
      overflow: hidden;
      background: #0f1a16;
    }

    .log-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: #b7d5c8;
      font: 500 12px/1 var(--mono);
      padding: 10px 12px;
      border-bottom: 1px solid rgba(170, 208, 193, 0.16);
      background: #13201b;
    }

    pre {
      margin: 0;
      padding: 12px;
      color: #d8f3e7;
      font: 400 12.5px/1.45 var(--mono);
      max-height: 360px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .help {
      margin-top: 8px;
      color: #37594b;
      font-size: 12px;
      line-height: 1.45;
    }

    @media (max-width: 880px) {
      .form-grid {
        grid-template-columns: 1fr;
      }

      .brief-grid {
        grid-template-columns: 1fr;
      }

      .field.span-2 {
        grid-column: span 1;
      }

      .meta {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <span class="eyebrow"><span class="dot"></span>FlakyGym Space</span>
      <h1>FlakyGym Control Center</h1>
      <p>This console runs flaky-test benchmark episodes and streams live logs. Use it to configure runs, estimate runtime, and review grader outcomes quickly.</p>
    </section>

    <section class="panel-grid">
      <div class="panel">
        <h2>Quick Brief: Dataset + Graders</h2>
        <div class="brief-grid">
          <div class="brief-card">
            <h3>Dataset: <code>dataset/py_tasks.csv</code></h3>
            <p>Each row is one flaky-test investigation task created from <code>py-data.csv</code> (repo + SHA + target test + labels + optional known fix diff).</p>
            <p class="field-note">Headers:</p>
            <div class="header-chips">
              <code>repo_url</code><code>sha</code><code>test_name</code><code>test_file</code>
              <code>category</code><code>label</code><code>status</code><code>pr_link</code>
              <code>task_types</code><code>test_code</code><code>known_fix_diff</code>
            </div>
          </div>

          <div class="brief-card">
            <h3>3 Graders (short)</h3>
            <ul class="brief-list">
              <li><strong>Task 1 (`classify`):</strong> exact-match flaky vs stable.</li>
              <li><strong>Task 2 (`root_cause`):</strong> category similarity matrix (partial credit allowed).</li>
              <li><strong>Task 3 (`fix_proposal`):</strong> weighted score from pattern match, patch applicability, and LLM judge.</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="panel">
        <h2>Run Configuration</h2>
        <form id="run-form" class="form-grid">
          <div class="field span-2">
            <label for="dataset_path">Dataset Path</label>
            <input id="dataset_path" name="dataset_path" value="dataset/py_tasks.csv" />
          </div>

          <div class="field">
            <label for="episodes_per_task">Episodes Per Task</label>
            <div class="slider-wrap">
              <input id="episodes_per_task" name="episodes_per_task" type="range" min="1" max="100" step="1" value="1" />
              <div class="slider-value-row">
                <span><strong id="episodes_per_task_value">1</strong> episode(s)</span>
                <span>1-100</span>
              </div>
            </div>
          </div>

          <div class="field">
            <label for="max_steps">Max Steps</label>
            <div class="slider-wrap">
              <input id="max_steps" name="max_steps" type="range" min="1" max="100" step="1" value="20" />
              <div class="slider-value-row">
                <span><strong id="max_steps_value">20</strong> step(s)</span>
                <span>1-100</span>
              </div>
            </div>
          </div>

          <div class="field span-2">
            <label for="task-type-select">Task Types</label>
            <div class="task-picker">
              <div id="task-chips" class="task-chips"></div>
              <div class="task-picker-row">
                <select id="task-type-select">
                  <option value="">Choose task type</option>
                </select>
                <button id="btn-add-task" class="btn-add-task" type="button">Add</button>
              </div>
              <input id="task_types" name="task_types" type="hidden" value="classify,root_cause,fix_proposal" />
            </div>
            <p class="field-note">Add from dropdown, remove with <code>x</code> on each chip.</p>
          </div>

          <div class="field span-2">
            <label for="benchmark_name">Benchmark Label</label>
            <input id="benchmark_name" name="benchmark_name" value="flakysleuth" />
          </div>

          <div class="field span-2">
            <label>Runtime ETA</label>
            <div class="eta-box">
              <div class="eta-value" id="eta-value">~09m 00s</div>
              <p class="field-note" id="eta-detail">3 task(s) × 1 episode(s) × 180s/episode</p>
              <div class="eta-warning" id="eta-warning"></div>
            </div>
          </div>

          <div class="field span-2">
            <label for="api_base_url">API Base URL (optional)</label>
            <input id="api_base_url" name="api_base_url" placeholder="https://api.openai.com/v1 or provider endpoint" />
          </div>

          <div class="field">
            <label for="model_name">Model Name (optional)</label>
            <input id="model_name" name="model_name" placeholder="gpt-4o-mini, qwen/qwen3.6-plus:free, etc." />
          </div>

          <div class="field">
            <label for="api_key">API Key (optional)</label>
            <input id="api_key" name="api_key" type="password" placeholder="Uses server env vars if empty" />
          </div>
        </form>

        <div class="actions">
          <button id="btn-run" class="btn-run" type="button">Start Inference</button>
          <button id="btn-stop" class="btn-stop" type="button">Stop Run</button>
          <a class="btn-docs" href="/docs" target="_blank" rel="noreferrer">Open API Docs</a>
        </div>
        <p class="help">Tip: if no API key is provided, <code>inference.py</code> falls back to its heuristic agent.</p>
      </div>

      <div class="panel">
        <h2>Run Status</h2>
        <div class="status-row">
          <div id="status-pill" class="pill"><span class="status-dot"></span><span id="status-text">idle</span></div>
          <div class="meta">
            <div><strong>Job ID:</strong> <span id="meta-job-id">-</span></div>
            <div><strong>Return Code:</strong> <span id="meta-return-code">-</span></div>
            <div><strong>Started:</strong> <span id="meta-started">-</span></div>
            <div><strong>Finished:</strong> <span id="meta-finished">-</span></div>
          </div>
          <div class="log-wrap">
            <div class="log-head">
              <span>Live Logs</span>
              <span id="log-count">0 lines</span>
            </div>
            <pre id="log-output">No run started yet.</pre>
          </div>
          <div class="help" id="summary-line"></div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const form = document.getElementById("run-form");
    const runButton = document.getElementById("btn-run");
    const stopButton = document.getElementById("btn-stop");
    const statusPill = document.getElementById("status-pill");
    const statusText = document.getElementById("status-text");
    const jobIdEl = document.getElementById("meta-job-id");
    const returnCodeEl = document.getElementById("meta-return-code");
    const startedEl = document.getElementById("meta-started");
    const finishedEl = document.getElementById("meta-finished");
    const logEl = document.getElementById("log-output");
    const logCountEl = document.getElementById("log-count");
    const summaryEl = document.getElementById("summary-line");
    const taskInput = document.getElementById("task_types");
    const taskChipsEl = document.getElementById("task-chips");
    const taskSelectEl = document.getElementById("task-type-select");
    const taskAddButton = document.getElementById("btn-add-task");
    const episodesInput = document.getElementById("episodes_per_task");
    const episodesValueEl = document.getElementById("episodes_per_task_value");
    const maxStepsInput = document.getElementById("max_steps");
    const maxStepsValueEl = document.getElementById("max_steps_value");
    const etaValueEl = document.getElementById("eta-value");
    const etaDetailEl = document.getElementById("eta-detail");
    const etaWarningEl = document.getElementById("eta-warning");

    const TASK_TYPE_ORDER = ["classify", "root_cause", "fix_proposal"];
    const TASK_TYPE_LABELS = {
      classify: "Classify",
      root_cause: "Root Cause",
      fix_proposal: "Fix Proposal",
    };
    const ETA_SECONDS_PER_EPISODE = 180;
    const HACKATHON_LIMIT_SECONDS = 20 * 60;

    function clampInt(raw, min, max, fallback) {
      const num = Number(raw);
      if (!Number.isFinite(num)) return fallback;
      return Math.max(min, Math.min(max, Math.trunc(num)));
    }

    function formatDuration(totalSeconds) {
      const seconds = Math.max(0, Math.round(totalSeconds));
      const mins = Math.floor(seconds / 60);
      const secs = seconds % 60;
      if (mins >= 60) {
        const hrs = Math.floor(mins / 60);
        const remMins = mins % 60;
        return `${hrs}h ${String(remMins).padStart(2, "0")}m ${String(secs).padStart(2, "0")}s`;
      }
      return `${String(mins).padStart(2, "0")}m ${String(secs).padStart(2, "0")}s`;
    }

    function refreshSliderValues() {
      const episodes = clampInt(episodesInput.value, 1, 100, 1);
      const maxSteps = clampInt(maxStepsInput.value, 1, 100, 20);
      episodesInput.value = String(episodes);
      maxStepsInput.value = String(maxSteps);
      episodesValueEl.textContent = String(episodes);
      maxStepsValueEl.textContent = String(maxSteps);
    }

    function parseTaskTypes(raw) {
      const tokens = String(raw || "")
        .split(",")
        .map((token) => token.trim())
        .filter(Boolean);
      const unique = [];
      for (const token of tokens) {
        if (TASK_TYPE_ORDER.includes(token) && !unique.includes(token)) {
          unique.push(token);
        }
      }
      return unique;
    }

    let selectedTaskTypes = parseTaskTypes(taskInput.value);
    if (!selectedTaskTypes.length) {
      selectedTaskTypes = [...TASK_TYPE_ORDER];
    }

    function renderTaskSelect() {
      taskSelectEl.innerHTML = "";

      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "Choose task type";
      placeholder.selected = true;
      taskSelectEl.appendChild(placeholder);

      for (const taskType of TASK_TYPE_ORDER) {
        if (selectedTaskTypes.includes(taskType)) continue;
        const option = document.createElement("option");
        option.value = taskType;
        option.textContent = TASK_TYPE_LABELS[taskType];
        taskSelectEl.appendChild(option);
      }

      const hasChoices = taskSelectEl.options.length > 1;
      taskSelectEl.disabled = !hasChoices;
      taskAddButton.disabled = !hasChoices;
    }

    function renderTaskChips() {
      taskChipsEl.innerHTML = "";

      if (!selectedTaskTypes.length) {
        const hint = document.createElement("span");
        hint.className = "task-empty";
        hint.textContent = "No task selected yet.";
        taskChipsEl.appendChild(hint);
        return;
      }

      for (const taskType of selectedTaskTypes) {
        const chip = document.createElement("span");
        chip.className = "task-chip";

        const chipText = document.createElement("span");
        chipText.textContent = TASK_TYPE_LABELS[taskType] || taskType;

        const chipRemove = document.createElement("button");
        chipRemove.type = "button";
        chipRemove.textContent = "x";
        chipRemove.setAttribute("aria-label", `Remove ${chipText.textContent}`);
        chipRemove.addEventListener("click", () => {
          selectedTaskTypes = selectedTaskTypes.filter((value) => value !== taskType);
          syncTaskTypes();
        });

        chip.appendChild(chipText);
        chip.appendChild(chipRemove);
        taskChipsEl.appendChild(chip);
      }
    }

    function syncTaskTypes() {
      taskInput.value = selectedTaskTypes.join(",");
      renderTaskChips();
      renderTaskSelect();
      updateRuntimeEstimate();
    }

    function addSelectedTaskType() {
      const selected = taskSelectEl.value.trim();
      if (!selected) return;
      if (!selectedTaskTypes.includes(selected)) {
        selectedTaskTypes.push(selected);
      }
      syncTaskTypes();
    }

    function updateRuntimeEstimate() {
      const episodes = clampInt(episodesInput.value, 1, 100, 1);
      const maxSteps = clampInt(maxStepsInput.value, 1, 100, 20);
      const taskCount = selectedTaskTypes.length;
      const totalEpisodes = taskCount * episodes;
      const etaSeconds = totalEpisodes * ETA_SECONDS_PER_EPISODE;

      etaValueEl.textContent = `~${formatDuration(etaSeconds)}`;
      etaDetailEl.textContent =
        `${taskCount} task(s) × ${episodes} episode(s) × ${ETA_SECONDS_PER_EPISODE}s/episode`;

      const notes = [];
      if (episodes > 2) {
        notes.push("Recommended: keep episodes per task at 1-2 for faster hackathon runs.");
      }
      if (etaSeconds > HACKATHON_LIMIT_SECONDS) {
        notes.push("Warning: ETA exceeds 20 minutes, which may violate hackathon runtime guidance.");
      }
      if (maxSteps > 20) {
        notes.push("Higher max steps can increase runtime beyond this ETA estimate.");
      }
      if (taskCount === 0) {
        notes.push("Add at least one task chip to run inference.");
      }

      etaWarningEl.classList.remove("warn", "ok");
      if (!notes.length) {
        etaWarningEl.textContent = "Runtime looks within limits for a quick benchmark run.";
        etaWarningEl.classList.add("ok");
        return;
      }

      etaWarningEl.textContent = notes.join(" ");
      if (etaSeconds > HACKATHON_LIMIT_SECONDS || episodes > 2 || taskCount === 0) {
        etaWarningEl.classList.add("warn");
      } else {
        etaWarningEl.classList.add("ok");
      }
    }

    function readFormPayload() {
      const episodes = clampInt(episodesInput.value, 1, 100, 1);
      const maxSteps = clampInt(maxStepsInput.value, 1, 100, 20);
      episodesInput.value = String(episodes);
      maxStepsInput.value = String(maxSteps);
      refreshSliderValues();
      updateRuntimeEstimate();
      return {
        dataset_path: form.dataset_path.value.trim(),
        episodes_per_task: episodes,
        task_types: form.task_types.value.trim(),
        max_steps: maxSteps,
        benchmark_name: form.benchmark_name.value.trim(),
        api_base_url: form.api_base_url.value.trim() || null,
        model_name: form.model_name.value.trim() || null,
        api_key: form.api_key.value.trim() || null,
      };
    }

    function formatTime(epoch) {
      if (!epoch) return "-";
      try {
        return new Date(epoch * 1000).toLocaleString();
      } catch (_) {
        return "-";
      }
    }

    function setStatus(status) {
      const normalized = (status || "idle").toLowerCase();
      statusPill.classList.remove("running", "failed", "completed", "stopped");
      if (["running", "failed", "completed", "stopped"].includes(normalized)) {
        statusPill.classList.add(normalized);
      }
      statusText.textContent = normalized;
      runButton.disabled = normalized === "running" || normalized === "starting";
      stopButton.disabled = !(normalized === "running" || normalized === "starting");
    }

    function renderSummary(summaries) {
      if (!Array.isArray(summaries) || summaries.length === 0) {
        summaryEl.textContent = "";
        return;
      }
      const last = summaries[summaries.length - 1];
      summaryEl.textContent = `Latest episode: success=${last.success} score=${last.score} steps=${last.steps}`;
    }

    function renderStatus(state) {
      setStatus(state.status || "idle");
      jobIdEl.textContent = state.job_id || "-";
      returnCodeEl.textContent = state.return_code === null || state.return_code === undefined ? "-" : String(state.return_code);
      startedEl.textContent = formatTime(state.started_at);
      finishedEl.textContent = formatTime(state.finished_at);

      const logs = Array.isArray(state.logs) ? state.logs : [];
      logCountEl.textContent = `${logs.length} lines`;
      logEl.textContent = logs.length ? logs.join("\\n") : "No logs yet.";
      logEl.scrollTop = logEl.scrollHeight;

      renderSummary(state.summaries || []);
    }

    async function fetchStatus() {
      try {
        const response = await fetch("/web/inference/status?tail=450", { method: "GET" });
        if (!response.ok) return;
        const state = await response.json();
        renderStatus(state);
      } catch (_) {}
    }

    async function startRun() {
      runButton.disabled = true;
      try {
        if (!selectedTaskTypes.length) {
          alert("Please add at least one task type.");
          return;
        }
        const response = await fetch("/web/inference/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(readFormPayload()),
        });

        const payload = await response.json();
        if (!response.ok) {
          const detail = typeof payload.detail === "string" ? payload.detail : "Could not start inference.";
          alert(detail);
          return;
        }
        renderStatus(payload);
      } catch (_) {
        alert("Could not start inference. Check logs and try again.");
      } finally {
        form.api_key.value = "";
        const isActive = ["running", "starting"].includes((statusText.textContent || "").toLowerCase());
        if (!isActive) {
          runButton.disabled = false;
        }
      }
    }

    async function stopRun() {
      stopButton.disabled = true;
      try {
        const response = await fetch("/web/inference/stop", { method: "POST" });
        if (!response.ok) return;
        const state = await response.json();
        renderStatus(state);
      } catch (_) {}
    }

    runButton.addEventListener("click", startRun);
    stopButton.addEventListener("click", stopRun);
    taskAddButton.addEventListener("click", addSelectedTaskType);
    taskSelectEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        addSelectedTaskType();
      }
    });
    episodesInput.addEventListener("input", () => {
      refreshSliderValues();
      updateRuntimeEstimate();
    });
    maxStepsInput.addEventListener("input", () => {
      refreshSliderValues();
      updateRuntimeEstimate();
    });

    refreshSliderValues();
    syncTaskTypes();
    updateRuntimeEstimate();

    fetchStatus();
    window.setInterval(fetchStatus, 2200);
  </script>
</body>
</html>
"""
