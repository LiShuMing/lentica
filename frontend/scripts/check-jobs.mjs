/** Focused JobsPage contract check covering loading, errors, degraded mode, empty state, and transitions. */
import assert from "node:assert/strict";
import fs from "node:fs";
import { createRequire } from "node:module";
import vm from "node:vm";

const require = createRequire(import.meta.url);
const ts = require("typescript");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");

function load(relativePath, overrides = {}) {
  const filename = new URL(relativePath, import.meta.url).pathname;
  const source = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX }
  }).outputText;
  const module = { exports: {} };
  const localRequire = createRequire(filename);
  vm.runInThisContext(`(function(require, module, exports) {${source}\n})`, { filename })(
    (name) => overrides[name] ?? localRequire(name), module, module.exports
  );
  return module.exports;
}

const refs = [];
const states = [];
const effects = [];
const queryState = {
  tasks: { data: undefined, error: null, isLoading: true, dataUpdatedAt: 0 },
  "queue-status": { data: undefined, error: null, isLoading: true, dataUpdatedAt: 0 },
  "vision-status": { data: undefined, error: null, isLoading: false, dataUpdatedAt: 0 }
};
const mutations = [];
let refIndex = 0, stateIndex = 0, mutationIndex = 0;

const hooks = {
  useRef(initial) {
    const index = refIndex++;
    if (!(index in refs)) refs[index] = { current: initial };
    return refs[index];
  },
  useState(initial) {
    const index = stateIndex++;
    if (!(index in states)) states[index] = initial;
    return [states[index], (value) => { states[index] = value; }];
  },
  useMemo(factory) {
    return factory();
  },
  useEffect(callback) {
    effects.push(callback);
  },
  useQueryClient: () => ({ invalidateQueries() {} }),
  useQuery({ queryKey }) {
    return queryState[queryKey[0]];
  },
  useMutation(options) {
    const index = mutationIndex++;
    if (!mutations[index]) {
      mutations[index] = { isPending: false, data: undefined, error: null, mutate() {} };
    }
    mutations[index].options = options;
    return mutations[index];
  }
};

globalThis.window = { setTimeout: () => 1, clearTimeout() {} };
const client = load("../src/api/client.ts");
const { JobsPage } = load("../src/pages/JobsPage.tsx", {
  react: { ...React, useRef: hooks.useRef, useState: hooks.useState, useMemo: hooks.useMemo, useEffect: hooks.useEffect },
  "@tanstack/react-query": hooks,
  "../api/client": client
});

function renderHtml() {
  refIndex = stateIndex = mutationIndex = 0;
  effects.length = 0;
  const html = renderToStaticMarkup(JobsPage());
  for (const effect of effects.splice(0)) effect();
  return html;
}

// 1. Loading state
let html = renderHtml();
assert.match(html, /Loading queue status/);
assert.match(html, /Loading tasks/);

// 2. Empty state when queries succeed with no tasks
queryState.tasks = { data: [], error: null, isLoading: false, dataUpdatedAt: 100 };
queryState["queue-status"] = {
  data: {
    settings: {
      enabled: false,
      auto_enqueue_photos: true,
      auto_enqueue_documents: true,
      auto_adjust_concurrency: true,
      worker_mode: "smoke",
      max_concurrency: 0,
      photo_limit: 100,
      photo_max_indexed: 10,
      photo_probe_limit: 1000,
      document_max_files: 100,
      document_max_mb: 20,
      document_interval_seconds: 600,
      vision_backend: "basic",
      vision_endpoint: "http://127.0.0.1:8100",
      vision_model: "qwen",
      vision_timeout_seconds: 180,
      vision_fallback: "basic"
    },
    gpu: { available: false },
    desired_concurrency: 0,
    workers: [],
    task_summary: { index_photos: {}, index_documents: {}, index_faces: {} },
    last_error: null,
    last_document_enqueue: null
  },
  error: null,
  isLoading: false,
  dataUpdatedAt: 100
};
html = renderHtml();
assert.match(html, /No queued or historical tasks yet/);

// 3. API/config error + degraded vision/fallback state + real vector/fallback load display
queryState["queue-status"].error = new Error("Invalid queue settings; refusing to enable defaults");
queryState["vision-status"] = {
  data: { backend: "vllm", endpoint: "http://127.0.0.1:8100", model: "qwen", healthy: false, models: [], error: "refused", fallback: "basic" },
  error: null,
  isLoading: false,
  dataUpdatedAt: 200
};
queryState.tasks = {
  data: [
    {
      id: 7,
      pipeline_stage: "index_photos",
      state: "running",
      priority: 100,
      attempt_count: 1,
      worker_id: "w1",
      error: null,
      payload: {},
      latest_run: {
        id: 1,
        started_at: "2026-11-21T00:00:00Z",
        finished_at: null,
        duration_seconds: 2.0,
        model_profile: "vision:vllm",
        stats: { indexed: 1, vectors_written: 6, vision_fallbacks: 1, degraded: true },
        throughput: { indexed_per_second: 0.5 },
        computational_load: { images_captioned: 1, total_vectors_written: 6, vision_fallbacks: 1, degraded: true },
        raw_output: null,
        error: null
      }
    }
  ],
  error: null,
  isLoading: false,
  dataUpdatedAt: 200
};
html = renderHtml();
assert.match(html, /Queue configuration\/status error: Invalid queue settings/);
assert.match(html, /Degraded mode active:/);
assert.match(html, /1 imgs · 6 vec · 1 fallback/);

// 4. Task state transition highlight (running -> succeeded) and stale poll rejection
queryState.tasks = {
  ...queryState.tasks,
  data: [{ ...queryState.tasks.data[0], state: "succeeded" }],
  dataUpdatedAt: 300
};
renderHtml();
html = renderHtml();
assert.match(html, /state-changed/);

// Stale poll (older timestamp 150 < 300) must not overwrite tracked state
queryState.tasks = {
  ...queryState.tasks,
  data: [{ ...queryState.tasks.data[0], state: "queued" }],
  dataUpdatedAt: 150
};
states[0] = new Set();
renderHtml();
assert.equal(states[0].size, 0);

console.log("JobsPage contract checks passed (mock hooks/HTTP; no browser).");
