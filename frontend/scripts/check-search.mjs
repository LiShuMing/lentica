/** Focused handler/render contract check; hooks and HTTP are mocked, no browser. */
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

const states = [], mutations = [], jobs = [], requests = [];
let stateIndex = 0, mutationIndex = 0, respond;
const hooks = {
  useState(initial) {
    const index = stateIndex++;
    if (!(index in states)) states[index] = initial;
    return [states[index], (value) => { states[index] = value; }];
  },
  useQueryClient: () => ({ invalidateQueries() {} }),
  useMutation(options) {
    const index = mutationIndex++;
    if (!mutations[index]) {
      const mutation = {
        isPending: false, data: undefined, error: null,
        reset() { this.data = undefined; this.error = null; },
        mutate(value) {
          this.isPending = true;
          this.error = null;
          jobs.push(Promise.resolve().then(() => this.options.mutationFn(value)).then(
            (data) => { this.data = data; this.options.onSuccess?.(data); },
            (error) => { this.error = error; }
          ).finally(() => { this.isPending = false; }));
        }
      };
      mutations[index] = mutation;
    }
    mutations[index].options = options;
    return mutations[index];
  }
};

const originalFetch = globalThis.fetch;
globalThis.fetch = (path, options) => {
  requests.push({ path, body: JSON.parse(options.body) });
  return new Promise((resolve) => { respond = resolve; });
};
const client = load("../src/api/client.ts");
const { PhotosPage } = load("../src/pages/PhotosPage.tsx", {
  react: { ...React, useState: hooks.useState },
  "@tanstack/react-query": hooks,
  "../api/client": client
});

function render() {
  stateIndex = mutationIndex = 0;
  return PhotosPage();
}

function find(tree, predicate) {
  if (Array.isArray(tree)) return tree.flatMap((node) => find(node, predicate));
  if (!React.isValidElement(tree)) return [];
  return [...(predicate(tree) ? [tree] : []), ...find(tree.props.children, predicate)];
}

const scopeControl = (tree) => find(tree, (node) => node.props["aria-label"] === "检索范围")[0];
function submit(text) {
  let tree = render();
  find(tree, (node) => node.type === "input")[0].props.onChange({ target: { value: text } });
  tree = render();
  find(tree, (node) => node.type === "form")[0].props.onSubmit({ preventDefault() {} });
}

try {
  assert.match(renderToStaticMarkup(render()), /先索引照片或扫描文档/);
  submit("   ");
  assert.equal(jobs.length, 0);
  submit("  red evidence  ");
  await Promise.resolve();
  assert.equal(requests[0].path, "/api/v2/search");
  assert.deepEqual(requests[0].body.asset_types, ["photo", "screenshot", "document"]);
  assert.equal(requests[0].body.query, "red evidence");
  assert.equal(scopeControl(render()).props.disabled, true);
  assert.match(renderToStaticMarkup(render()), /搜索中/);

  const citation = { chunk_id: 1, windows_path: "document.md", page_or_segment: "seg.1", excerpt: "unique evidence" };
  const photo = {
    file_name: "photo.png", windows_path: "photo.png", caption: "red object", thumb_path: "/thumb.png",
    confidence: .5, score: .1, image_score: .3, text_score: .2, metadata_score: .1,
    namespace_scores: {}, tags: [], matched_terms: [], match_reason: "visual"
  };
  // Deliberately inconsistent compatibility arrays prove rendering follows results.
  respond({ ok: true, json: async () => ({ answer: "mixed", hits: [], document_hits: [], results: [
    { file_id: 2, rank: 1, photo: null, documents: [citation] },
    { file_id: 1, rank: 2, photo, documents: [] }
  ] }) });
  await Promise.all(jobs.splice(0));
  const html = renderToStaticMarkup(render());
  assert.ok(html.indexOf("unique evidence") < html.indexOf("photo.png"));
  assert.match(html, /seg.1/);
  assert.match(html, /api\/photos\/thumb\/thumb.png/);

  scopeControl(render()).props.onChange({ target: { value: "document" } });
  assert.doesNotMatch(renderToStaticMarkup(render()), /unique evidence/);
  submit("none");
  await Promise.resolve();
  assert.deepEqual(requests[1].body.asset_types, ["document"]);
  respond({ ok: true, json: async () => ({ answer: "empty", results: [] }) });
  await Promise.all(jobs.splice(0));
  assert.match(renderToStaticMarkup(render()), /没有匹配当前范围的结果/);

  submit("fail");
  await Promise.resolve();
  respond({ ok: false, text: async () => "retrieval unavailable" });
  await Promise.all(jobs.splice(0));
  assert.match(renderToStaticMarkup(render()), /retrieval unavailable/);
  assert.equal(scopeControl(render()).props.disabled, false);
  console.log("Unified search handler/render checks passed (mock hooks/HTTP; no browser).");
} finally {
  globalThis.fetch = originalFetch;
}
