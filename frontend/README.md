# Lentica Web

React/TypeScript UI for the local knowledge product. Install with `npm ci`, run
`npm run dev -- --host 127.0.0.1 --port 5173`, and validate with `npm run build`.
`LENTICA_API_PROXY` selects the dev backend (default `http://127.0.0.1:8000`).
See the [workspace spec](../specs/0001-agentic-workspace/README.md) and root README.
The additive V2 search contract is in
[0002](../specs/0002-index-task-consistency/README.md).

[0009](../specs/0009-product-hybrid-integration/README.md) connects the search page
to `/api/v2/search`. It offers all/photo/screenshot/document scope and renders
the authoritative `results` order with image previews and document citations.
The old response arrays remain for compatibility; they do not drive this view.
Run `node frontend/scripts/check-search.mjs` from the repository root for focused
submit/scope/loading/error/empty/mixed-order checks. It uses actual TypeScript
transpilation, API client and React static rendering with mocked hooks/HTTP; it
does not validate a browser, real network or model quality.
