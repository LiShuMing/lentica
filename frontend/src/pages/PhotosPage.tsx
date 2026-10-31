import { FormEvent, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ImagePlus, Search } from "lucide-react";
import { api, AssetSearchResponse } from "../api/client";

type Mode = "smoke" | "gpu";
type Scope = "all" | "photo" | "screenshot" | "document";

export function PhotosPage() {
  const queryClient = useQueryClient();
  const [mode, setMode] = useState<Mode>("gpu");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<AssetSearchResponse | null>(null);
  const [scope, setScope] = useState<Scope>("all");
  const indexMutation = useMutation({
    mutationFn: () =>
      api.indexNextPhotoBatch(
        mode === "gpu"
          ? {
              limit: 100,
              max_indexed: 10,
              probe_limit: 1000,
              captioner_kind: "qwen-vl",
              vision_backend: "vllm",
              vision_timeout_seconds: 180,
              vision_fallback: "basic",
              embedder_kind: "clip",
              text_embedder_kind: "bge",
              device: "cuda",
              embedding_device: "cpu",
              unindexed_only: true
            }
          : { limit: 100, max_indexed: 5, probe_limit: 1000, captioner_kind: "basic", embedder_kind: "hash", device: "cpu", unindexed_only: true }
      ),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["tasks"] })
  });
  const searchMutation = useMutation({
    mutationFn: (text: string) =>
      api.searchAssets(
        text,
        scope === "all" ? ["photo", "screenshot", "document"] : [scope],
        mode === "gpu" ? "hybrid" : "hash"
      ),
    onSuccess: setResults
  });

  function submit(event: FormEvent) {
    event.preventDefault();
    if (query.trim()) searchMutation.mutate(query.trim());
  }

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <h1>本地检索</h1>
          <p>统一搜索照片、截图和文档，并查看可追溯来源。</p>
        </div>
        <div className="button-row">
          <select value={mode} onChange={(event) => setMode(event.target.value as Mode)}>
            <option value="gpu">GPU: Qwen-VL + CLIP</option>
            <option value="smoke">Smoke: basic + hash</option>
          </select>
          <button onClick={() => indexMutation.mutate()} disabled={indexMutation.isPending}>
            <ImagePlus size={18} />
            <span>{indexMutation.isPending ? "Queued" : "Queue batch"}</span>
          </button>
        </div>
      </header>
      {indexMutation.data ? <div className="status-line">{indexMutation.data.message}</div> : null}
      {indexMutation.error ? <div className="error">{indexMutation.error.message}</div> : null}
      <form className="ask-row" onSubmit={submit}>
        <select
          aria-label="检索范围"
          value={scope}
          disabled={searchMutation.isPending}
          onChange={(event) => {
            setScope(event.target.value as Scope);
            setResults(null);
            searchMutation.reset();
          }}
        >
          <option value="all">全部</option>
          <option value="photo">照片</option>
          <option value="screenshot">截图</option>
          <option value="document">文档</option>
        </select>
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="例如：海边日落、截图中的错误码、文档内容" />
        <button type="submit" disabled={searchMutation.isPending}>
          <Search size={18} />
          <span>{searchMutation.isPending ? "搜索中…" : "Search"}</span>
        </button>
      </form>
      {searchMutation.error ? <div className="error">{searchMutation.error.message}</div> : null}
      {results ? (
        <div className="photo-results">
          <div className="status-line">{results.answer}</div>
          {results.results.length === 0 ? <p className="muted">没有匹配当前范围的结果。</p> : null}
          <div className="photo-grid">
            {results.results.map((hit) =>
              hit.photo ? (
                <PhotoResult key={hit.file_id} hit={hit.photo} rank={hit.rank} />
              ) : (
                <DocumentResult key={hit.file_id} hit={hit} />
              )
            )}
          </div>
        </div>
      ) : (
        <div className="empty-state">先索引照片或扫描文档，再选择范围进行搜索。</div>
      )}
    </section>
  );
}

function topNamespaces(scores: Record<string, number>) {
  return Object.entries(scores)
    .filter(([, score]) => score !== 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);
}

function PhotoResult({ hit, rank }: { hit: AssetSearchResponse["hits"][number]; rank: number }) {
  return (
    <div className="photo-card">
      <div className="photo-thumb">
        {hit.thumb_path ? (
          <img src={thumbUrl(hit.thumb_path)} alt={hit.caption ?? hit.file_name} loading="lazy" />
        ) : (
          <span>No thumbnail</span>
        )}
      </div>
      <div className="photo-card-body">
        <div className="source-meta">
          <strong>{rank}. {hit.file_name}</strong>
          <span>
            相关度参考 {hit.confidence.toFixed(2)} · score {hit.score.toFixed(3)}
            {hit.exif_taken_at ? ` · ${hit.exif_taken_at.slice(0, 10)}` : ""}
          </span>
        </div>
        <div className="score-grid">
          <span>visual {hit.image_score.toFixed(3)}</span>
          <span>caption {hit.text_score.toFixed(3)}</span>
          <span>meta {hit.metadata_score.toFixed(3)}</span>
        </div>
        <div className="score-grid wide">
          {topNamespaces(hit.namespace_scores).map(([name, score]) => (
            <span key={name}>
              {name} {score.toFixed(3)}
            </span>
          ))}
        </div>
        <span className="match-reason">{hit.match_reason}</span>
        <p>{hit.caption ?? hit.file_name}</p>
        {hit.dense_caption && hit.dense_caption !== hit.caption ? (
          <p className="dense-caption">{hit.dense_caption.slice(0, 180)}</p>
        ) : null}
        {hit.ocr_text ? <p className="ocr-snippet">{hit.ocr_text.slice(0, 140)}</p> : null}
        {hit.matched_terms.length ? (
          <div className="chips">
            {hit.matched_terms.map((term) => (
              <span className="chip" key={term}>
                {term}
              </span>
            ))}
          </div>
        ) : null}
        <div className="chips">
          {hit.tags.map((tag) => (
            <span className="chip on" key={tag}>
              {tag}
            </span>
          ))}
        </div>
        <code>{hit.windows_path}</code>
      </div>
    </div>
  );
}

function DocumentResult({ hit }: { hit: AssetSearchResponse["results"][number] }) {
  return (
    <article className="source-card">
      <div className="source-meta">
        <strong>{hit.rank}. {hit.documents[0]?.windows_path ?? `文档 ${hit.file_id}`}</strong>
        <span>文档</span>
      </div>
      {hit.documents.map((source) => (
        <div key={source.chunk_id}>
          <span className="muted">{source.page_or_segment}</span>
          <p>{source.excerpt}</p>
        </div>
      ))}
    </article>
  );
}

function thumbUrl(path: string | null) {
  if (!path) return "";
  const name = path.split("/").pop();
  return name ? `/api/photos/thumb/${encodeURIComponent(name)}` : "";
}
