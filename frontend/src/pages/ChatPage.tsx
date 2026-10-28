import { FormEvent, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Send } from "lucide-react";
import { api, ChatResponse } from "../api/client";

export function ChatPage() {
  const [question, setQuestion] = useState("");
  const [latest, setLatest] = useState<ChatResponse | null>(null);
  const mutation = useMutation({
    mutationFn: api.chat,
    onSuccess: setLatest
  });

  function submit(event: FormEvent) {
    event.preventDefault();
    if (question.trim()) mutation.mutate(question.trim());
  }

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <h1>Ask about your photos and documents</h1>
          <p>答案必须来自本地索引，并显示可追溯来源。</p>
        </div>
      </header>
      <form className="ask-row" onSubmit={submit}>
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="例如：我为 GPU 学习计划了哪些事情？"
        />
        <button type="submit" disabled={mutation.isPending}>
          <Send size={18} />
          <span>{mutation.isPending ? "Sending" : "Send"}</span>
        </button>
      </form>
      {mutation.error ? <div className="error">{mutation.error.message}</div> : null}
      {latest ? (
        <div className="answer-layout">
          <article className="answer">
            <div className={`confidence ${latest.confidence}`}>{latest.confidence}</div>
            <div className="answer-mode">{latest.answer_mode}</div>
            <p>{latest.answer}</p>
          </article>
          <section className="sources">
            <h2>Sources</h2>
            {latest.sources.length === 0 ? <p className="muted">No grounded sources found.</p> : null}
            {latest.sources.map((source) => (
              <div className="source-card" key={source.chunk_id}>
                <div className="source-meta">
                  <strong>{source.windows_path}</strong>
                  <span>{source.page_or_segment} · score {source.score.toFixed(3)}</span>
                </div>
                <p>{source.excerpt}</p>
              </div>
            ))}
            <h2>Retrieval trace</h2>
            <div className="trace">{latest.trace.join(" -> ")}</div>
          </section>
        </div>
      ) : (
        <div className="empty-state">Run a Library scan first, then ask in Chinese.</div>
      )}
    </section>
  );
}
