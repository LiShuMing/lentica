import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

export function ModelsPage() {
  const models = useQuery({ queryKey: ["models"], queryFn: api.models, retry: false });

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <h1>Models</h1>
          <p>显示本地模型配置与依赖检测；服务健康请在 Jobs 查看。</p>
        </div>
      </header>
      {models.error ? <div className="error">{models.error.message}</div> : null}
      <div className="model-grid">
        {(models.data ?? []).map((model) => (
          <div className="model-card" key={model.id}>
            <strong>{model.id}</strong>
            <span>{model.role} · {model.runtime_type} · {model.status}</span>
            <span>Device: {model.device}{model.dtype ? ` · ${model.dtype}` : ""}</span>
            {model.model_path ? <span className="path-cell">{model.model_path}</span> : null}
            <div className="chips">
              <span className={model.enabled ? "chip on" : "chip"}>{model.enabled ? "enabled" : "disabled"}</span>
              {Object.entries(model.capabilities).map(([capability, enabled]) => (
                <span className={enabled ? "chip on" : "chip"} key={capability}>
                  {capability}
                </span>
              ))}
            </div>
            <p>{model.notes}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
