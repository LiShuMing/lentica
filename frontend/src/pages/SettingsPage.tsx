import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

export function SettingsPage() {
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settings });
  const data = settings.data;

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <h1>Settings</h1>
          <p>显示配置是否存在，不显示任何密钥原文。</p>
        </div>
      </header>
      {data ? (
        <div className="settings-list">
          <Setting label="~/.env present" value={data.env_file_present ? "yes" : "no"} />
          <Setting label="Runtime mode" value={data.runtime_mode} />
          <Setting label="External API enabled" value={data.external_api_enabled ? "yes" : "no"} />
          <Setting label="Legacy LLM env detected" value={data.legacy_llm_env_present ? "yes, ignored" : "no"} />
          <Setting label="Documents source" value={`${data.documents_source} (${data.documents_source_exists ? "exists" : "missing"})`} />
          <Setting label="Photos source" value={`${data.photos_source} (${data.photos_source_exists ? "exists" : "missing"})`} />
          <Setting label="SQLite database" value={data.database_path} />
        </div>
      ) : (
        <div className="empty-state">Loading settings...</div>
      )}
    </section>
  );
}

function Setting({ label, value }: { label: string; value: string }) {
  return (
    <div className="setting-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
