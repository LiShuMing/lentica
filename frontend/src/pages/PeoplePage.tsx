import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, UserRound } from "lucide-react";
import { api, PersonCluster } from "../api/client";

export function PeoplePage() {
  const queryClient = useQueryClient();
  const people = useQuery({ queryKey: ["people"], queryFn: api.people, refetchInterval: 5000 });
  const indexFaces = useMutation({
    mutationFn: () => api.indexPeople({ limit: 500, force: false }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["tasks"] })
  });
  const updatePerson = useMutation({
    mutationFn: ({
      id,
      display_name,
      aliases,
      ignored
    }: {
      id: number;
      display_name?: string | null;
      aliases?: string[] | null;
      ignored?: boolean | null;
    }) => api.updatePerson(id, { display_name, aliases, ignored }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["people"] })
  });

  const clusters = people.data ?? [];
  const visible = clusters.filter((cluster) => !cluster.ignored);
  const ignored = clusters.filter((cluster) => cluster.ignored);

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <h1>People</h1>
          <p>本地人脸 embedding 聚类；这里只给 cluster 命名，不做自动身份识别。</p>
        </div>
        <button onClick={() => indexFaces.mutate()} disabled={indexFaces.isPending}>
          <RefreshCw size={18} />
          <span>Index faces</span>
        </button>
      </header>
      {indexFaces.data ? <div className="status-line">Queued face indexing task #{indexFaces.data.id}</div> : null}
      {people.error ? <div className="error">{people.error.message}</div> : null}
      {!clusters.length ? (
        <div className="empty-state">还没有人物聚类。先运行照片索引，或点击 Index faces 补跑已索引照片。</div>
      ) : null}
      <div className="people-grid">
        {visible.map((cluster) => (
          <PersonCard key={cluster.id} cluster={cluster} updatePerson={updatePerson.mutate} />
        ))}
      </div>
      {ignored.length ? (
        <>
          <h2>Ignored</h2>
          <div className="people-grid compact">
            {ignored.map((cluster) => (
              <PersonCard key={cluster.id} cluster={cluster} updatePerson={updatePerson.mutate} />
            ))}
          </div>
        </>
      ) : null}
    </section>
  );
}

function PersonCard({
  cluster,
  updatePerson
}: {
  cluster: PersonCluster;
  updatePerson: (payload: { id: number; display_name?: string | null; aliases?: string[] | null; ignored?: boolean | null }) => void;
}) {
  const [name, setName] = useState(cluster.display_name ?? "");
  const [aliases, setAliases] = useState((cluster.aliases ?? []).join(", "));
  const parsedAliases = aliases
    .split(/[,，\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
  return (
    <div className={cluster.ignored ? "person-card ignored" : "person-card"}>
      <div className="person-samples">
        {cluster.samples.length ? (
          cluster.samples.slice(0, 6).map((sample) => (
            <img key={sample.face_id} src={sample.thumb_url} alt={cluster.display_name ?? cluster.label} loading="lazy" />
          ))
        ) : (
          <div className="person-placeholder">
            <UserRound size={28} />
          </div>
        )}
      </div>
      <div className="person-body">
        <div className="source-meta">
          <strong>{cluster.display_name || cluster.label}</strong>
          <span>
            {cluster.face_count} faces · {cluster.photo_count} photos
            {cluster.confidence != null ? ` · sim ${cluster.confidence.toFixed(2)}` : ""}
          </span>
        </div>
        <div className="person-edit">
          <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Display name" />
          <input value={aliases} onChange={(event) => setAliases(event.target.value)} placeholder="Aliases, comma separated" />
          <button onClick={() => updatePerson({ id: cluster.id, display_name: name, aliases: parsedAliases })}>Save</button>
          <button className="secondary-button" onClick={() => updatePerson({ id: cluster.id, ignored: !cluster.ignored })}>
            {cluster.ignored ? "Restore" : "Ignore"}
          </button>
        </div>
      </div>
    </div>
  );
}
