import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileText, Play, RotateCcw, X } from "lucide-react";
import { api, TaskInfo } from "../api/client";

export function JobsPage() {
  const queryClient = useQueryClient();
  const previousStates = useRef<Map<number, string>>(new Map());
  const [changedIds, setChangedIds] = useState<Set<number>>(new Set());
  const tasks = useQuery({ queryKey: ["tasks"], queryFn: api.tasks, refetchInterval: 2500 });
  const queue = useQuery({ queryKey: ["queue-status"], queryFn: api.queueStatus, refetchInterval: 2500 });
  const vision = useQuery({ queryKey: ["vision-status"], queryFn: api.visionStatus, refetchInterval: 5000 });
  const refreshTasks = () => queryClient.invalidateQueries({ queryKey: ["tasks"] });
  const refreshQueue = () => queryClient.invalidateQueries({ queryKey: ["queue-status"] });
  const updateQueue = useMutation({
    mutationFn: api.updateQueueSettings,
    onSuccess: () => {
      refreshQueue();
      refreshTasks();
    }
  });
  const enqueueNext = useMutation({
    mutationFn: () =>
      api.indexNextPhotoBatch(
        queue.data?.settings.worker_mode === "smoke"
          ? {
              limit: queue.data.settings.photo_limit,
              max_indexed: queue.data.settings.photo_max_indexed,
              probe_limit: queue.data.settings.photo_probe_limit,
              captioner_kind: "basic",
              embedder_kind: "hash",
              device: "cpu",
              unindexed_only: true
            }
          : {
              limit: queue.data?.settings.photo_limit ?? 100,
              max_indexed: queue.data?.settings.photo_max_indexed ?? 10,
              probe_limit: queue.data?.settings.photo_probe_limit ?? 1000,
              captioner_kind: "qwen-vl",
              vision_backend: queue.data?.settings.vision_backend ?? "vllm",
              vision_endpoint: queue.data?.settings.vision_endpoint,
              vision_model: queue.data?.settings.vision_model ?? "qwen-vl-photo-context",
              vision_timeout_seconds: queue.data?.settings.vision_timeout_seconds ?? 180,
              vision_fallback: queue.data?.settings.vision_fallback ?? "transformers",
              embedder_kind: "clip",
              text_embedder_kind: "bge",
              device: "cuda",
              embedding_device: queue.data?.settings.vision_backend === "vllm" ? "cpu" : "cuda",
              unindexed_only: true
            }
      ),
    onSuccess: refreshTasks
  });
  const enqueueDocs = useMutation({ mutationFn: api.indexDocuments, onSuccess: refreshTasks });
  const cancelTask = useMutation({ mutationFn: api.cancelTask, onSuccess: refreshTasks });
  const retryTask = useMutation({ mutationFn: api.retryTask, onSuccess: refreshTasks });
  const taskRows = useMemo(() => tasks.data ?? [], [tasks.data]);

  useEffect(() => {
    const nextChanged = new Set<number>();
    for (const task of taskRows) {
      const previous = previousStates.current.get(task.id);
      if (previous && previous !== task.state) {
        nextChanged.add(task.id);
      }
      previousStates.current.set(task.id, task.state);
    }
    if (nextChanged.size > 0) {
      setChangedIds(nextChanged);
      const timer = window.setTimeout(() => setChangedIds(new Set()), 2200);
      return () => window.clearTimeout(timer);
    }
  }, [taskRows]);

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <h1>Jobs</h1>
          <p>本地 worker 队列、运行状态、错误和最近输出。</p>
        </div>
        <div className="button-row">
          <button onClick={() => enqueueNext.mutate()} disabled={enqueueNext.isPending}>
            <Play size={18} />
            <span>Queue next photos</span>
          </button>
          <button onClick={() => enqueueDocs.mutate()} disabled={enqueueDocs.isPending}>
            <FileText size={18} />
            <span>Queue documents</span>
          </button>
        </div>
      </header>
      {queue.data ? (
        <div className="queue-panel">
          <div className="setting-row compact">
            <span>Auto queue</span>
            <label className="switch-row">
              <input
                type="checkbox"
                checked={queue.data.settings.enabled}
                onChange={(event) => updateQueue.mutate({ enabled: event.target.checked })}
              />
              <strong>{queue.data.settings.enabled ? "on" : "off"}</strong>
            </label>
          </div>
          <div className="setting-row compact">
            <span>Photos / Documents</span>
            <div className="button-row">
              <label className="switch-row">
                <input
                  type="checkbox"
                  checked={queue.data.settings.auto_enqueue_photos}
                  onChange={(event) => updateQueue.mutate({ auto_enqueue_photos: event.target.checked })}
                />
                <strong>photos</strong>
              </label>
              <label className="switch-row">
                <input
                  type="checkbox"
                  checked={queue.data.settings.auto_enqueue_documents}
                  onChange={(event) => updateQueue.mutate({ auto_enqueue_documents: event.target.checked })}
                />
                <strong>documents</strong>
              </label>
            </div>
          </div>
          <div className="setting-row compact">
            <span>Mode / concurrency</span>
            <div className="button-row">
              <select
                value={queue.data.settings.worker_mode}
                onChange={(event) => updateQueue.mutate({ worker_mode: event.target.value })}
              >
                <option value="gpu">GPU: Qwen-VL + CLIP</option>
                <option value="smoke">Smoke: basic + hash</option>
              </select>
              <label className="switch-row">
                <input
                  type="checkbox"
                  checked={queue.data.settings.auto_adjust_concurrency}
                  onChange={(event) => updateQueue.mutate({ auto_adjust_concurrency: event.target.checked })}
                />
                <strong>auto</strong>
              </label>
              <input
                className="number-input"
                type="number"
                min={0}
                max={4}
                value={queue.data.settings.max_concurrency}
                onChange={(event) => updateQueue.mutate({ max_concurrency: Number(event.target.value) })}
              />
            </div>
          </div>
          <div className="setting-row compact">
            <span>Vision service</span>
            <div className="button-row">
              <select
                value={queue.data.settings.vision_backend}
                onChange={(event) => updateQueue.mutate({ vision_backend: event.target.value })}
              >
                <option value="vllm">vLLM service</option>
                <option value="transformers">Transformers fallback</option>
                <option value="basic">Basic metadata</option>
              </select>
              <select
                value={queue.data.settings.vision_fallback}
                onChange={(event) => updateQueue.mutate({ vision_fallback: event.target.value })}
              >
                <option value="transformers">fallback transformers</option>
                <option value="basic">fallback basic</option>
                <option value="none">no fallback</option>
              </select>
              <input
                className="endpoint-input"
                value={queue.data.settings.vision_endpoint}
                onChange={(event) => updateQueue.mutate({ vision_endpoint: event.target.value })}
              />
            </div>
          </div>
          <div className="queue-metrics">
            <span>Desired workers: {queue.data.desired_concurrency}</span>
            <span>Running workers: {queue.data.workers.length}</span>
            <span>
              GPU: {String(queue.data.gpu.available)}
              {"memory_free_mb" in queue.data.gpu ? ` · free ${queue.data.gpu.memory_free_mb} MB` : ""}
              {"utilization_percent" in queue.data.gpu ? ` · util ${queue.data.gpu.utilization_percent}%` : ""}
            </span>
          </div>
          {vision.data ? (
            <div className="queue-metrics">
              <span>Vision: {vision.data.backend}</span>
              <span>{vision.data.healthy ? "healthy" : "unavailable"}</span>
              <span>{vision.data.model}</span>
              {vision.data.error ? <span className="error-inline">{vision.data.error}</span> : null}
            </div>
          ) : null}
          <div className="queue-summary">
            <QueueSummary title="Photos" counts={queue.data.task_summary.index_photos} />
            <QueueSummary title="Documents" counts={queue.data.task_summary.index_documents} />
            <QueueSummary title="Faces" counts={queue.data.task_summary.index_faces} />
          </div>
          {queue.data.last_error ? <div className="error-inline">{queue.data.last_error}</div> : null}
        </div>
      ) : null}
      {enqueueNext.data ? <div className="status-line">{enqueueNext.data.message}</div> : null}
      {enqueueNext.error ? <div className="error">{enqueueNext.error.message}</div> : null}
      {enqueueDocs.error ? <div className="error">{enqueueDocs.error.message}</div> : null}
      <div className="table-block">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Stage</th>
              <th>State</th>
              <th>Started</th>
              <th>Finished</th>
              <th>Duration</th>
              <th>Workload</th>
              <th>Throughput</th>
              <th>Attempts</th>
              <th>Worker</th>
              <th>Action</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {taskRows.map((task) => (
              <tr key={task.id} className={changedIds.has(task.id) ? "state-changed" : ""}>
                <td>{task.id}</td>
                <td>{task.pipeline_stage}</td>
                <td>
                  <span className={`state-pill ${task.state}`}>{task.state}</span>
                </td>
                <td>{formatDate(task.latest_run?.started_at)}</td>
                <td>{formatDate(task.latest_run?.finished_at)}</td>
                <td>{formatDuration(task.latest_run?.duration_seconds)}</td>
                <td>{formatLoad(task)}</td>
                <td>{formatThroughput(task)}</td>
                <td>{task.attempt_count}</td>
                <td className="path-cell">{task.worker_id ?? "-"}</td>
                <td>
                  <div className="row-actions">
                    {task.state === "queued" ? (
                      <button className="icon-button" title="Cancel task" onClick={() => cancelTask.mutate(task.id)}>
                        <X size={16} />
                      </button>
                    ) : null}
                    {task.state === "failed" || task.state === "cancelled" ? (
                      <button className="icon-button" title="Retry task" onClick={() => retryTask.mutate(task.id)}>
                        <RotateCcw size={16} />
                      </button>
                    ) : null}
                  </div>
                </td>
                <td className="path-cell">
                  {task.error ? <div className="error-inline">{task.error}</div> : null}
                  <code>{JSON.stringify(task.latest_run?.stats ?? task.payload)}</code>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function QueueSummary({ title, counts }: { title: string; counts?: Record<string, number> }) {
  const safeCounts = counts ?? {};
  return (
    <div className="queue-summary-item">
      <strong>{title}</strong>
      <span>queued {safeCounts.queued ?? 0}</span>
      <span>running {safeCounts.running ?? 0}</span>
      <span>done {safeCounts.succeeded ?? 0}</span>
      <span>failed {safeCounts.failed ?? 0}</span>
    </div>
  );
}

function formatDate(value?: string | null) {
  if (!value) return "-";
  return new Date(value).toLocaleTimeString();
}

function formatDuration(value?: number | null) {
  if (value == null) return "-";
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}m ${seconds}s`;
}

function formatLoad(task: TaskInfo) {
  const load = task.latest_run?.computational_load ?? {};
  if (task.pipeline_stage === "index_photos") {
    return `${load.images_captioned ?? 0} imgs · ${load.total_vectors_written ?? 0} vec`;
  }
  if (task.pipeline_stage === "index_documents") {
    return `${load.documents_indexed ?? 0} docs · ${load.documents_skipped ?? 0} skip`;
  }
  if (task.pipeline_stage === "index_faces") {
    return `${load.faces_detected ?? 0} faces · ${load.clusters_created ?? 0} clusters`;
  }
  return "-";
}

function formatThroughput(task: TaskInfo) {
  const throughput = task.latest_run?.throughput ?? {};
  if (task.pipeline_stage === "index_photos") {
    const value = throughput.indexed_per_second;
    return typeof value === "number" ? `${value.toFixed(2)} img/s` : "-";
  }
  if (task.pipeline_stage === "index_documents") {
    const value = throughput.total_files_per_second;
    return typeof value === "number" ? `${value.toFixed(2)} file/s` : "-";
  }
  if (task.pipeline_stage === "index_faces") {
    const value = throughput.indexed_per_second;
    return typeof value === "number" ? `${value.toFixed(2)} photo/s` : "-";
  }
  return "-";
}
