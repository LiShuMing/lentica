import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { api } from "../api/client";

export function LibraryPage() {
  const queryClient = useQueryClient();
  const sources = useQuery({ queryKey: ["sources"], queryFn: api.sources });
  const files = useQuery({ queryKey: ["files"], queryFn: api.files });
  const scan = useMutation({
    mutationFn: api.scan,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["files"] });
    }
  });

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <h1>Library</h1>
          <p>默认文档源会被只读扫描，索引数据保存在 WSL 的 data 目录。</p>
        </div>
        <button onClick={() => scan.mutate()} disabled={scan.isPending}>
          <RefreshCw size={18} />
          <span>{scan.isPending ? "Scanning" : "Run incremental scan"}</span>
        </button>
      </header>
      {scan.data ? (
        <div className="status-line">
          Job #{scan.data.id}: {scan.data.status}, indexed {scan.data.indexed_files}, skipped {scan.data.skipped_files},
          failed {scan.data.failed_files}
        </div>
      ) : null}
      {scan.error ? <div className="error">{scan.error.message}</div> : null}
      <div className="table-block">
        <h2>Sources</h2>
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Kind</th>
              <th>Files</th>
              <th>Last scan</th>
              <th>Root</th>
            </tr>
          </thead>
          <tbody>
            {(sources.data ?? []).map((source) => (
              <tr key={source.id}>
                <td>{source.name}</td>
                <td>{source.kind}</td>
                <td>{source.file_count}</td>
                <td>{source.last_scan_at ? new Date(source.last_scan_at).toLocaleString() : "-"}</td>
                <td className="path-cell">{source.root_path}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-block">
        <h2>Indexed files</h2>
        <table>
          <thead>
            <tr>
              <th>Title</th>
              <th>Type</th>
              <th>Status</th>
              <th>Chunks</th>
              <th>Windows path</th>
            </tr>
          </thead>
          <tbody>
            {(files.data ?? []).map((file) => (
              <tr key={file.id}>
                <td>{file.title}</td>
                <td>{file.file_type}</td>
                <td>{file.ingestion_status}</td>
                <td>{file.chunk_count}</td>
                <td className="path-cell">{file.windows_path}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
