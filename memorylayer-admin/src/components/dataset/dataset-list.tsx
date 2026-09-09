"use client";

import { Badge } from "@/components/ui/badge";
import { TimeAgo } from "@/components/shared/time-ago";
import { formatNumber } from "@/lib/format";
import { useWorkspaceContext } from "@/providers/workspace-provider";
import type { DatasetInfo } from "@scitrera/memorylayer-sdk";
import Link from "next/link";

function statusVariant(status: string): "default" | "destructive" | "secondary" {
  if (status === "ready" || status === "completed") return "default";
  if (status === "failed") return "destructive";
  return "secondary";
}

interface DatasetListProps {
  datasets: DatasetInfo[];
}

export function DatasetList({ datasets }: DatasetListProps) {
  const { activeWorkspace } = useWorkspaceContext();
  const showWorkspace = activeWorkspace === null;

  return (
    <div className="overflow-x-auto rounded-lg border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b bg-muted/50">
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Name</th>
            {showWorkspace && (
              <th className="px-4 py-3 text-left font-medium text-muted-foreground">Workspace</th>
            )}
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Status</th>
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Rows</th>
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Columns</th>
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Format</th>
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Created</th>
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Actions</th>
          </tr>
        </thead>
        <tbody>
          {datasets.map((ds) => (
            <tr key={ds.id} className="border-b last:border-0 hover:bg-muted/30">
              <td className="px-4 py-3">
                <Link
                  href={`/datasets/${ds.id}`}
                  className="font-medium hover:underline truncate max-w-xs block"
                  title={ds.filename}
                >
                  {ds.name || ds.filename}
                </Link>
                <span className="text-xs text-muted-foreground font-mono">{ds.filename}</span>
              </td>
              {showWorkspace && (
                <td className="px-4 py-3">
                  <Link
                    href={`/workspaces/${(ds as unknown as { workspace_id?: string }).workspace_id}`}
                    className="text-xs text-primary hover:underline"
                  >
                    {(ds as unknown as { workspace_id?: string }).workspace_id ?? ""}
                  </Link>
                </td>
              )}
              <td className="px-4 py-3">
                <Badge variant={statusVariant(ds.status)}>{ds.status}</Badge>
              </td>
              <td className="px-4 py-3 text-muted-foreground">{formatNumber(ds.row_count)}</td>
              <td className="px-4 py-3 text-muted-foreground">{ds.column_count}</td>
              <td className="px-4 py-3">
                <Badge variant="secondary" className="text-xs uppercase">{ds.format}</Badge>
              </td>
              <td className="px-4 py-3 text-muted-foreground">
                <TimeAgo date={ds.created_at} />
              </td>
              <td className="px-4 py-3">
                <Link
                  href={`/datasets/${ds.id}`}
                  className="text-xs text-primary hover:underline"
                >
                  View
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
