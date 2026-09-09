"use client";

import { Badge } from "@/components/ui/badge";
import { TimeAgo } from "@/components/shared/time-ago";
import { formatNumber } from "@/lib/format";
import { useWorkspaceContext } from "@/providers/workspace-provider";
import type { DocumentInfo } from "@scitrera/memorylayer-sdk";
import Link from "next/link";

function statusVariant(status: string): "default" | "destructive" | "secondary" {
  if (status === "ready" || status === "completed") return "default";
  if (status === "failed") return "destructive";
  return "secondary";
}

interface DocumentListProps {
  documents: DocumentInfo[];
}

export function DocumentList({ documents }: DocumentListProps) {
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
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Pages</th>
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Size</th>
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Created</th>
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Actions</th>
          </tr>
        </thead>
        <tbody>
          {documents.map((doc) => (
            <tr key={doc.id} className="border-b last:border-0 hover:bg-muted/30">
              <td className="px-4 py-3">
                <Link
                  href={`/documents/${doc.id}`}
                  className="font-medium hover:underline truncate max-w-xs block"
                  title={doc.filename}
                >
                  {doc.filename}
                </Link>
                <span className="text-xs text-muted-foreground">{doc.document_type}</span>
              </td>
              {showWorkspace && (
                <td className="px-4 py-3">
                  <Link
                    href={`/workspaces/${(doc as unknown as { workspace_id?: string }).workspace_id}`}
                    className="text-xs text-primary hover:underline"
                  >
                    {(doc as unknown as { workspace_id?: string }).workspace_id ?? ""}
                  </Link>
                </td>
              )}
              <td className="px-4 py-3">
                <Badge variant={statusVariant(doc.status)}>{doc.status}</Badge>
              </td>
              <td className="px-4 py-3 text-muted-foreground">{doc.page_count}</td>
              <td className="px-4 py-3 text-muted-foreground">
                {formatNumber(doc.size_bytes)}B
              </td>
              <td className="px-4 py-3 text-muted-foreground">
                <TimeAgo date={doc.created_at} />
              </td>
              <td className="px-4 py-3">
                <Link
                  href={`/documents/${doc.id}`}
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
