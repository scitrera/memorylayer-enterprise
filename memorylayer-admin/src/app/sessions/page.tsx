// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/shared/empty-state";
import { TimeAgo } from "@/components/shared/time-ago";
import { ConfirmDialog } from "@/components/shared/confirm-dialog";
import { useListSessions, useDeleteSession } from "@/hooks/use-sessions";
import { useWorkspaceContext } from "@/providers/workspace-provider";
import { Clock, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import Link from "next/link";
import type { Session } from "@scitrera/memorylayer-sdk";

function isExpired(session: Session): boolean {
  return new Date(session.expires_at) < new Date();
}

export default function SessionsPage() {
  const { activeWorkspace } = useWorkspaceContext();
  const showWorkspace = activeWorkspace === null;
  const [includeExpired, setIncludeExpired] = useState(false);
  const { data: sessions, isLoading, error } = useListSessions(includeExpired);
  const deleteSession = useDeleteSession();
  const [deleteTarget, setDeleteTarget] = useState<Session | null>(null);

  const handleDelete = () => {
    if (!deleteTarget) return;
    deleteSession.mutate(deleteTarget.id, {
      onSuccess: () => toast.success("Session deleted"),
      onError: () => toast.error("Failed to delete session"),
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Sessions</h1>
          <p className="text-muted-foreground">Active and recent working sessions.</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIncludeExpired((v) => !v)}
            className={`rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
              includeExpired
                ? "border-primary bg-primary text-primary-foreground"
                : "border-input bg-background hover:bg-accent"
            }`}
          >
            {includeExpired ? "Showing All" : "Active Only"}
          </button>
        </div>
      </div>

      {isLoading && (
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      )}

      {error && <p className="text-sm text-destructive">Failed to load sessions.</p>}

      {!isLoading && !error && sessions && sessions.length === 0 && (
        <EmptyState
          icon={Clock}
          title="No sessions"
          description="No active sessions found."
        />
      )}

      {!isLoading && !error && sessions && sessions.length > 0 && (
        <div className="overflow-x-auto rounded-lg border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Session ID</th>
                {showWorkspace && (
                  <th className="px-4 py-3 text-left font-medium text-muted-foreground">Workspace</th>
                )}
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Status</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Created</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Expires</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Actions</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((session) => {
                const expired = isExpired(session);
                return (
                  <tr key={session.id} className="border-b last:border-0 hover:bg-muted/30">
                    <td className="px-4 py-3">
                      <Link
                        href={`/sessions/${session.id}`}
                        className="font-mono text-xs hover:underline text-primary"
                      >
                        {session.id.slice(0, 12)}…
                      </Link>
                    </td>
                    {showWorkspace && (
                      <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                        {session.workspace_id}
                      </td>
                    )}
                    <td className="px-4 py-3">
                      <Badge variant={expired ? "secondary" : "default"}>
                        {expired ? "Expired" : "Active"}
                      </Badge>
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      <TimeAgo date={session.created_at} />
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      <TimeAgo date={session.expires_at} />
                    </td>
                    <td className="px-4 py-3">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setDeleteTarget(session)}
                      >
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(open) => !open && setDeleteTarget(null)}
        title="Delete session"
        description={`Delete session ${deleteTarget?.id.slice(0, 12)}…? This will remove the session and its working memory.`}
        confirmLabel="Delete"
        destructive
        onConfirm={handleDelete}
      />
    </div>
  );
}
