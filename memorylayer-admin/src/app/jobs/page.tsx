// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/shared/empty-state";
import { ConfirmDialog } from "@/components/shared/confirm-dialog";
import { TimeAgo } from "@/components/shared/time-ago";
import { useJobs, useCancelJob, type UnifiedJob } from "@/hooks/use-jobs";
import { useWorkspaceContext } from "@/providers/workspace-provider";
import { Layers, XCircle } from "lucide-react";
import { toast } from "sonner";

const STATUS_FILTERS = [
  { label: "All", value: undefined },
  { label: "Pending", value: "pending" },
  { label: "Running", value: "running" },
  { label: "Completed", value: "completed" },
  { label: "Failed", value: "failed" },
] as const;

function statusVariant(status: string): "default" | "destructive" | "secondary" {
  if (status === "completed") return "default";
  if (status === "failed") return "destructive";
  return "secondary";
}

function ProgressBar({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary transition-all"
          style={{ width: `${Math.min(100, value)}%` }}
        />
      </div>
      <span className="text-xs text-muted-foreground">{value}%</span>
    </div>
  );
}

export default function JobsPage() {
  const { activeWorkspace } = useWorkspaceContext();
  const showWorkspace = activeWorkspace === null;
  const [statusFilter, setStatusFilter] = useState<string | undefined>(undefined);
  const { data: jobs, isLoading, error } = useJobs(statusFilter);
  const cancelJob = useCancelJob();
  const [cancelTarget, setCancelTarget] = useState<UnifiedJob | null>(null);

  const handleCancel = () => {
    if (!cancelTarget) return;
    cancelJob.mutate(
      { jobId: cancelTarget.id, jobType: cancelTarget.job_type },
      {
        onSuccess: () => toast.success("Job cancelled"),
        onError: () => toast.error("Failed to cancel job"),
      }
    );
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Jobs</h1>
        <p className="text-muted-foreground">Document and dataset processing job queue.</p>
      </div>

      <div className="flex gap-2">
        {STATUS_FILTERS.map((f) => (
          <button
            key={String(f.value)}
            onClick={() => setStatusFilter(f.value)}
            className={`rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
              statusFilter === f.value
                ? "border-primary bg-primary text-primary-foreground"
                : "border-input bg-background hover:bg-accent"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {isLoading && (
        <div className="space-y-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      )}

      {error && <p className="text-sm text-destructive">Failed to load jobs.</p>}

      {!isLoading && !error && jobs && jobs.length === 0 && (
        <EmptyState
          icon={Layers}
          title="No jobs"
          description="No processing jobs found."
        />
      )}

      {!isLoading && !error && jobs && jobs.length > 0 && (
        <div className="overflow-x-auto rounded-lg border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">ID</th>
                {showWorkspace && (
                  <th className="px-4 py-3 text-left font-medium text-muted-foreground">Workspace</th>
                )}
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Type</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Status</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Progress</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Memories</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Created</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Actions</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr key={job.id} className="border-b last:border-0 hover:bg-muted/30">
                  <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                    {job.id.slice(0, 12)}…
                  </td>
                  {showWorkspace && (
                    <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                      {job.workspace_id}
                    </td>
                  )}
                  <td className="px-4 py-3">
                    <Badge variant="secondary" className="text-xs capitalize">
                      {job.job_type}
                    </Badge>
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant={statusVariant(job.status)}>{job.status}</Badge>
                  </td>
                  <td className="px-4 py-3">
                    <ProgressBar value={job.progress_percent} />
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {job.total_memories_created}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    <TimeAgo date={job.created_at} />
                  </td>
                  <td className="px-4 py-3">
                    {(job.status === "running" || job.status === "pending") && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setCancelTarget(job)}
                        title="Cancel job"
                      >
                        <XCircle className="h-4 w-4 text-destructive" />
                      </Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <ConfirmDialog
        open={!!cancelTarget}
        onOpenChange={(open) => !open && setCancelTarget(null)}
        title="Cancel job"
        description={`Cancel job ${cancelTarget?.id.slice(0, 12)}…? The job will be stopped and cannot be resumed.`}
        confirmLabel="Cancel Job"
        destructive
        onConfirm={handleCancel}
      />
    </div>
  );
}
