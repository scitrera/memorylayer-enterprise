"use client";

import Link from "next/link";
import { Activity } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/shared/empty-state";
import { TimeAgo } from "@/components/shared/time-ago";
import type { Trajectory } from "@/types/admin";

interface TrajectoryListProps {
  trajectories: Trajectory[];
  isLoading: boolean;
}

export function TrajectoryList({ trajectories, isLoading }: TrajectoryListProps) {
  if (isLoading) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} className="h-14 w-full" />
        ))}
      </div>
    );
  }

  if (trajectories.length === 0) {
    return (
      <EmptyState
        icon={Activity}
        title="No trajectories"
        description="No search trajectories have been recorded yet."
      />
    );
  }

  return (
    <div className="rounded-lg border overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-slate-50 border-b">
          <tr>
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Query</th>
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Results</th>
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Latency</th>
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Workspace</th>
            <th className="px-4 py-3 text-left font-medium text-muted-foreground">Created</th>
          </tr>
        </thead>
        <tbody>
          {trajectories.map((traj) => (
            <tr key={traj.id} className="border-b hover:bg-slate-50 transition-colors">
              <td className="px-4 py-3">
                <Link
                  href={`/trajectories/${traj.id}`}
                  className="font-medium text-primary hover:underline line-clamp-1 max-w-xs block"
                >
                  {traj.query}
                </Link>
              </td>
              <td className="px-4 py-3">
                <Badge variant="secondary">{traj.results_count}</Badge>
              </td>
              <td className="px-4 py-3 text-muted-foreground">
                {traj.latency_ms}ms
              </td>
              <td className="px-4 py-3 text-muted-foreground font-mono text-xs">
                {traj.workspace_id}
              </td>
              <td className="px-4 py-3 text-muted-foreground">
                <TimeAgo date={traj.created_at} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
