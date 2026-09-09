// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useConnection } from "@/providers/connection-provider";
import { useWorkspaceContext } from "@/providers/workspace-provider";
import { listAdminJobs } from "@/lib/admin-api";
import type { JobInfo, DatasetJobInfo } from "@scitrera/memorylayer-sdk";

export type UnifiedJob =
  | (JobInfo & { job_type: "document" })
  | (DatasetJobInfo & { job_type: "dataset" });

export function useJobs(status?: string) {
  const { client, isConnected } = useConnection();
  const { activeWorkspace } = useWorkspaceContext();

  return useQuery<UnifiedJob[]>({
    queryKey: ["jobs", { status, workspace: activeWorkspace }],
    queryFn: async () => {
      if (activeWorkspace !== null) {
        // Per-workspace: use SDK client
        const [docResult, datasetResult] = await Promise.all([
          client.listJobs({ status, limit: 100 }).catch(() => ({ jobs: [] as JobInfo[] })),
          client.listDatasetJobs({ status, limit: 100 }).catch(() => ({ jobs: [] as DatasetJobInfo[] })),
        ]);
        const docJobs: UnifiedJob[] = docResult.jobs.map((j) => ({ ...j, job_type: "document" as const }));
        const datasetJobs: UnifiedJob[] = datasetResult.jobs.map((j) => ({ ...j, job_type: "dataset" as const }));
        const all = [...docJobs, ...datasetJobs];
        all.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
        return all;
      }
      // Cross-workspace: use admin endpoint
      const result = await listAdminJobs({ status, limit: 100 });
      return result.items as unknown as UnifiedJob[];
    },
    enabled: isConnected,
    staleTime: 15_000,
    refetchInterval: 10_000,
  });
}

export function useCancelJob() {
  const { client } = useConnection();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ jobId, jobType }: { jobId: string; jobType: "document" | "dataset" }) => {
      if (jobType === "document") {
        await client.cancelJob(jobId);
      } else {
        await client.cancelDatasetJob(jobId);
      }
      return { jobId, jobType };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}
