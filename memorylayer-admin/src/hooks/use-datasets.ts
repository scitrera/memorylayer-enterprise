"use client";

import { useQuery } from "@tanstack/react-query";
import { useConnection } from "@/providers/connection-provider";
import { useWorkspaceContext } from "@/providers/workspace-provider";
import { listAdminDatasets } from "@/lib/admin-api";
import type { DatasetInfo, DatasetJobInfo } from "@scitrera/memorylayer-sdk";

export function useDatasets(status?: string) {
  const { client, isConnected } = useConnection();
  const { activeWorkspace } = useWorkspaceContext();

  return useQuery<DatasetInfo[]>({
    queryKey: ["datasets", { status, workspace: activeWorkspace }],
    queryFn: async () => {
      if (activeWorkspace !== null) {
        // Per-workspace: use SDK client
        const result = await client.listDatasets({ status, limit: 100 });
        return result.datasets;
      }
      // Cross-workspace: use admin endpoint
      const result = await listAdminDatasets({ status, limit: 100 });
      return result.items as unknown as DatasetInfo[];
    },
    enabled: isConnected,
    staleTime: 30_000,
  });
}

export function useDataset(id: string) {
  const { client } = useConnection();
  return useQuery<DatasetInfo>({
    queryKey: ["datasets", id],
    queryFn: () => client.getDataset(id),
    enabled: !!id,
    retry: false,
  });
}

export function useDatasetJobs(datasetId: string) {
  const { client } = useConnection();
  return useQuery<DatasetJobInfo[]>({
    queryKey: ["dataset-jobs", datasetId],
    queryFn: async () => {
      const result = await client.listDatasetJobs({ limit: 50 });
      return result.jobs.filter((j) => j.dataset_ids.includes(datasetId));
    },
    enabled: !!datasetId,
    staleTime: 15_000,
  });
}
