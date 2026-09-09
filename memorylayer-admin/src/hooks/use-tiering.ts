"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import * as adminApi from "@/lib/admin-api";
import type { TieringConfig } from "@/types/admin";

export function useTieringStats() {
  return useQuery({
    queryKey: ["tiering-stats"],
    queryFn: () => adminApi.getTieringStats(),
    staleTime: 30_000,
  });
}

export function useTieringConfig() {
  return useQuery({
    queryKey: ["tiering-config"],
    queryFn: () => adminApi.getTieringConfig(),
    staleTime: 60_000,
  });
}

export function useUpdateTieringConfig() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (config: Partial<TieringConfig>) => adminApi.updateTieringConfig(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tiering-config"] });
    },
  });
}

export function useArchiveMemories() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (params: {
      workspace_id?: string;
      older_than_days?: number;
      importance_below?: number;
    }) => adminApi.archiveMemories(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tiering-stats"] });
    },
  });
}

export function useRestoreMemories() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (params: { workspace_id?: string; memory_ids?: string[] }) =>
      adminApi.restoreMemories(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tiering-stats"] });
    },
  });
}

export function useHealthDependencies() {
  return useQuery({
    queryKey: ["health-dependencies"],
    queryFn: () => adminApi.getHealthDependencies(),
    staleTime: 15_000,
    refetchInterval: 30_000,
  });
}
