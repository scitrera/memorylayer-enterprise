"use client";

import { useQuery } from "@tanstack/react-query";
import * as adminApi from "@/lib/admin-api";

export function useEntities(params?: {
  workspace_id?: string;
  query?: string;
  limit?: number;
  offset?: number;
}) {
  return useQuery({
    queryKey: ["entities", params],
    queryFn: () => adminApi.listEntities(params),
    staleTime: 30_000,
  });
}

export function useEntityCard(id: string) {
  return useQuery({
    queryKey: ["entity-card", id],
    queryFn: () => adminApi.getEntityCard(id),
    enabled: !!id,
    staleTime: 60_000,
  });
}

export function useEntityInsights(id: string) {
  return useQuery({
    queryKey: ["entity-insights", id],
    queryFn: () => adminApi.getEntityInsights(id),
    enabled: !!id,
    staleTime: 60_000,
  });
}
