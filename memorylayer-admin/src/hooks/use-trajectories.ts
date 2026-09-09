// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useQuery } from "@tanstack/react-query";
import * as adminApi from "@/lib/admin-api";

export function useTrajectories(params?: {
  workspace_id?: string;
  limit?: number;
  offset?: number;
}) {
  return useQuery({
    queryKey: ["trajectories", params],
    queryFn: () => adminApi.listTrajectories(params),
    staleTime: 30_000,
  });
}

export function useTrajectory(id: string) {
  return useQuery({
    queryKey: ["trajectory", id],
    queryFn: () => adminApi.getTrajectory(id),
    enabled: !!id,
    staleTime: 60_000,
  });
}
