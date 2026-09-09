// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useQuery } from "@tanstack/react-query";
import * as adminApi from "@/lib/admin-api";

export function useAuditEvents(params?: {
  event_type?: string;
  action?: string;
  workspace_id?: string;
  user_id?: string;
  from?: string;
  to?: string;
  limit?: number;
  offset?: number;
}) {
  return useQuery({
    queryKey: ["audit-events", params],
    queryFn: () => adminApi.listAuditEvents(params),
    staleTime: 30_000,
  });
}

export function useAuditSummary(params?: { from?: string; to?: string }) {
  return useQuery({
    queryKey: ["audit-summary", params],
    queryFn: () => adminApi.getAuditSummary(params),
    staleTime: 60_000,
  });
}
