// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useQuery } from "@tanstack/react-query";
import { useConnection } from "@/providers/connection-provider";
import { useWorkspaceContext } from "@/providers/workspace-provider";
import { getAdminStats } from "@/lib/admin-api";
import type { AdminStats } from "@/types/admin";

export function useAdminStats() {
  const { client, isConnected } = useConnection();
  const { activeWorkspace } = useWorkspaceContext();

  return useQuery<AdminStats>({
    queryKey: ["admin-stats", activeWorkspace],
    queryFn: async () => {
      if (activeWorkspace === null) {
        // Cross-workspace: use admin endpoint
        const stats = await getAdminStats();
        return {
          workspace_count: stats.workspace_count,
          memory_count: stats.memory_count,
          session_count: stats.session_count,
          token_count: stats.token_count,
          document_count: stats.document_count,
          dataset_count: stats.dataset_count,
        };
      }
      // Per-workspace: use SDK client (original approach)
      const [workspaces, sessions] = await Promise.all([
        client.listWorkspaces().catch(() => []),
        client.listSessions().catch(() => []),
      ]);
      return {
        workspace_count: workspaces.length,
        memory_count: 0,
        session_count: sessions.length,
        token_count: 0,
        document_count: 0,
        dataset_count: 0,
      };
    },
    enabled: isConnected,
    staleTime: 30_000,
  });
}
