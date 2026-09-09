// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useConnection } from "@/providers/connection-provider";
import { useWorkspaceContext } from "@/providers/workspace-provider";
import { listAdminSessions } from "@/lib/admin-api";
import type { Session, SessionCreateOptions, CommitOptions, CommitResponse } from "@/types";

export function useListSessions(includeExpired = false) {
  const { client, isConnected } = useConnection();
  const { activeWorkspace } = useWorkspaceContext();

  return useQuery<Session[]>({
    queryKey: ["sessions", { includeExpired, workspace: activeWorkspace }],
    queryFn: async () => {
      if (activeWorkspace !== null) {
        // Per-workspace: use SDK client
        return client.listSessions({ includeExpired });
      }
      // Cross-workspace: use admin endpoint
      const result = await listAdminSessions({ include_expired: includeExpired, limit: 100 });
      return result.items as unknown as Session[];
    },
    enabled: isConnected,
    staleTime: 30_000,
  });
}

export function useSession(sessionId: string) {
  const { client } = useConnection();
  return useQuery({
    queryKey: ["sessions", sessionId],
    queryFn: () => client.getSession(sessionId),
    enabled: !!sessionId,
    retry: false,
  });
}

export function useWorkingMemory(sessionId: string) {
  const { client } = useConnection();
  return useQuery({
    queryKey: ["sessions", sessionId, "memory"],
    queryFn: () => client.getWorkingMemory(sessionId),
    enabled: !!sessionId,
  });
}

export function useCreateSession() {
  const { client } = useConnection();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (options: SessionCreateOptions) => {
      return client.createSession(options, false);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });
}

export function useTouchSession() {
  const { client } = useConnection();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ sessionId, ttlSeconds }: { sessionId: string; ttlSeconds?: number }) => {
      return client.touchSession(sessionId, ttlSeconds);
    },
    onSuccess: (data) => {
      queryClient.setQueryData(["sessions", data.id], data);
    },
  });
}

export function useSetWorkingMemory() {
  const { client } = useConnection();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ sessionId, key, value }: { sessionId: string; key: string; value: unknown }) => {
      await client.setWorkingMemory(sessionId, key, value);
      return { sessionId, key, value };
    },
    onSuccess: ({ sessionId }) => {
      queryClient.invalidateQueries({ queryKey: ["sessions", sessionId, "memory"] });
    },
  });
}

export function useCommitSession() {
  const { client } = useConnection();
  const queryClient = useQueryClient();
  return useMutation<CommitResponse, Error, { sessionId: string; options?: CommitOptions }>({
    mutationFn: ({ sessionId, options }) => client.commitSession(sessionId, options),
    onSuccess: (_, { sessionId }) => {
      queryClient.invalidateQueries({ queryKey: ["sessions", sessionId] });
    },
  });
}

export function useDeleteSession() {
  const { client } = useConnection();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (sessionId: string) => {
      await client.deleteSession(sessionId);
      return sessionId;
    },
    onSuccess: (sessionId) => {
      queryClient.removeQueries({ queryKey: ["sessions", sessionId] });
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });
}
