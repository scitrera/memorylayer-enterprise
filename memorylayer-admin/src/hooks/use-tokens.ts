// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { listTokens, createToken, revokeToken, deleteToken } from "@/lib/admin-api";
import type { TokenCreateRequest } from "@/types/admin";

export function useTokens(includeRevoked = false) {
  return useQuery({
    queryKey: ["tokens", { includeRevoked }],
    queryFn: () => listTokens(includeRevoked),
    staleTime: 30_000,
  });
}

export function useCreateToken() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: TokenCreateRequest) => createToken(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tokens"] });
    },
  });
}

export function useRevokeToken() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (tokenId: string) => revokeToken(tokenId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tokens"] });
    },
  });
}

export function useDeleteToken() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (tokenId: string) => deleteToken(tokenId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tokens"] });
    },
  });
}
