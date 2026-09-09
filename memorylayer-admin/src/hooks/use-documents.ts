"use client";

import { useQuery } from "@tanstack/react-query";
import { useConnection } from "@/providers/connection-provider";
import { useWorkspaceContext } from "@/providers/workspace-provider";
import { listAdminDocuments } from "@/lib/admin-api";
import type { DocumentInfo, JobInfo } from "@scitrera/memorylayer-sdk";

export function useDocuments(status?: string) {
  const { client, isConnected } = useConnection();
  const { activeWorkspace } = useWorkspaceContext();

  return useQuery<DocumentInfo[]>({
    queryKey: ["documents", { status, workspace: activeWorkspace }],
    queryFn: async () => {
      if (activeWorkspace !== null) {
        // Per-workspace: use SDK client (goes through gateway with workspace context)
        const result = await client.listDocuments({ status, limit: 100 });
        return result.documents;
      }
      // Cross-workspace: use admin endpoint
      const result = await listAdminDocuments({ status, limit: 100 });
      return result.items as unknown as DocumentInfo[];
    },
    enabled: isConnected,
    staleTime: 30_000,
  });
}

export function useDocument(id: string) {
  const { client } = useConnection();
  return useQuery<DocumentInfo>({
    queryKey: ["documents", id],
    queryFn: () => client.getDocument(id),
    enabled: !!id,
    retry: false,
  });
}

export function useDocumentJobs(documentId: string) {
  const { client } = useConnection();
  return useQuery<JobInfo[]>({
    queryKey: ["document-jobs", documentId],
    queryFn: async () => {
      const result = await client.listJobs({ limit: 50 });
      return result.jobs.filter((j) => j.document_ids.includes(documentId));
    },
    enabled: !!documentId,
    staleTime: 15_000,
  });
}
