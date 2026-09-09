// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { use } from "react";
import Link from "next/link";
import { ArrowLeft, Database } from "lucide-react";
import { useWorkspace, useWorkspaceSchema } from "@/hooks/use-workspaces";
import { useMemoryList } from "@/hooks/use-memories";
import { MemoryCard } from "@/components/memory/memory-card";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/shared/empty-state";
import { JsonViewer } from "@/components/shared/json-viewer";
import { TimeAgo } from "@/components/shared/time-ago";
import { formatDate } from "@/lib/format";

export default function WorkspaceDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data: workspace, isLoading: wsLoading } = useWorkspace(id);
  const { data: schema } = useWorkspaceSchema(id);
  const { data: memoriesData, isLoading: memLoading } = useMemoryList({ limit: 12 });

  const memories = memoriesData?.memories ?? [];

  if (wsLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-32 w-full rounded-2xl" />
      </div>
    );
  }

  if (!workspace) {
    return (
      <EmptyState
        title="Workspace not found"
        description="The workspace could not be loaded."
      />
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link
          href="/workspaces"
          className="inline-flex h-8 w-8 items-center justify-center rounded-md border text-muted-foreground hover:bg-accent"
        >
          <ArrowLeft className="h-4 w-4" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold tracking-tight">{workspace.name}</h1>
          <p className="font-mono text-xs text-muted-foreground">{workspace.id}</p>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card className="rounded-2xl">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Created
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm font-medium">{formatDate(workspace.created_at)}</p>
          </CardContent>
        </Card>
        <Card className="rounded-2xl">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Last Updated
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm font-medium">
              <TimeAgo date={workspace.updated_at} />
            </p>
          </CardContent>
        </Card>
        <Card className="rounded-2xl">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Memories
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">
              {memoriesData?.total_count ?? "—"}
            </p>
          </CardContent>
        </Card>
      </div>

      {schema && (
        <Card className="rounded-2xl">
          <CardHeader>
            <CardTitle className="text-base">Ontology Schema</CardTitle>
          </CardHeader>
          <CardContent>
            <JsonViewer data={schema as unknown as Record<string, unknown>} defaultExpanded={false} />
          </CardContent>
        </Card>
      )}

      <div>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Recent Memories</h2>
          <Link
            href="/memories"
            className="text-sm text-primary hover:underline"
          >
            View all
          </Link>
        </div>
        {memLoading ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-28 rounded-2xl" />
            ))}
          </div>
        ) : memories.length === 0 ? (
          <EmptyState
            icon={Database}
            title="No memories"
            description="No memories exist in this workspace yet."
          />
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {memories.map((memory) => (
              <MemoryCard key={memory.id} memory={memory} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
