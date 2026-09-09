// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { use } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { TimeAgo } from "@/components/shared/time-ago";
import { useDocument, useDocumentJobs } from "@/hooks/use-documents";
import { ArrowLeft, RefreshCw } from "lucide-react";
import Link from "next/link";
import { formatNumber } from "@/lib/format";

interface PageProps {
  params: Promise<{ id: string }>;
}

function statusVariant(status: string): "default" | "destructive" | "secondary" {
  if (status === "ready" || status === "completed") return "default";
  if (status === "failed") return "destructive";
  return "secondary";
}

export default function DocumentDetailPage({ params }: PageProps) {
  const { id } = use(params);
  const { data: doc, isLoading, error, refetch } = useDocument(id);
  const { data: jobs, isLoading: jobsLoading } = useDocumentJobs(id);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link href="/documents" className="text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <div className="flex-1 min-w-0">
          <h1 className="text-2xl font-bold tracking-tight truncate">
            {doc?.filename ?? id}
          </h1>
          <p className="text-muted-foreground">Document detail</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh
        </Button>
      </div>

      {isLoading && (
        <div className="space-y-4">
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-32 w-full" />
        </div>
      )}

      {error && <p className="text-sm text-destructive">Document not found or failed to load.</p>}

      {doc && (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                Metadata
                <Badge variant={statusVariant(doc.status)}>{doc.status}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-4 text-sm md:grid-cols-3">
              <div>
                <p className="text-muted-foreground">Type</p>
                <p>{doc.document_type}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Size</p>
                <p>{formatNumber(doc.size_bytes)}B</p>
              </div>
              <div>
                <p className="text-muted-foreground">Pages</p>
                <p>{doc.page_count}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Chunks</p>
                <p>{doc.chunk_count}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Memories</p>
                <p>{doc.memory_ids.length}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Workspace</p>
                <p className="font-mono text-xs">{doc.workspace_id}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Uploaded</p>
                <TimeAgo date={doc.created_at} />
              </div>
              {doc.processing_completed_at && (
                <div>
                  <p className="text-muted-foreground">Processed</p>
                  <TimeAgo date={doc.processing_completed_at} />
                </div>
              )}
            </CardContent>
          </Card>

          {doc.memory_ids.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Extracted Memories ({doc.memory_ids.length})</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {doc.memory_ids.map((memId) => (
                    <Badge key={memId} variant="secondary" className="font-mono text-xs">
                      {memId.slice(0, 8)}…
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Processing Jobs</CardTitle>
            </CardHeader>
            <CardContent>
              {jobsLoading ? (
                <Skeleton className="h-16 w-full" />
              ) : jobs && jobs.length > 0 ? (
                <div className="space-y-2">
                  {jobs.map((job) => (
                    <div key={job.id} className="flex items-center justify-between rounded-md border px-4 py-2 text-sm">
                      <span className="font-mono text-xs text-muted-foreground">{job.id.slice(0, 12)}…</span>
                      <Badge variant={statusVariant(job.status)}>{job.status}</Badge>
                      <span className="text-muted-foreground">{job.progress_percent}%</span>
                      <TimeAgo date={job.created_at} />
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No jobs found for this document.</p>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
