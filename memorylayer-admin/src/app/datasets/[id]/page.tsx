// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { use } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { TimeAgo } from "@/components/shared/time-ago";
import { useDataset, useDatasetJobs } from "@/hooks/use-datasets";
import { ArrowLeft, RefreshCw } from "lucide-react";
import Link from "next/link";
import { formatNumber } from "@/lib/format";
import type { DatasetColumn } from "@scitrera/memorylayer-sdk";

interface PageProps {
  params: Promise<{ id: string }>;
}

function statusVariant(status: string): "default" | "destructive" | "secondary" {
  if (status === "ready" || status === "completed") return "default";
  if (status === "failed") return "destructive";
  return "secondary";
}

function ColumnRow({ col }: { col: DatasetColumn }) {
  return (
    <tr className="border-b last:border-0 hover:bg-muted/30 text-sm">
      <td className="px-4 py-2 font-mono font-medium">{col.name}</td>
      <td className="px-4 py-2 text-muted-foreground">{col.dtype}</td>
      <td className="px-4 py-2">
        <Badge variant="secondary" className="text-xs">{col.column_type}</Badge>
      </td>
      <td className="px-4 py-2 text-muted-foreground">{col.null_count} ({col.null_percent.toFixed(1)}%)</td>
      <td className="px-4 py-2 text-muted-foreground">{formatNumber(col.unique_count)}</td>
      <td className="px-4 py-2 text-muted-foreground">
        {col.mean_value != null ? col.mean_value.toFixed(2) : "—"}
      </td>
    </tr>
  );
}

export default function DatasetDetailPage({ params }: PageProps) {
  const { id } = use(params);
  const { data: dataset, isLoading, error, refetch } = useDataset(id);
  const { data: jobs, isLoading: jobsLoading } = useDatasetJobs(id);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link href="/datasets" className="text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <div className="flex-1 min-w-0">
          <h1 className="text-2xl font-bold tracking-tight truncate">
            {dataset?.name || dataset?.filename || id}
          </h1>
          <p className="text-muted-foreground">Dataset detail</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh
        </Button>
      </div>

      {isLoading && (
        <div className="space-y-4">
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-48 w-full" />
        </div>
      )}

      {error && <p className="text-sm text-destructive">Dataset not found or failed to load.</p>}

      {dataset && (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                Metadata
                <Badge variant={statusVariant(dataset.status)}>{dataset.status}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-4 text-sm md:grid-cols-3">
              <div>
                <p className="text-muted-foreground">Format</p>
                <p className="uppercase">{dataset.format}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Rows</p>
                <p>{formatNumber(dataset.row_count)}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Columns</p>
                <p>{dataset.column_count}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Size</p>
                <p>{formatNumber(dataset.size_bytes)}B</p>
              </div>
              <div>
                <p className="text-muted-foreground">Memories</p>
                <p>{dataset.memory_ids.length}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Workspace</p>
                <p className="font-mono text-xs">{dataset.workspace_id}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Uploaded</p>
                <TimeAgo date={dataset.created_at} />
              </div>
              {dataset.profiling_completed_at && (
                <div>
                  <p className="text-muted-foreground">Profiled</p>
                  <TimeAgo date={dataset.profiling_completed_at} />
                </div>
              )}
            </CardContent>
          </Card>

          {dataset.profile_summary && (
            <Card>
              <CardHeader>
                <CardTitle>Profile Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground whitespace-pre-wrap">{dataset.profile_summary}</p>
              </CardContent>
            </Card>
          )}

          {dataset.columns && dataset.columns.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Column Statistics</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/50">
                        <th className="px-4 py-2 text-left font-medium text-muted-foreground">Column</th>
                        <th className="px-4 py-2 text-left font-medium text-muted-foreground">Type</th>
                        <th className="px-4 py-2 text-left font-medium text-muted-foreground">Category</th>
                        <th className="px-4 py-2 text-left font-medium text-muted-foreground">Nulls</th>
                        <th className="px-4 py-2 text-left font-medium text-muted-foreground">Unique</th>
                        <th className="px-4 py-2 text-left font-medium text-muted-foreground">Mean</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dataset.columns.map((col) => (
                        <ColumnRow key={col.name} col={col} />
                      ))}
                    </tbody>
                  </table>
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
                <p className="text-sm text-muted-foreground">No jobs found for this dataset.</p>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
