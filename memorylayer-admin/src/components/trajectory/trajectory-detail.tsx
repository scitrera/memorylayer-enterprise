// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { TimeAgo } from "@/components/shared/time-ago";
import type { Trajectory } from "@/types/admin";

interface TrajectoryDetailProps {
  trajectory: Trajectory;
}

export function TrajectoryDetail({ trajectory }: TrajectoryDetailProps) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Query</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm leading-relaxed">{trajectory.query}</p>
          <Separator />
          <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
            <span>
              Workspace:{" "}
              <span className="font-mono text-foreground">{trajectory.workspace_id}</span>
            </span>
            <span>
              Results: <span className="font-semibold text-foreground">{trajectory.results_count}</span>
            </span>
            <span>
              Total latency: <span className="font-semibold text-foreground">{trajectory.latency_ms}ms</span>
            </span>
            <span>
              Created: <TimeAgo date={trajectory.created_at} />
            </span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Pipeline Steps</CardTitle>
        </CardHeader>
        <CardContent>
          {trajectory.steps.length === 0 ? (
            <p className="text-sm text-muted-foreground">No step data available.</p>
          ) : (
            <ol className="space-y-4">
              {trajectory.steps.map((step, idx) => (
                <li key={idx} className="flex gap-4">
                  <div className="flex flex-col items-center">
                    <div className="flex h-7 w-7 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-bold">
                      {idx + 1}
                    </div>
                    {idx < trajectory.steps.length - 1 && (
                      <div className="mt-1 w-px flex-1 bg-slate-200" />
                    )}
                  </div>
                  <div className="flex-1 pb-4">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="font-medium text-sm">{step.stage}</span>
                      <Badge variant="outline" className="text-xs">
                        {step.latency_ms}ms
                      </Badge>
                      <Badge variant="secondary" className="text-xs">
                        {step.memory_ids.length} memories
                      </Badge>
                    </div>
                    {step.memory_ids.length > 0 && (
                      <div className="space-y-1">
                        {step.memory_ids.map((id, mIdx) => (
                          <div
                            key={id}
                            className="flex items-center gap-3 rounded-md bg-slate-50 px-3 py-1.5 text-xs"
                          >
                            <span className="font-mono text-muted-foreground">{id.slice(0, 12)}…</span>
                            {step.scores[mIdx] !== undefined && (
                              <span className="ml-auto text-muted-foreground">
                                score: {step.scores[mIdx].toFixed(4)}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </li>
              ))}
            </ol>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
