// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useState } from "react";
import { ArchiveRestore, Archive } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { TieringStatsView } from "@/components/tiering/tiering-stats";
import { TieringConfigForm } from "@/components/tiering/tiering-config";
import { ArchiveDialog, RestoreDialog } from "@/components/tiering/archive-restore-dialog";
import { useTieringStats, useHealthDependencies } from "@/hooks/use-tiering";

function statusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
  if (status === "healthy") return "default";
  if (status === "degraded") return "secondary";
  return "destructive";
}

function statusClass(status: string): string {
  if (status === "healthy") return "bg-green-100 text-green-800 border-green-200";
  if (status === "degraded") return "bg-yellow-100 text-yellow-800 border-yellow-200";
  return "bg-red-100 text-red-800 border-red-200";
}

export default function SystemPage() {
  const [archiveOpen, setArchiveOpen] = useState(false);
  const [restoreOpen, setRestoreOpen] = useState(false);

  const { data: tieringStats, isLoading: statsLoading } = useTieringStats();
  const { data: healthDeps, isLoading: healthLoading } = useHealthDependencies();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">System</h1>
        <p className="text-muted-foreground">Manage tiering, storage, and monitor system health.</p>
      </div>

      <Tabs defaultValue="tiering">
        <TabsList>
          <TabsTrigger value="tiering">Tiering</TabsTrigger>
          <TabsTrigger value="health">Health</TabsTrigger>
        </TabsList>

        <TabsContent value="tiering" className="mt-4 space-y-6">
          <TieringStatsView stats={tieringStats} isLoading={statsLoading} />

          <Separator />

          <TieringConfigForm />

          <Separator />

          <div className="flex gap-3">
            <Button variant="outline" onClick={() => setArchiveOpen(true)}>
              <Archive className="h-4 w-4 mr-2" />
              Archive Memories
            </Button>
            <Button variant="outline" onClick={() => setRestoreOpen(true)}>
              <ArchiveRestore className="h-4 w-4 mr-2" />
              Restore Memories
            </Button>
          </div>
        </TabsContent>

        <TabsContent value="health" className="mt-4">
          {healthLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : !healthDeps || healthDeps.length === 0 ? (
            <p className="text-sm text-muted-foreground">No health dependency data available.</p>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Dependencies</CardTitle>
              </CardHeader>
              <CardContent className="divide-y">
                {healthDeps.map((dep) => (
                  <div key={dep.name} className="flex items-center justify-between py-3">
                    <div>
                      <span className="font-medium text-sm">{dep.name}</span>
                      {dep.details && (
                        <p className="text-xs text-muted-foreground mt-0.5">{dep.details}</p>
                      )}
                    </div>
                    <div className="flex items-center gap-3">
                      {dep.latency_ms != null && (
                        <span className="text-xs text-muted-foreground">{dep.latency_ms}ms</span>
                      )}
                      <span
                        className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold ${statusClass(dep.status)}`}
                      >
                        {dep.status}
                      </span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      <ArchiveDialog open={archiveOpen} onOpenChange={setArchiveOpen} />
      <RestoreDialog open={restoreOpen} onOpenChange={setRestoreOpen} />
    </div>
  );
}
