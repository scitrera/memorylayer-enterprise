// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { use } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { TrajectoryDetail } from "@/components/trajectory/trajectory-detail";
import { useTrajectory } from "@/hooks/use-trajectories";

export default function TrajectoryDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data: trajectory, isLoading } = useTrajectory(id);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="sm" asChild>
          <Link href="/trajectories">
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back
          </Link>
        </Button>
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Trajectory Detail</h1>
          <p className="text-muted-foreground font-mono text-xs">{id}</p>
        </div>
      </div>

      {isLoading ? (
        <div className="space-y-4">
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : trajectory ? (
        <TrajectoryDetail trajectory={trajectory} />
      ) : (
        <p className="text-muted-foreground">Trajectory not found.</p>
      )}
    </div>
  );
}
