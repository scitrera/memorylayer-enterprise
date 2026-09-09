// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { TrajectoryList } from "@/components/trajectory/trajectory-list";
import { useTrajectories } from "@/hooks/use-trajectories";

export default function TrajectoriesPage() {
  const [workspaceId, setWorkspaceId] = useState("");

  const { data: trajectories = [], isLoading } = useTrajectories({
    ...(workspaceId ? { workspace_id: workspaceId } : {}),
    limit: 100,
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Trajectories</h1>
        <p className="text-muted-foreground">Inspect retrieval pipeline execution traces.</p>
      </div>

      <div className="flex gap-3">
        <Input
          placeholder="Filter by workspace ID..."
          className="w-64"
          value={workspaceId}
          onChange={(e) => setWorkspaceId(e.target.value)}
        />
      </div>

      <TrajectoryList trajectories={trajectories} isLoading={isLoading} />
    </div>
  );
}
