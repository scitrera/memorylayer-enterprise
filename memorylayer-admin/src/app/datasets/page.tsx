// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/shared/empty-state";
import { DatasetList } from "@/components/dataset/dataset-list";
import { useDatasets } from "@/hooks/use-datasets";
import { Table2 } from "lucide-react";

const STATUS_FILTERS = [
  { label: "All", value: undefined },
  { label: "Processing", value: "processing" },
  { label: "Ready", value: "ready" },
  { label: "Failed", value: "failed" },
] as const;

export default function DatasetsPage() {
  const [status, setStatus] = useState<string | undefined>(undefined);
  const { data: datasets, isLoading, error } = useDatasets(status);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Datasets</h1>
        <p className="text-muted-foreground">Uploaded datasets and their processing status.</p>
      </div>

      <div className="flex gap-2">
        {STATUS_FILTERS.map((f) => (
          <button
            key={String(f.value)}
            onClick={() => setStatus(f.value)}
            className={`rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
              status === f.value
                ? "border-primary bg-primary text-primary-foreground"
                : "border-input bg-background hover:bg-accent"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {isLoading && (
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      )}

      {error && <p className="text-sm text-destructive">Failed to load datasets.</p>}

      {!isLoading && !error && datasets && datasets.length === 0 && (
        <EmptyState
          icon={Table2}
          title="No datasets"
          description="No datasets found for the selected filter."
        />
      )}

      {!isLoading && !error && datasets && datasets.length > 0 && (
        <DatasetList datasets={datasets} />
      )}
    </div>
  );
}
