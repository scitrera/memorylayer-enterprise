"use client";

import { useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/shared/empty-state";
import { DocumentList } from "@/components/document/document-list";
import { useDocuments } from "@/hooks/use-documents";
import { FileText } from "lucide-react";

const STATUS_FILTERS = [
  { label: "All", value: undefined },
  { label: "Processing", value: "processing" },
  { label: "Ready", value: "ready" },
  { label: "Failed", value: "failed" },
] as const;

export default function DocumentsPage() {
  const [status, setStatus] = useState<string | undefined>(undefined);
  const { data: documents, isLoading, error } = useDocuments(status);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Documents</h1>
          <p className="text-muted-foreground">Uploaded documents and their processing status.</p>
        </div>
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

      {error && <p className="text-sm text-destructive">Failed to load documents.</p>}

      {!isLoading && !error && documents && documents.length === 0 && (
        <EmptyState
          icon={FileText}
          title="No documents"
          description="No documents found for the selected filter."
        />
      )}

      {!isLoading && !error && documents && documents.length > 0 && (
        <DocumentList documents={documents} />
      )}
    </div>
  );
}
