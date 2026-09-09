"use client";

import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-slate-200 bg-white p-12 text-center">
      <div className="mb-4 rounded-full bg-red-50 p-3">
        <AlertTriangle className="h-6 w-6 text-red-500" />
      </div>
      <h3 className="text-lg font-medium text-foreground">Something went wrong</h3>
      <p className="mt-1 max-w-sm text-sm text-muted-foreground">
        {error.message || "An unexpected error occurred."}
      </p>
      <Button variant="outline" className="mt-4" onClick={reset}>
        Try again
      </Button>
    </div>
  );
}
