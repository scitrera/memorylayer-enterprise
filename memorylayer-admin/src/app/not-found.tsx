// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

import Link from "next/link";
import { FileQuestion } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-slate-200 bg-white p-12 text-center">
      <div className="mb-4 rounded-full bg-muted p-3">
        <FileQuestion className="h-6 w-6 text-muted-foreground" />
      </div>
      <h3 className="text-lg font-medium text-foreground">Page not found</h3>
      <p className="mt-1 max-w-sm text-sm text-muted-foreground">
        The page you are looking for does not exist.
      </p>
      <Button variant="outline" className="mt-4" asChild>
        <Link href="/">Go to Dashboard</Link>
      </Button>
    </div>
  );
}
