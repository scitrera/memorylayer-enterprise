// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

import { cn } from "@/lib/cn";

const statusStyles: Record<string, string> = {
  active: "bg-green-50 text-green-700 border-green-200",
  healthy: "bg-green-50 text-green-700 border-green-200",
  completed: "bg-green-50 text-green-700 border-green-200",
  running: "bg-blue-50 text-blue-700 border-blue-200",
  processing: "bg-blue-50 text-blue-700 border-blue-200",
  pending: "bg-amber-50 text-amber-700 border-amber-200",
  degraded: "bg-amber-50 text-amber-700 border-amber-200",
  expired: "bg-amber-50 text-amber-700 border-amber-200",
  revoked: "bg-red-50 text-red-700 border-red-200",
  failed: "bg-red-50 text-red-700 border-red-200",
  unhealthy: "bg-red-50 text-red-700 border-red-200",
  detected: "bg-orange-50 text-orange-700 border-orange-200",
  resolved: "bg-slate-50 text-slate-700 border-slate-200",
  dismissed: "bg-slate-50 text-slate-500 border-slate-200",
};

interface StatusBadgeProps {
  status: string;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const style = statusStyles[status] ?? "bg-slate-50 text-slate-700 border-slate-200";
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium capitalize",
        style,
        className,
      )}
    >
      {status}
    </span>
  );
}
