"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { JsonViewer } from "@/components/shared/json-viewer";
import { EmptyState } from "@/components/shared/empty-state";
import { TimeAgo } from "@/components/shared/time-ago";
import type { AuditEvent } from "@/types/admin";
import { ClipboardList } from "lucide-react";

interface AuditEventListProps {
  events: AuditEvent[];
  isLoading: boolean;
  filters: {
    event_type: string;
    action: string;
    workspace_id: string;
    from: string;
    to: string;
  };
  onFilterChange: (key: string, value: string) => void;
}

const EVENT_TYPE_OPTIONS = [
  "memory",
  "session",
  "workspace",
  "token",
  "document",
  "dataset",
  "entity",
  "contradiction",
];

function AuditEventRow({ event }: { event: AuditEvent }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      <tr
        className="cursor-pointer border-b hover:bg-slate-50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <td className="px-4 py-3 text-sm text-muted-foreground whitespace-nowrap">
          <TimeAgo date={event.timestamp} />
        </td>
        <td className="px-4 py-3">
          <Badge variant="outline" className="font-mono text-xs">
            {event.event_type}
          </Badge>
        </td>
        <td className="px-4 py-3 text-sm font-mono">{event.action}</td>
        <td className="px-4 py-3 text-sm text-muted-foreground">
          {event.workspace_id ?? "—"}
        </td>
        <td className="px-4 py-3 text-sm text-muted-foreground">
          {event.user_id ?? "—"}
        </td>
        <td className="px-4 py-3 text-sm text-muted-foreground">
          {event.resource_type ? (
            <span>
              {event.resource_type}
              {event.resource_id ? (
                <span className="font-mono text-xs ml-1 text-slate-400">
                  /{event.resource_id.slice(0, 8)}
                </span>
              ) : null}
            </span>
          ) : (
            "—"
          )}
        </td>
        <td className="px-4 py-3 text-muted-foreground">
          {expanded ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </td>
      </tr>
      {expanded && (
        <tr className="border-b bg-slate-50">
          <td colSpan={7} className="px-4 py-3">
            <JsonViewer data={event.metadata} defaultExpanded={true} />
          </td>
        </tr>
      )}
    </>
  );
}

export function AuditEventList({
  events,
  isLoading,
  filters,
  onFilterChange,
}: AuditEventListProps) {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3">
        <Select
          value={filters.event_type || "all"}
          onValueChange={(v) => onFilterChange("event_type", v === "all" ? "" : v)}
        >
          <SelectTrigger className="w-40">
            <SelectValue placeholder="Event type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All types</SelectItem>
            {EVENT_TYPE_OPTIONS.map((t) => (
              <SelectItem key={t} value={t}>
                {t}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Input
          placeholder="Action..."
          className="w-40"
          value={filters.action}
          onChange={(e) => onFilterChange("action", e.target.value)}
        />

        <Input
          placeholder="Workspace ID..."
          className="w-48"
          value={filters.workspace_id}
          onChange={(e) => onFilterChange("workspace_id", e.target.value)}
        />

        <Input
          type="datetime-local"
          className="w-52"
          value={filters.from}
          onChange={(e) => onFilterChange("from", e.target.value)}
        />

        <Input
          type="datetime-local"
          className="w-52"
          value={filters.to}
          onChange={(e) => onFilterChange("to", e.target.value)}
        />
      </div>

      {isLoading ? (
        <div className="space-y-2">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      ) : events.length === 0 ? (
        <EmptyState
          icon={ClipboardList}
          title="No audit events"
          description="No events match your current filters."
        />
      ) : (
        <div className="rounded-lg border overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 border-b">
              <tr>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Time</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Event Type</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Action</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Workspace</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">User</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Resource</th>
                <th className="px-4 py-3 w-8" />
              </tr>
            </thead>
            <tbody>
              {events.map((event) => (
                <AuditEventRow key={event.id} event={event} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
