// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AuditEventList } from "@/components/audit/audit-event-list";
import { AuditSummaryView } from "@/components/audit/audit-summary";
import { useAuditEvents, useAuditSummary } from "@/hooks/use-audit";

const PAGE_SIZE = 50;

export default function AuditPage() {
  const [filters, setFilters] = useState({
    event_type: "",
    action: "",
    workspace_id: "",
    from: "",
    to: "",
  });

  const queryParams = {
    ...(filters.event_type ? { event_type: filters.event_type } : {}),
    ...(filters.action ? { action: filters.action } : {}),
    ...(filters.workspace_id ? { workspace_id: filters.workspace_id } : {}),
    ...(filters.from ? { from: new Date(filters.from).toISOString() } : {}),
    ...(filters.to ? { to: new Date(filters.to).toISOString() } : {}),
    limit: PAGE_SIZE,
  };

  const summaryParams = {
    ...(filters.from ? { from: new Date(filters.from).toISOString() } : {}),
    ...(filters.to ? { to: new Date(filters.to).toISOString() } : {}),
  };

  const { data: events = [], isLoading: eventsLoading } = useAuditEvents(queryParams);
  const { data: summary, isLoading: summaryLoading } = useAuditSummary(summaryParams);

  function handleFilterChange(key: string, value: string) {
    setFilters((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Audit Log</h1>
        <p className="text-muted-foreground">Review system events and activity history.</p>
      </div>

      <Tabs defaultValue="events">
        <TabsList>
          <TabsTrigger value="events">Events</TabsTrigger>
          <TabsTrigger value="summary">Summary</TabsTrigger>
        </TabsList>

        <TabsContent value="events" className="mt-4">
          <AuditEventList
            events={events}
            isLoading={eventsLoading}
            filters={filters}
            onFilterChange={handleFilterChange}
          />
        </TabsContent>

        <TabsContent value="summary" className="mt-4">
          <AuditSummaryView summary={summary} isLoading={summaryLoading} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
