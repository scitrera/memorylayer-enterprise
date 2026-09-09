// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useState } from "react";
import { EntitySearch } from "@/components/entity/entity-search";
import { useEntities } from "@/hooks/use-entities";

export default function EntitiesPage() {
  const [query, setQuery] = useState("");
  const [entityType, setEntityType] = useState("");

  const { data: entities = [], isLoading } = useEntities({
    ...(query ? { query } : {}),
    limit: 60,
  });

  const filtered = entityType
    ? entities.filter((e) => e.entity_type === entityType)
    : entities;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Entities</h1>
        <p className="text-muted-foreground">Browse and search knowledge graph entities.</p>
      </div>

      <EntitySearch
        entities={filtered}
        isLoading={isLoading}
        query={query}
        entityType={entityType}
        onQueryChange={setQuery}
        onTypeChange={setEntityType}
      />
    </div>
  );
}
