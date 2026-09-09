"use client";

import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/shared/empty-state";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { TimeAgo } from "@/components/shared/time-ago";
import Link from "next/link";
import type { Entity } from "@/types/admin";

const ENTITY_TYPES = ["person", "organization", "place", "concept", "event", "product", "other"];

interface EntitySearchProps {
  entities: Entity[];
  isLoading: boolean;
  query: string;
  entityType: string;
  onQueryChange: (q: string) => void;
  onTypeChange: (t: string) => void;
}

export function EntitySearch({
  entities,
  isLoading,
  query,
  entityType,
  onQueryChange,
  onTypeChange,
}: EntitySearchProps) {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3">
        <div className="relative flex-1 min-w-48">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            className="pl-9"
            placeholder="Search entities..."
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
          />
        </div>
        <Select
          value={entityType || "all"}
          onValueChange={(v) => onTypeChange(v === "all" ? "" : v)}
        >
          <SelectTrigger className="w-44">
            <SelectValue placeholder="Entity type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All types</SelectItem>
            {ENTITY_TYPES.map((t) => (
              <SelectItem key={t} value={t}>
                {t}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
      ) : entities.length === 0 ? (
        <EmptyState
          icon={Search}
          title="No entities found"
          description="Try adjusting your search or filters."
        />
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {entities.map((entity) => (
            <Link key={entity.id} href={`/entities/${entity.id}`}>
              <Card className="hover:shadow-md transition-shadow cursor-pointer h-full">
                <CardContent className="p-4 space-y-2">
                  <div className="flex items-start justify-between gap-2">
                    <span className="font-semibold text-sm leading-tight">{entity.name}</span>
                    <Badge variant="outline" className="shrink-0 text-xs">
                      {entity.entity_type}
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground space-y-1">
                    <div>{entity.memory_count} memories</div>
                    <div className="font-mono truncate">{entity.workspace_id}</div>
                    <div>
                      Updated <TimeAgo date={entity.updated_at} />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
