// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { TimeAgo } from "@/components/shared/time-ago";
import Link from "next/link";
import type { EntityCard as EntityCardType } from "@/types/admin";

interface EntityCardProps {
  card: EntityCardType | undefined;
  isLoading: boolean;
}

export function EntityCardView({ card, isLoading }: EntityCardProps) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-40 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  if (!card) return null;

  const { entity } = card;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between gap-3">
            <div>
              <CardTitle className="text-xl">{entity.name}</CardTitle>
              <div className="flex items-center gap-2 mt-1">
                <Badge variant="outline">{entity.entity_type}</Badge>
                <span className="text-xs text-muted-foreground">
                  {entity.memory_count} memories
                </span>
              </div>
            </div>
            <div className="text-xs text-muted-foreground text-right space-y-0.5">
              <div className="font-mono">{entity.workspace_id}</div>
              <div>
                Updated <TimeAgo date={entity.updated_at} />
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {card.summary && (
            <>
              <p className="text-sm leading-relaxed">{card.summary}</p>
              <Separator />
            </>
          )}

          {card.key_facts.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold mb-2">Key Facts</h4>
              <ul className="space-y-1">
                {card.key_facts.map((fact, i) => (
                  <li key={i} className="flex gap-2 text-sm">
                    <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-primary shrink-0" />
                    {fact}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>

      {card.related_entities.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Related Entities</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {card.related_entities.map((rel) => (
                <div
                  key={rel.id}
                  className="flex items-center justify-between rounded-md bg-slate-50 px-3 py-2 text-sm"
                >
                  <Link
                    href={`/entities/${rel.id}`}
                    className="font-medium text-primary hover:underline"
                  >
                    {rel.name}
                  </Link>
                  <Badge variant="secondary" className="text-xs">
                    {rel.relationship}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
