// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { use } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { EntityCardView } from "@/components/entity/entity-card";
import { useEntityCard, useEntityInsights } from "@/hooks/use-entities";

export default function EntityDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data: card, isLoading: cardLoading } = useEntityCard(id);
  const { data: insights, isLoading: insightsLoading } = useEntityInsights(id);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="sm" asChild>
          <Link href="/entities">
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back
          </Link>
        </Button>
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            {card?.entity.name ?? "Entity Detail"}
          </h1>
          <p className="text-muted-foreground font-mono text-xs">{id}</p>
        </div>
      </div>

      <EntityCardView card={card} isLoading={cardLoading} />

      {insightsLoading ? (
        <Skeleton className="h-32 w-full" />
      ) : insights && insights.insights.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              Insights
              <span className="text-xs font-normal text-muted-foreground">
                confidence: {(insights.confidence * 100).toFixed(0)}%
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {insights.insights.map((insight, i) => (
                <li key={i} className="flex gap-2 text-sm">
                  <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-violet-500 shrink-0" />
                  {insight}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
