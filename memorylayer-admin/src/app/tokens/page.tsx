// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/shared/empty-state";
import { TokenList } from "@/components/token/token-list";
import { TokenCreateDialog } from "@/components/token/token-create-dialog";
import { useTokens } from "@/hooks/use-tokens";
import { KeyRound, Plus } from "lucide-react";

export default function TokensPage() {
  const [showRevoked, setShowRevoked] = useState(false);
  const { data: tokens, isLoading, error } = useTokens(showRevoked);
  const [createOpen, setCreateOpen] = useState(false);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">API Tokens</h1>
          <p className="text-muted-foreground">Manage API keys and access tokens.</p>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              checked={showRevoked}
              onChange={(e) => setShowRevoked(e.target.checked)}
              className="rounded border-input"
            />
            Show revoked
          </label>
          <Button onClick={() => setCreateOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Create Token
          </Button>
        </div>
      </div>

      {isLoading && (
        <div className="space-y-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      )}

      {error && (
        <p className="text-sm text-destructive">Failed to load tokens.</p>
      )}

      {!isLoading && !error && tokens && tokens.length === 0 && (
        <EmptyState
          icon={KeyRound}
          title="No tokens yet"
          description="Create an API token to allow clients to authenticate with the server."
          action={{ label: "Create Token", onClick: () => setCreateOpen(true) }}
        />
      )}

      {!isLoading && !error && tokens && tokens.length > 0 && (
        <TokenList tokens={tokens} />
      )}

      <TokenCreateDialog open={createOpen} onOpenChange={setCreateOpen} />
    </div>
  );
}
