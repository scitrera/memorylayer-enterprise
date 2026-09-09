// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/shared/confirm-dialog";
import { TimeAgo } from "@/components/shared/time-ago";
import { useRevokeToken, useDeleteToken } from "@/hooks/use-tokens";
import type { Token } from "@/types/admin";
import { useState } from "react";
import { toast } from "sonner";
import { ShieldOff, Trash2 } from "lucide-react";

function statusVariant(status: Token["status"]): "default" | "destructive" | "secondary" {
  if (status === "active") return "default";
  if (status === "revoked") return "destructive";
  return "secondary";
}

function statusLabel(status: Token["status"]): string {
  if (status === "active") return "Active";
  if (status === "revoked") return "Revoked";
  return "Expired";
}

interface TokenListProps {
  tokens: Token[];
}

export function TokenList({ tokens }: TokenListProps) {
  const revokeToken = useRevokeToken();
  const deleteToken = useDeleteToken();
  const [revokeTarget, setRevokeTarget] = useState<Token | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Token | null>(null);

  const handleRevoke = () => {
    if (!revokeTarget) return;
    revokeToken.mutate(revokeTarget.id, {
      onSuccess: () => toast.success("Token revoked"),
      onError: () => toast.error("Failed to revoke token"),
    });
  };

  const handleDelete = () => {
    if (!deleteTarget) return;
    deleteToken.mutate(deleteTarget.id, {
      onSuccess: () => toast.success("Token deleted"),
      onError: () => toast.error("Failed to delete token"),
    });
  };

  return (
    <>
      <div className="overflow-x-auto rounded-lg border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="px-4 py-3 text-left font-medium text-muted-foreground">Name</th>
              <th className="px-4 py-3 text-left font-medium text-muted-foreground">Type</th>
              <th className="px-4 py-3 text-left font-medium text-muted-foreground">Key Prefix</th>
              <th className="px-4 py-3 text-left font-medium text-muted-foreground">Status</th>
              <th className="px-4 py-3 text-left font-medium text-muted-foreground">Scopes</th>
              <th className="px-4 py-3 text-left font-medium text-muted-foreground">Created</th>
              <th className="px-4 py-3 text-left font-medium text-muted-foreground">Last Used</th>
              <th className="px-4 py-3 text-left font-medium text-muted-foreground">Actions</th>
            </tr>
          </thead>
          <tbody>
            {tokens.map((token) => (
              <tr key={token.id} className="border-b last:border-0 hover:bg-muted/30">
                <td className="px-4 py-3 font-medium">{token.name}</td>
                <td className="px-4 py-3">
                  <Badge variant={token.principal_type === "Agent" ? "secondary" : "default"} className="text-xs">
                    {token.principal_type || "User"}
                  </Badge>
                </td>
                <td className="px-4 py-3 font-mono text-xs text-muted-foreground">{token.key_prefix}…</td>
                <td className="px-4 py-3">
                  <Badge variant={statusVariant(token.status)}>{statusLabel(token.status)}</Badge>
                </td>
                <td className="px-4 py-3">
                  <div className="flex flex-wrap gap-1">
                    {token.scopes.length === 0 ? (
                      <span className="text-muted-foreground">—</span>
                    ) : (
                      token.scopes.map((s) => (
                        <Badge key={s} variant="secondary" className="text-xs">
                          {s}
                        </Badge>
                      ))
                    )}
                  </div>
                </td>
                <td className="px-4 py-3 text-muted-foreground">
                  <TimeAgo date={token.created_at} />
                </td>
                <td className="px-4 py-3 text-muted-foreground">
                  <TimeAgo date={token.last_used_at} />
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-1">
                    {token.status === "active" && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setRevokeTarget(token)}
                        title="Revoke token"
                      >
                        <ShieldOff className="h-4 w-4" />
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setDeleteTarget(token)}
                      title="Delete token"
                    >
                      <Trash2 className="h-4 w-4 text-destructive" />
                    </Button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <ConfirmDialog
        open={!!revokeTarget}
        onOpenChange={(open) => !open && setRevokeTarget(null)}
        title="Revoke token"
        description={`Revoke "${revokeTarget?.name}"? This cannot be undone. Any client using this token will lose access immediately.`}
        confirmLabel="Revoke"
        destructive
        onConfirm={handleRevoke}
      />

      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(open) => !open && setDeleteTarget(null)}
        title="Delete token"
        description={`Permanently delete "${deleteTarget?.name}"?`}
        confirmLabel="Delete"
        destructive
        onConfirm={handleDelete}
      />
    </>
  );
}
