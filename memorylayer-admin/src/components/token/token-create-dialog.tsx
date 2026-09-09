"use client";

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { TagInput } from "@/components/shared/tag-input";
import { useCreateToken } from "@/hooks/use-tokens";
import { toast } from "sonner";
import { Copy, Check } from "lucide-react";

const AVAILABLE_SCOPES = ["read", "write", "admin", "tokens:read", "tokens:write"];

interface TokenCreateDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function TokenCreateDialog({ open, onOpenChange }: TokenCreateDialogProps) {
  const createToken = useCreateToken();
  const [name, setName] = useState("");
  const [principalType, setPrincipalType] = useState<"User" | "Agent">("User");
  const [workspacePatterns, setWorkspacePatterns] = useState<string[]>([]);
  const [scopes, setScopes] = useState<string[]>([]);
  const [expiresInDays, setExpiresInDays] = useState("");
  const [generatedKey, setGeneratedKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const toggleScope = (scope: string) => {
    setScopes((prev) =>
      prev.includes(scope) ? prev.filter((s) => s !== scope) : [...prev, scope]
    );
  };

  const handleSubmit = () => {
    if (!name.trim()) {
      toast.error("Name is required");
      return;
    }
    createToken.mutate(
      {
        name: name.trim(),
        principal_type: principalType,
        workspace_patterns: workspacePatterns.length > 0 ? workspacePatterns : undefined,
        scopes: scopes.length > 0 ? scopes : undefined,
        expires_in_days: expiresInDays ? parseInt(expiresInDays, 10) : undefined,
      },
      {
        onSuccess: (res) => {
          setGeneratedKey(res.token);
          toast.success("Token created");
        },
        onError: () => toast.error("Failed to create token"),
      }
    );
  };

  const handleCopy = async () => {
    if (!generatedKey) return;
    await navigator.clipboard.writeText(generatedKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleClose = () => {
    setName("");
    setPrincipalType("User");
    setWorkspacePatterns([]);
    setScopes([]);
    setExpiresInDays("");
    setGeneratedKey(null);
    setCopied(false);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Create API Token</DialogTitle>
        </DialogHeader>

        {generatedKey ? (
          <div className="space-y-4 py-2">
            <p className="text-sm text-muted-foreground">
              Copy your API key now. It will not be shown again.
            </p>
            <div className="flex items-center gap-2 rounded-md border bg-muted p-3 font-mono text-sm break-all">
              <span className="flex-1">{generatedKey}</span>
              <Button size="sm" variant="ghost" onClick={handleCopy}>
                {copied ? <Check className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
              </Button>
            </div>
            <DialogFooter>
              <Button onClick={handleClose}>Done</Button>
            </DialogFooter>
          </div>
        ) : (
          <div className="space-y-4 py-2">
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Name *</label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. production-agent"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-medium">Principal Type</label>
              <div className="flex gap-2">
                {(["User", "Agent"] as const).map((type) => (
                  <button
                    key={type}
                    type="button"
                    onClick={() => setPrincipalType(type)}
                    className={`rounded-md border px-4 py-1.5 text-sm font-medium transition-colors ${
                      principalType === type
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-input bg-background hover:bg-accent"
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">User tokens are for dashboards and humans. Agent tokens are for services and automated clients.</p>
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-medium">Workspace Patterns</label>
              <TagInput
                tags={workspacePatterns}
                onChange={setWorkspacePatterns}
                placeholder="e.g. project-*, _default"
              />
              <p className="text-xs text-muted-foreground">Glob patterns for allowed workspaces. Leave empty for all.</p>
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-medium">Scopes</label>
              <div className="flex flex-wrap gap-2">
                {AVAILABLE_SCOPES.map((scope) => (
                  <button
                    key={scope}
                    type="button"
                    onClick={() => toggleScope(scope)}
                    className={`rounded-md border px-3 py-1 text-xs font-medium transition-colors ${
                      scopes.includes(scope)
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-input bg-background hover:bg-accent"
                    }`}
                  >
                    {scope}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-medium">Expires in (days)</label>
              <Input
                type="number"
                min={1}
                value={expiresInDays}
                onChange={(e) => setExpiresInDays(e.target.value)}
                placeholder="Leave empty for no expiry"
              />
            </div>

            <DialogFooter>
              <Button variant="outline" onClick={handleClose}>Cancel</Button>
              <Button onClick={handleSubmit} disabled={createToken.isPending}>
                {createToken.isPending ? "Creating..." : "Create Token"}
              </Button>
            </DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
