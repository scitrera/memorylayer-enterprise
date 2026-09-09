"use client";

import { useState } from "react";
import Link from "next/link";
import { Plus, Briefcase, ChevronRight } from "lucide-react";
import { useWorkspaceList, useCreateWorkspace } from "@/hooks/use-workspaces";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { EmptyState } from "@/components/shared/empty-state";
import { TimeAgo } from "@/components/shared/time-ago";
import { toast } from "sonner";

function CreateWorkspaceDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [name, setName] = useState("");
  const createMutation = useCreateWorkspace();

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    createMutation.mutate(
      { name: name.trim() },
      {
        onSuccess: () => {
          toast.success("Workspace created");
          setName("");
          onOpenChange(false);
        },
        onError: (err) => toast.error(`Failed to create workspace: ${err.message}`),
      }
    );
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Create Workspace</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Workspace Name
            </label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. my-project"
              className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              autoFocus
            />
          </div>
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => onOpenChange(false)}
              className="inline-flex h-9 items-center rounded-md border border-input bg-background px-4 text-sm font-medium shadow-sm hover:bg-accent"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!name.trim() || createMutation.isPending}
              className="inline-flex h-9 items-center rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground shadow hover:bg-primary/90 disabled:opacity-50"
            >
              {createMutation.isPending ? "Creating..." : "Create"}
            </button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export default function WorkspacesPage() {
  const [createOpen, setCreateOpen] = useState(false);
  const { data: workspaces, isLoading, isError, error } = useWorkspaceList();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Workspaces</h1>
          <p className="text-muted-foreground">
            Manage isolated memory namespaces
          </p>
        </div>
        <button
          onClick={() => setCreateOpen(true)}
          className="inline-flex h-9 items-center gap-2 rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground shadow hover:bg-primary/90"
        >
          <Plus className="h-4 w-4" />
          Create Workspace
        </button>
      </div>

      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-28 w-full rounded-2xl" />
          ))}
        </div>
      ) : isError ? (
        <EmptyState
          title="Failed to load workspaces"
          description={
            error instanceof Error ? error.message : "Check your connection settings"
          }
        />
      ) : !workspaces || workspaces.length === 0 ? (
        <EmptyState
          icon={Briefcase}
          title="No workspaces"
          description="Create a workspace to get started."
        />
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {workspaces.map((ws) => (
            <Link key={ws.id} href={`/workspaces/${ws.id}`}>
              <Card className="group cursor-pointer rounded-2xl border border-slate-200 shadow-sm transition-shadow hover:shadow-md">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center justify-between text-base">
                    <span className="truncate">{ws.name}</span>
                    <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-1">
                  <p className="font-mono text-xs text-muted-foreground truncate">
                    {ws.id}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Created <TimeAgo date={ws.created_at} />
                  </p>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}

      <CreateWorkspaceDialog open={createOpen} onOpenChange={setCreateOpen} />
    </div>
  );
}
