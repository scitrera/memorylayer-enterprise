"use client";

import { useState } from "react";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useArchiveMemories, useRestoreMemories } from "@/hooks/use-tiering";

interface ArchiveDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ArchiveDialog({ open, onOpenChange }: ArchiveDialogProps) {
  const [workspaceId, setWorkspaceId] = useState("");
  const [olderThanDays, setOlderThanDays] = useState("");
  const [importanceBelow, setImportanceBelow] = useState("");
  const archive = useArchiveMemories();

  async function handleSubmit() {
    try {
      const result = await archive.mutateAsync({
        ...(workspaceId ? { workspace_id: workspaceId } : {}),
        ...(olderThanDays ? { older_than_days: Number(olderThanDays) } : {}),
        ...(importanceBelow ? { importance_below: Number(importanceBelow) } : {}),
      });
      toast.success(`Archived ${result.archived_count} memories`);
      onOpenChange(false);
      setWorkspaceId("");
      setOlderThanDays("");
      setImportanceBelow("");
    } catch {
      toast.error("Archive operation failed");
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Archive Memories</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div className="space-y-1.5">
            <label className="text-sm font-medium">Workspace ID (optional)</label>
            <Input
              placeholder="All workspaces"
              value={workspaceId}
              onChange={(e) => setWorkspaceId(e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-sm font-medium">Older than (days)</label>
            <Input
              type="number"
              min={1}
              placeholder="No age filter"
              value={olderThanDays}
              onChange={(e) => setOlderThanDays(e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-sm font-medium">Importance below</label>
            <Input
              type="number"
              min={0}
              max={1}
              step={0.01}
              placeholder="No importance filter"
              value={importanceBelow}
              onChange={(e) => setImportanceBelow(e.target.value)}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={archive.isPending}>
            {archive.isPending ? "Archiving..." : "Archive"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

interface RestoreDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function RestoreDialog({ open, onOpenChange }: RestoreDialogProps) {
  const [workspaceId, setWorkspaceId] = useState("");
  const [memoryIds, setMemoryIds] = useState("");
  const restore = useRestoreMemories();

  async function handleSubmit() {
    const ids = memoryIds
      .split(/[\n,]+/)
      .map((s) => s.trim())
      .filter(Boolean);

    try {
      const result = await restore.mutateAsync({
        ...(workspaceId ? { workspace_id: workspaceId } : {}),
        ...(ids.length > 0 ? { memory_ids: ids } : {}),
      });
      toast.success(`Restored ${result.restored_count} memories`);
      onOpenChange(false);
      setWorkspaceId("");
      setMemoryIds("");
    } catch {
      toast.error("Restore operation failed");
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Restore Memories</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div className="space-y-1.5">
            <label className="text-sm font-medium">Workspace ID (optional)</label>
            <Input
              placeholder="All workspaces"
              value={workspaceId}
              onChange={(e) => setWorkspaceId(e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-sm font-medium">Memory IDs (optional)</label>
            <textarea
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring min-h-24 resize-y font-mono"
              placeholder="Paste memory IDs, one per line or comma-separated"
              value={memoryIds}
              onChange={(e) => setMemoryIds(e.target.value)}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={restore.isPending}>
            {restore.isPending ? "Restoring..." : "Restore"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
