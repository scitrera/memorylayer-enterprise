"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Skeleton } from "@/components/ui/skeleton";
import { useTieringConfig, useUpdateTieringConfig } from "@/hooks/use-tiering";

export function TieringConfigForm() {
  const { data: config, isLoading } = useTieringConfig();
  const updateConfig = useUpdateTieringConfig();

  const [autoArchiveDays, setAutoArchiveDays] = useState<string>("");
  const [importanceThreshold, setImportanceThreshold] = useState<number>(0.3);
  const [compressionEnabled, setCompressionEnabled] = useState(false);

  useEffect(() => {
    if (config) {
      setAutoArchiveDays(config.auto_archive_days != null ? String(config.auto_archive_days) : "");
      setImportanceThreshold(config.archive_importance_threshold ?? 0.3);
      setCompressionEnabled(config.compression_enabled);
    }
  }, [config]);

  async function handleSave() {
    try {
      await updateConfig.mutateAsync({
        auto_archive_days: autoArchiveDays ? Number(autoArchiveDays) : null,
        archive_importance_threshold: importanceThreshold,
        compression_enabled: compressionEnabled,
      });
      toast.success("Tiering config saved");
    } catch {
      toast.error("Failed to save tiering config");
    }
  }

  if (isLoading) {
    return <Skeleton className="h-56 w-full" />;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Tiering Configuration</CardTitle>
        <CardDescription>Configure automatic archival and compression settings.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <label className="text-sm font-medium">Auto-archive after (days)</label>
          <Input
            type="number"
            min={1}
            placeholder="Disabled"
            value={autoArchiveDays}
            onChange={(e) => setAutoArchiveDays(e.target.value)}
            className="w-40"
          />
          <p className="text-xs text-muted-foreground">
            Memories older than this threshold will be archived automatically. Leave blank to disable.
          </p>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Archive importance threshold</label>
            <span className="text-sm font-mono text-muted-foreground">
              {importanceThreshold.toFixed(2)}
            </span>
          </div>
          <Slider
            min={0}
            max={1}
            step={0.01}
            value={[importanceThreshold]}
            onValueChange={([v]) => setImportanceThreshold(v)}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">
            Memories with importance below this value will be eligible for archival.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <input
            id="compression-enabled"
            type="checkbox"
            checked={compressionEnabled}
            onChange={(e) => setCompressionEnabled(e.target.checked)}
            className="h-4 w-4 rounded border-slate-300"
          />
          <label htmlFor="compression-enabled" className="text-sm font-medium cursor-pointer">
            Enable compression for archived memories
          </label>
        </div>

        <Button onClick={handleSave} disabled={updateConfig.isPending}>
          {updateConfig.isPending ? "Saving..." : "Save Changes"}
        </Button>
      </CardContent>
    </Card>
  );
}
