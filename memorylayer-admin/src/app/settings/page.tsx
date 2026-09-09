"use client";

import { useState } from "react";
import { toast } from "sonner";
import { useConnection } from "@/providers/connection-provider";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export default function SettingsPage() {
  const { connectionConfig, setConnectionConfig, testConnection, isConnected } = useConnection();
  const [baseUrl, setBaseUrl] = useState(connectionConfig.baseUrl);
  const [apiKey, setApiKey] = useState(connectionConfig.apiKey ?? "");
  const [testing, setTesting] = useState(false);

  const handleSave = async () => {
    setConnectionConfig({
      baseUrl,
      apiKey: apiKey || undefined,
    });
    setTesting(true);
    const ok = await testConnection();
    setTesting(false);
    if (ok) {
      toast.success("Connected successfully");
    } else {
      toast.error("Connection failed — check your settings");
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">Configure your admin dashboard connection.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Connection</CardTitle>
          <CardDescription>
            Configure the MemoryLayer API endpoint and authentication.
            Use the workspace switcher in the sidebar to change workspaces.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">API Base URL</label>
            <Input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="/api/ml"
            />
            <p className="text-xs text-muted-foreground">
              The base URL for the MemoryLayer API. Use /api/ml for the built-in proxy,
              or a direct URL like http://localhost:40080.
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">API Key</label>
            <Input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your API key"
            />
            <p className="text-xs text-muted-foreground">
              Admin-scoped API key (level 40+) from Aether.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Button onClick={handleSave} disabled={testing}>
              {testing ? "Testing..." : "Save & Test"}
            </Button>
            <span className="flex items-center gap-2 text-sm">
              <span
                className={`h-2 w-2 rounded-full ${isConnected ? "bg-green-500" : "bg-red-500"}`}
              />
              {isConnected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
