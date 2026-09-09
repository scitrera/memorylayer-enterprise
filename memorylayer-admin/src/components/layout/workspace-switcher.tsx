// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { ChevronsUpDown, Globe, Briefcase } from "lucide-react";
import { useWorkspaceContext } from "@/providers/workspace-provider";
import { cn } from "@/lib/cn";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export function WorkspaceSwitcher() {
  const { activeWorkspace, setActiveWorkspace, workspaces, isLoading } =
    useWorkspaceContext();

  return (
    <div className="px-3 pb-3">
      <Select
        value={activeWorkspace ?? "__all__"}
        onValueChange={(value) =>
          setActiveWorkspace(value === "__all__" ? null : value)
        }
      >
        <SelectTrigger
          className={cn(
            "w-full text-sm",
            activeWorkspace === null && "text-brand-600 font-medium",
          )}
        >
          <SelectValue placeholder={isLoading ? "Loading..." : "All Workspaces"} />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="__all__">
            <span className="flex items-center gap-2">
              <Globe className="h-3.5 w-3.5 text-muted-foreground" />
              All Workspaces
            </span>
          </SelectItem>
          {workspaces.map((ws) => (
            <SelectItem key={ws.id} value={ws.id}>
              <span className="flex items-center gap-2">
                <Briefcase className="h-3.5 w-3.5 text-muted-foreground" />
                {ws.name}
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
