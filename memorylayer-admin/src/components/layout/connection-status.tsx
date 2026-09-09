// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import { useConnection } from "@/providers/connection-provider";

export function ConnectionStatus() {
  const { isConnected } = useConnection();

  return (
    <div className="flex items-center gap-2 px-3 py-2 text-sm">
      <span
        className={`h-2 w-2 rounded-full ${
          isConnected ? "bg-green-500" : "bg-red-500"
        }`}
      />
      <span className="text-muted-foreground">
        {isConnected ? "Connected" : "Disconnected"}
      </span>
    </div>
  );
}
