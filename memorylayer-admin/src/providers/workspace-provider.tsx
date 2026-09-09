"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";
import { useConnection } from "@/providers/connection-provider";
import type { Workspace } from "@/types";

interface WorkspaceContextValue {
  activeWorkspace: string | null; // null = "All Workspaces"
  setActiveWorkspace: (id: string | null) => void;
  workspaces: Workspace[];
  isLoading: boolean;
}

const STORAGE_KEY = "memorylayer-workspace";

const WorkspaceContext = createContext<WorkspaceContextValue | null>(null);

function loadActiveWorkspace(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "null" || stored === null) return null;
    return stored;
  } catch {
    return null;
  }
}

function saveActiveWorkspace(id: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (id === null) {
      localStorage.setItem(STORAGE_KEY, "null");
    } else {
      localStorage.setItem(STORAGE_KEY, id);
    }
  } catch {
    // ignore storage errors
  }
}

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const { client, isConnected } = useConnection();
  const [activeWorkspace, setActiveWorkspaceState] = useState<string | null>(null);
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Load persisted workspace on mount
  useEffect(() => {
    setActiveWorkspaceState(loadActiveWorkspace());
  }, []);

  // Fetch workspace list when connected
  useEffect(() => {
    if (!isConnected) {
      setWorkspaces([]);
      return;
    }
    let cancelled = false;
    setIsLoading(true);
    client
      .listWorkspaces()
      .then((ws) => {
        if (!cancelled) setWorkspaces(ws);
      })
      .catch(() => {
        if (!cancelled) setWorkspaces([]);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [client, isConnected]);

  const setActiveWorkspace = useCallback((id: string | null) => {
    setActiveWorkspaceState(id);
    saveActiveWorkspace(id);
  }, []);

  return (
    <WorkspaceContext.Provider
      value={{
        activeWorkspace,
        setActiveWorkspace,
        workspaces,
        isLoading,
      }}
    >
      {children}
    </WorkspaceContext.Provider>
  );
}

export function useWorkspaceContext(): WorkspaceContextValue {
  const context = useContext(WorkspaceContext);
  if (!context) {
    throw new Error("useWorkspaceContext must be used within a WorkspaceProvider");
  }
  return context;
}
