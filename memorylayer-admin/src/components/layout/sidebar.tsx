// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Database,
  Key,
  Clock,
  FileText,
  BarChart3,
  Briefcase,
  ScrollText,
  Route,
  Users,
  Server,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { ConnectionStatus } from "./connection-status";
import { WorkspaceSwitcher } from "./workspace-switcher";

interface NavGroup {
  label: string;
  items: { href: string; label: string; icon: React.ElementType }[];
}

const navGroups: NavGroup[] = [
  {
    label: "Overview",
    items: [
      { href: "/", label: "Dashboard", icon: LayoutDashboard },
    ],
  },
  {
    label: "Data",
    items: [
      { href: "/workspaces", label: "Workspaces", icon: Briefcase },
      { href: "/memories", label: "Memories", icon: Database },
      { href: "/sessions", label: "Sessions", icon: Clock },
      { href: "/documents", label: "Documents", icon: FileText },
      { href: "/datasets", label: "Datasets", icon: BarChart3 },
    ],
  },
  {
    label: "Access",
    items: [
      { href: "/tokens", label: "API Tokens", icon: Key },
    ],
  },
  {
    label: "Observability",
    items: [
      { href: "/audit", label: "Audit Log", icon: ScrollText },
      { href: "/trajectories", label: "Trajectories", icon: Route },
      { href: "/entities", label: "Entities", icon: Users },
      { href: "/jobs", label: "Jobs", icon: Briefcase },
    ],
  },
  {
    label: "System",
    items: [
      { href: "/system", label: "Tiering & Health", icon: Server },
      { href: "/settings", label: "Settings", icon: Settings },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-full w-64 flex-col border-r bg-card">
      <div className="flex items-baseline gap-1 px-4 py-5">
        <span className="text-lg font-semibold text-brand-600">MemoryLayer</span>
        <span className="text-sm text-slate-500">Admin</span>
      </div>

      <WorkspaceSwitcher />

      <nav className="flex-1 space-y-4 overflow-y-auto px-2 pb-4">
        {navGroups.map((group) => (
          <div key={group.label}>
            <div className="mb-1 px-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              {group.label}
            </div>
            <div className="space-y-0.5">
              {group.items.map((item) => {
                const isActive =
                  item.href === "/"
                    ? pathname === "/"
                    : pathname.startsWith(item.href);
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                      isActive
                        ? "bg-brand-50 text-brand-600"
                        : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="border-t">
        <ConnectionStatus />
        <a
          href="https://github.com/scitrera/memorylayer-enterprise"
          target="_blank"
          rel="noopener noreferrer"
          className="block px-4 pb-3 text-xs text-muted-foreground hover:text-foreground"
        >
          Source code · AGPLv3
        </a>
      </div>
    </aside>
  );
}
