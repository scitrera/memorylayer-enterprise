// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import {
  LayoutDashboard,
  Database,
  Clock,
  Key,
  FileText,
  BarChart3,
} from "lucide-react";
import Link from "next/link";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useAdminStats } from "@/hooks/use-admin-stats";
import { useTieringStats } from "@/hooks/use-tiering";
import { useWorkspaceList } from "@/hooks/use-workspaces";
import { TimeAgo } from "@/components/shared/time-ago";

function StatCard({
  title,
  value,
  icon: Icon,
  loading,
  href,
}: {
  title: string;
  value: number | string;
  icon: React.ElementType;
  loading?: boolean;
  href?: string;
}) {
  const content = (
    <Card className="transition-shadow hover:shadow-md">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-8 w-20" />
        ) : (
          <div className="text-2xl font-bold">{value}</div>
        )}
      </CardContent>
    </Card>
  );

  if (href) {
    return <Link href={href}>{content}</Link>;
  }
  return content;
}

const TIER_COLORS = ["#3b82f6", "#94a3b8"];
const BAR_COLOR = "#6366f1";

export default function DashboardPage() {
  const { data: stats, isLoading } = useAdminStats();
  const { data: tieringStats } = useTieringStats();
  const { data: workspaces } = useWorkspaceList();

  // Resource breakdown from real admin stats
  const resourceData = stats
    ? [
        { name: "Memories", count: stats.memory_count },
        { name: "Documents", count: stats.document_count },
        { name: "Datasets", count: stats.dataset_count },
        { name: "Sessions", count: stats.session_count },
      ]
    : [];

  // Real tiering data from the tiering stats endpoint
  const tierData = tieringStats
    ? [
        { name: "Hot", value: tieringStats.hot_memory_count },
        { name: "Cold", value: tieringStats.cold_memory_count },
      ]
    : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Enterprise admin overview for your MemoryLayer deployment.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Workspaces"
          value={stats?.workspace_count ?? 0}
          icon={LayoutDashboard}
          loading={isLoading}
          href="/workspaces"
        />
        <StatCard
          title="Memories"
          value={stats?.memory_count ?? 0}
          icon={Database}
          loading={isLoading}
          href="/memories"
        />
        <StatCard
          title="Sessions"
          value={stats?.session_count ?? 0}
          icon={Clock}
          loading={isLoading}
          href="/sessions"
        />
        <StatCard
          title="API Tokens"
          value={stats?.token_count ?? 0}
          icon={Key}
          loading={isLoading}
          href="/tokens"
        />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {/* Resource Breakdown */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
              Resource Breakdown
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!stats ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                Connect to see resource data.
              </p>
            ) : resourceData.every((d) => d.count === 0) ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                No resources yet. Create workspaces and memories to see data here.
              </p>
            ) : (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={resourceData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill={BAR_COLOR} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Storage Tier Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Database className="h-4 w-4 text-muted-foreground" />
              Storage Tier Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!tieringStats ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                Connect to see storage tier data.
              </p>
            ) : tierData.every((d) => d.value === 0) ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                No memories stored yet.
              </p>
            ) : (
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie
                    data={tierData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={75}
                    dataKey="value"
                    paddingAngle={3}
                  >
                    {tierData.map((_, index) => (
                      <Cell key={index} fill={TIER_COLORS[index % TIER_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend iconSize={10} />
                </PieChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {/* Additional Stats */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Additional Counts</CardTitle>
          </CardHeader>
          <CardContent>
            {!stats ? (
              <p className="py-4 text-center text-sm text-muted-foreground">
                Connect to see data.
              </p>
            ) : (
              <div className="space-y-3">
                {[
                  { label: "Documents", value: stats.document_count, icon: FileText, href: "/documents" },
                  { label: "Datasets", value: stats.dataset_count, icon: BarChart3, href: "/datasets" },
                ].map((item) => (
                  <Link
                    key={item.label}
                    href={item.href}
                    className="flex items-center justify-between py-2 hover:opacity-80"
                  >
                    <span className="flex items-center gap-2 text-sm">
                      <item.icon className="h-4 w-4 text-muted-foreground" />
                      {item.label}
                    </span>
                    <span className="text-sm font-semibold">{item.value.toLocaleString()}</span>
                  </Link>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Workspaces */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Recent Workspaces</CardTitle>
            <Link href="/workspaces" className="text-xs text-primary hover:underline">
              View all
            </Link>
          </CardHeader>
          <CardContent>
            {!workspaces || workspaces.length === 0 ? (
              <p className="py-4 text-center text-sm text-muted-foreground">
                Connect to see workspaces.
              </p>
            ) : (
              <div className="divide-y">
                {workspaces.slice(0, 5).map((ws) => (
                  <Link
                    key={ws.id}
                    href={`/workspaces/${ws.id}`}
                    className="flex items-center justify-between py-2.5 hover:opacity-80"
                  >
                    <span className="truncate text-sm font-medium">{ws.name}</span>
                    <TimeAgo
                      date={ws.created_at}
                      className="shrink-0 text-xs text-muted-foreground"
                    />
                  </Link>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
