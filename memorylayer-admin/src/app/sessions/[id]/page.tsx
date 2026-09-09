"use client";

import { use } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { TimeAgo } from "@/components/shared/time-ago";
import { JsonViewer } from "@/components/shared/json-viewer";
import { useSession, useWorkingMemory } from "@/hooks/use-sessions";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";

interface PageProps {
  params: Promise<{ id: string }>;
}

export default function SessionDetailPage({ params }: PageProps) {
  const { id } = use(params);
  const { data: session, isLoading, error } = useSession(id);
  const { data: workingMemory, isLoading: memLoading } = useWorkingMemory(id);

  const expired = session ? new Date(session.expires_at) < new Date() : false;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link href="/sessions" className="text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold tracking-tight font-mono">{id}</h1>
          <p className="text-muted-foreground">Session detail</p>
        </div>
      </div>

      {isLoading && (
        <div className="space-y-4">
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-48 w-full" />
        </div>
      )}

      {error && <p className="text-sm text-destructive">Session not found or failed to load.</p>}

      {session && (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                Session Info
                <Badge variant={expired ? "secondary" : "default"}>
                  {expired ? "Expired" : "Active"}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Workspace</p>
                <p className="font-mono">{session.workspace_id}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Context</p>
                <p className="font-mono">{session.context_id}</p>
              </div>
              {session.user_id && (
                <div>
                  <p className="text-muted-foreground">User</p>
                  <p className="font-mono">{session.user_id}</p>
                </div>
              )}
              <div>
                <p className="text-muted-foreground">Created</p>
                <TimeAgo date={session.created_at} />
              </div>
              <div>
                <p className="text-muted-foreground">Expires</p>
                <TimeAgo date={session.expires_at} />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Working Memory</CardTitle>
            </CardHeader>
            <CardContent>
              {memLoading ? (
                <Skeleton className="h-32 w-full" />
              ) : workingMemory && Object.keys(workingMemory).length > 0 ? (
                <div className="overflow-x-auto rounded-md border">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/50">
                        <th className="px-4 py-2 text-left font-medium text-muted-foreground">Key</th>
                        <th className="px-4 py-2 text-left font-medium text-muted-foreground">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(workingMemory).map(([key, value]) => (
                        <tr key={key} className="border-b last:border-0">
                          <td className="px-4 py-2 font-mono text-xs font-medium">{key}</td>
                          <td className="px-4 py-2 font-mono text-xs text-muted-foreground max-w-md">
                            <JsonViewer data={value} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No working memory entries.</p>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
