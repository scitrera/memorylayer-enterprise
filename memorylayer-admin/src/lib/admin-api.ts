// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

import type {
  Token,
  TokenCreateRequest,
  TokenCreateResponse,
  AuditEvent,
  AuditSummary,
  TieringStats,
  TieringConfig,
  Trajectory,
  Entity,
  EntityCard,
  EntityInsights,
  HealthDependency,
  Contradiction,
} from "@/types/admin";

class AdminApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "AdminApiError";
  }
}

function getHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "X-Workspace-ID": "_system",
  };
  if (typeof window !== "undefined") {
    try {
      const stored = localStorage.getItem("memorylayer-connection");
      if (stored) {
        const config = JSON.parse(stored);
        if (config.apiKey) {
          headers["Authorization"] = `Bearer ${config.apiKey}`;
        }
      }
    } catch {
      // ignore
    }
  }
  return headers;
}

function getBaseUrl(): string {
  if (typeof window !== "undefined") {
    try {
      const stored = localStorage.getItem("memorylayer-connection");
      if (stored) {
        const config = JSON.parse(stored);
        if (config.baseUrl) return config.baseUrl;
      }
    } catch {
      // ignore
    }
  }
  return "/api/ml";
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const baseUrl = getBaseUrl();
  const response = await fetch(`${baseUrl}${path}`, {
    ...options,
    headers: {
      ...getHeaders(),
      ...options?.headers,
    },
  });
  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error");
    throw new AdminApiError(response.status, text);
  }
  return response.json();
}

// ── Tokens ──────────────────────────────────────────────────────────

interface RawTokenResponse {
  id: string;
  name: string;
  principal_type?: string;
  workspace_patterns: string[];
  scopes: string[];
  created_at: string;
  expires_at: string | null;
  revoked: boolean;
}

function mapToken(raw: RawTokenResponse): Token {
  let status: Token["status"] = "active";
  if (raw.revoked) {
    status = "revoked";
  } else if (raw.expires_at && new Date(raw.expires_at) < new Date()) {
    status = "expired";
  }
  return {
    id: raw.id,
    name: raw.name,
    principal_type: (raw.principal_type as Token["principal_type"]) || "User",
    key_prefix: "",
    workspace_patterns: raw.workspace_patterns,
    scopes: raw.scopes,
    created_at: raw.created_at,
    expires_at: raw.expires_at,
    revoked_at: raw.revoked ? raw.created_at : null, // not available from gRPC
    last_used_at: null, // not available from gRPC list
    status,
  };
}

export async function listTokens(includeRevoked = false): Promise<Token[]> {
  const params = includeRevoked ? "?include_revoked=true" : "";
  const res = await request<{ tokens: RawTokenResponse[] }>(`/v1/tokens${params}`);
  return res.tokens.map(mapToken);
}

export async function createToken(data: TokenCreateRequest): Promise<TokenCreateResponse> {
  return request<TokenCreateResponse>("/v1/tokens", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function revokeToken(tokenId: string): Promise<void> {
  await request(`/v1/tokens/${encodeURIComponent(tokenId)}/revoke`, {
    method: "POST",
  });
}

export async function deleteToken(tokenId: string): Promise<void> {
  await request(`/v1/tokens/${encodeURIComponent(tokenId)}`, {
    method: "DELETE",
  });
}

// ── Audit ───────────────────────────────────────────────────────────

export async function listAuditEvents(params?: {
  event_type?: string;
  action?: string;
  workspace_id?: string;
  user_id?: string;
  from?: string;
  to?: string;
  limit?: number;
  offset?: number;
}): Promise<AuditEvent[]> {
  const searchParams = new URLSearchParams();
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined) searchParams.set(key, String(value));
    }
  }
  const qs = searchParams.toString();
  const result = await request<{ events: AuditEvent[]; count: number }>(`/v1/audit/events${qs ? `?${qs}` : ""}`);
  return result.events;
}

export async function getAuditSummary(params?: {
  from?: string;
  to?: string;
}): Promise<AuditSummary> {
  const searchParams = new URLSearchParams();
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined) searchParams.set(key, String(value));
    }
  }
  const qs = searchParams.toString();
  return request<AuditSummary>(`/v1/audit/events/summary${qs ? `?${qs}` : ""}`);
}

// ── Tiering ─────────────────────────────────────────────────────────

export async function getTieringStats(): Promise<TieringStats> {
  return request<TieringStats>("/v1/tiering/stats");
}

export async function getTieringConfig(): Promise<TieringConfig> {
  return request<TieringConfig>("/v1/tiering/config");
}

export async function updateTieringConfig(config: Partial<TieringConfig>): Promise<TieringConfig> {
  return request<TieringConfig>("/v1/tiering/config", {
    method: "PUT",
    body: JSON.stringify(config),
  });
}

export async function archiveMemories(params: {
  workspace_id?: string;
  older_than_days?: number;
  importance_below?: number;
}): Promise<{ archived_count: number }> {
  return request("/v1/tiering/archive", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function restoreMemories(params: {
  workspace_id?: string;
  memory_ids?: string[];
}): Promise<{ restored_count: number }> {
  return request("/v1/tiering/restore", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

// ── Trajectories ────────────────────────────────────────────────────

export async function listTrajectories(params?: {
  workspace_id?: string;
  limit?: number;
  offset?: number;
}): Promise<Trajectory[]> {
  const searchParams = new URLSearchParams();
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined) searchParams.set(key, String(value));
    }
  }
  const qs = searchParams.toString();
  return request<Trajectory[]>(`/trajectories${qs ? `?${qs}` : ""}`);
}

export async function getTrajectory(id: string): Promise<Trajectory> {
  return request<Trajectory>(`/trajectories/${encodeURIComponent(id)}`);
}

// ── Entities ────────────────────────────────────────────────────────

export async function listEntities(params?: {
  workspace_id?: string;
  query?: string;
  limit?: number;
  offset?: number;
}): Promise<Entity[]> {
  const searchParams = new URLSearchParams();
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined) searchParams.set(key, String(value));
    }
  }
  const qs = searchParams.toString();
  return request<Entity[]>(`/v1/entities${qs ? `?${qs}` : ""}`);
}

export async function getEntityCard(id: string): Promise<EntityCard> {
  return request<EntityCard>(`/v1/entities/${encodeURIComponent(id)}/card`);
}

export async function getEntityInsights(id: string): Promise<EntityInsights> {
  return request<EntityInsights>(`/v1/entities/${encodeURIComponent(id)}/insights`);
}

export async function deriveEntity(id: string): Promise<Entity> {
  return request<Entity>(`/v1/entities/${encodeURIComponent(id)}/derive`, {
    method: "POST",
  });
}

// ── Contradictions ──────────────────────────────────────────────────

export async function listContradictions(workspaceId: string): Promise<Contradiction[]> {
  return request<Contradiction[]>(
    `/v1/workspaces/${encodeURIComponent(workspaceId)}/contradictions`,
  );
}

export async function resolveContradiction(
  contradictionId: string,
  resolution: { keep_memory_id: string; action: "keep_a" | "keep_b" | "merge" },
): Promise<void> {
  await request(`/v1/contradictions/${encodeURIComponent(contradictionId)}/resolve`, {
    method: "POST",
    body: JSON.stringify(resolution),
  });
}

// ── Health ──────────────────────────────────────────────────────────

export async function getHealthDependencies(): Promise<HealthDependency[]> {
  const result = await request<{
    status: string;
    dependencies: Record<string, { status: string; details?: Record<string, unknown> }>;
  }>("/v1/health/dependencies");

  return Object.entries(result.dependencies).map(([name, dep]) => ({
    name,
    status: dep.status === "connected" ? "healthy" : dep.status === "disconnected" ? "unhealthy" : "degraded",
    latency_ms: null,
    details: dep.details ? JSON.stringify(dep.details) : null,
  })) as HealthDependency[];
}

// ── Admin Overview ──────────────────────────────────────────────────

export interface AdminStatsResponse {
  workspace_count: number;
  memory_count: number;
  session_count: number;
  document_count: number;
  dataset_count: number;
  token_count: number;
}

export interface PaginatedResponse<T = Record<string, unknown>> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export async function getAdminStats(): Promise<AdminStatsResponse> {
  return request<AdminStatsResponse>("/v1/admin/stats");
}

function buildAdminParams(params?: {
  workspace_id?: string | null;
  status?: string;
  limit?: number;
  offset?: number;
  include_expired?: boolean;
}): string {
  const searchParams = new URLSearchParams();
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null) searchParams.set(key, String(value));
    }
  }
  const qs = searchParams.toString();
  return qs ? `?${qs}` : "";
}

export async function listAdminMemories(params?: {
  workspace_id?: string | null;
  status?: string;
  limit?: number;
  offset?: number;
}): Promise<PaginatedResponse> {
  return request<PaginatedResponse>(`/v1/admin/memories${buildAdminParams(params)}`);
}

export async function listAdminSessions(params?: {
  workspace_id?: string | null;
  include_expired?: boolean;
  limit?: number;
  offset?: number;
}): Promise<PaginatedResponse> {
  return request<PaginatedResponse>(`/v1/admin/sessions${buildAdminParams(params)}`);
}

export async function listAdminDocuments(params?: {
  workspace_id?: string | null;
  status?: string;
  limit?: number;
  offset?: number;
}): Promise<PaginatedResponse> {
  return request<PaginatedResponse>(`/v1/admin/documents${buildAdminParams(params)}`);
}

export async function listAdminDatasets(params?: {
  workspace_id?: string | null;
  status?: string;
  limit?: number;
  offset?: number;
}): Promise<PaginatedResponse> {
  return request<PaginatedResponse>(`/v1/admin/datasets${buildAdminParams(params)}`);
}

export async function listAdminJobs(params?: {
  workspace_id?: string | null;
  status?: string;
  limit?: number;
  offset?: number;
}): Promise<PaginatedResponse> {
  return request<PaginatedResponse>(`/v1/admin/jobs${buildAdminParams(params)}`);
}
