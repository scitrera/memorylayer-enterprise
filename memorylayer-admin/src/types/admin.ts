export interface Token {
  id: string;
  name: string;
  principal_type: "User" | "Agent" | string;
  key_prefix: string;
  workspace_patterns: string[];
  scopes: string[];
  expires_at: string | null;
  revoked_at: string | null;
  created_at: string;
  last_used_at: string | null;
  status: "active" | "revoked" | "expired";
}

export interface TokenCreateRequest {
  name: string;
  principal_type?: "User" | "Agent";
  workspace_patterns?: string[];
  scopes?: string[];
  expires_in_days?: number;
}

export interface TokenCreateResponse extends Token {
  token: string; // plaintext API key (only returned once on creation)
}

export interface AuditEvent {
  id: string;
  event_type: string;
  action: string;
  workspace_id: string | null;
  user_id: string | null;
  resource_type: string | null;
  resource_id: string | null;
  metadata: Record<string, unknown>;
  timestamp: string;
}

export interface AuditSummary {
  total_events: number;
  by_event_type: Record<string, number>;
  by_action: Record<string, number>;
  period_start: string;
  period_end: string;
}

export interface TieringStats {
  hot_memory_count: number;
  cold_memory_count: number;
  hot_storage_bytes: number;
  cold_storage_bytes: number;
  compression_ratio: number;
  estimated_savings_bytes: number;
  archival_candidates_count: number;
}

export interface TieringConfig {
  auto_archive_days: number | null;
  archive_importance_threshold: number | null;
  compression_enabled: boolean;
}

export interface Trajectory {
  id: string;
  workspace_id: string;
  query: string;
  results_count: number;
  latency_ms: number;
  created_at: string;
  steps: TrajectoryStep[];
}

export interface TrajectoryStep {
  stage: string;
  memory_ids: string[];
  scores: number[];
  latency_ms: number;
}

export interface Entity {
  id: string;
  name: string;
  entity_type: string;
  workspace_id: string;
  memory_count: number;
  created_at: string;
  updated_at: string;
}

export interface EntityCard {
  entity: Entity;
  summary: string;
  key_facts: string[];
  related_entities: { id: string; name: string; relationship: string }[];
}

export interface EntityInsights {
  entity: Entity;
  insights: string[];
  confidence: number;
}

export interface HealthDependency {
  name: string;
  status: "healthy" | "degraded" | "unhealthy";
  latency_ms: number | null;
  details: string | null;
}

export interface Contradiction {
  id: string;
  workspace_id: string;
  memory_a_id: string;
  memory_b_id: string;
  description: string;
  status: "detected" | "resolved" | "dismissed";
  resolved_at: string | null;
  created_at: string;
}

export interface AdminStats {
  workspace_count: number;
  memory_count: number;
  session_count: number;
  token_count: number;
  document_count: number;
  dataset_count: number;
}
