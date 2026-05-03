// lib/api.ts
// SpatialVoiceAI — Typed API Client
// All calls to the FastAPI backend go through here. Never write raw fetch() in components.

// ─── Base ─────────────────────────────────────────────────────────────────────

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${options.method ?? "GET"} ${path} → ${res.status}: ${body}`);
  }

  return res.json() as Promise<T>;
}

// ─── Response Types ───────────────────────────────────────────────────────────

export interface StartSessionResponse {
  sessionId: string;
  createdAt: number;
}

export interface SessionMetaResponse {
  sessionId: string;
  name: string;
  status: string;
  createdAt: number;
  endedAt: number | null;
  nodeCount: number;
  actionItemCount: number;
}

export interface EndSessionResponse {
  sessionId: string;
  totalUtterances: number;
  actionItemCount: number;
  graphEdgeCount: number;
  durationSeconds: number;
}

export type EdgeType = "semantic" | "turn" | "reference";
export type EventFlagType =
  | "decision"
  | "action_item"
  | "question"
  | "disagreement"
  | "none";

export interface GraphNodeResponse {
  id: string;
  speakerId: string;
  speakerName: string;
  text: string;
  timestamp: number;
  eventFlags: EventFlagType[];
  confidence: number;
}

export interface GraphEdgeResponse {
  source: string;
  target: string;
  weight: number;
  type: EdgeType;
}

export interface GraphResponse {
  sessionId: string;
  nodes: GraphNodeResponse[];
  edges: GraphEdgeResponse[];
}

export interface QARequest {
  sessionId: string;
  question: string;
}

export interface QAResponse {
  question: string;
  answer: string;
  citationNodeIds: string[];
  latencyMs: number;
}

export interface ActionItemResponse {
  id: string;
  ownerSpeaker: string;
  ownerName: string;
  taskText: string;
  deadlineHint: string | null;
  sourceNodeId: string;
  createdAt: number;
}

export interface SessionStatsResponse {
  sessionId: string;
  dominantSpeaker: string | null;
  interruptionCount: number;
  avgResponseLatencyMs: number;
  topicClusters: string[];
}

export interface ExportResponse {
  sessionId: string;
  exportFormat: "samsung_notes";
  data: {
    title: string;
    createdAt: number;
    transcript: { speaker: string; text: string; timestamp: number }[];
    actionItems: { owner: string; task: string; deadline: string | null }[];
  };
}

// ─── Session Endpoints ────────────────────────────────────────────────────────

/**
 * Start a new call session.
 * Call this when the user clicks "Start Listening" in SpeakerModal.
 */
export async function startSession(
  name: string,
  speakerNames: { SPK_0: string; SPK_1: string }
): Promise<StartSessionResponse> {
  return request<StartSessionResponse>("/session/start", {
    method: "POST",
    body: JSON.stringify({ name, speakerNames }),
  });
}

/**
 * Get metadata for an existing session.
 * Use on the /call/[id] page load to verify session exists.
 */
export async function getSession(
  sessionId: string
): Promise<SessionMetaResponse> {
  return request<SessionMetaResponse>(`/session/${sessionId}`);
}

/**
 * End a session. Backend serializes graph, compiles action items.
 * Call when user clicks "End Call".
 */
export async function endSession(
  sessionId: string
): Promise<EndSessionResponse> {
  return request<EndSessionResponse>(`/session/${sessionId}/end`, {
    method: "POST",
  });
}

// ─── Graph Endpoints ──────────────────────────────────────────────────────────

/**
 * Fetch the full conversation graph for a session.
 * Use on the ended page and optionally to seed GraphView mid-call.
 */
export async function getGraph(sessionId: string): Promise<GraphResponse> {
  return request<GraphResponse>(`/graph/${sessionId}`);
}

// ─── Q&A Endpoint ─────────────────────────────────────────────────────────────

/**
 * Submit a question against the session's conversation graph.
 * Returns answer + citation node IDs. Expect 700-900ms response time.
 */
export async function submitQuestion(
  sessionId: string,
  question: string
): Promise<QAResponse> {
  return request<QAResponse>("/qa", {
    method: "POST",
    body: JSON.stringify({ sessionId, question } satisfies QARequest),
  });
}

// ─── Action Items ─────────────────────────────────────────────────────────────

/**
 * Fetch all action items for a session.
 * Use on the ended/summary page.
 */
export async function getActionItems(
  sessionId: string
): Promise<ActionItemResponse[]> {
  return request<ActionItemResponse[]>(`/session/${sessionId}/action-items`);
}

// ─── Analytics ────────────────────────────────────────────────────────────────

/**
 * Get analytics for a completed session.
 * Dominant speaker, interruption count, avg latency, topic clusters.
 */
export async function getSessionStats(
  sessionId: string
): Promise<SessionStatsResponse> {
  return request<SessionStatsResponse>(`/session/${sessionId}/stats`);
}

// ─── Export ───────────────────────────────────────────────────────────────────

/**
 * Export session data as Samsung Notes compatible JSON.
 * Triggers a file download in the browser.
 */
export async function exportSession(sessionId: string): Promise<void> {
  const data = await request<ExportResponse>(
    `/session/${sessionId}/export`
  );
  const blob = new Blob([JSON.stringify(data.data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `spatialvoice-${sessionId}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── WebSocket URL Builder ────────────────────────────────────────────────────

/**
 * Returns the WebSocket URL for a session.
 * Use this in useWebSocket hook — never hardcode the WS URL.
 */
export function getWebSocketURL(sessionId: string): string {
  const wsBase = BASE.replace(/^http/, "ws");
  return `${wsBase}/ws/${sessionId}`;
}