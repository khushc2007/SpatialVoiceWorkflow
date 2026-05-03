// lib/store.ts
// SpatialVoiceAI — Central Zustand Store
// All UI state lives here. Components read from slices, WebSocket hook writes to slices.

import { create } from "zustand";

// ─── Types ────────────────────────────────────────────────────────────────────

export type SessionStatus = "idle" | "starting" | "active" | "ending" | "ended";

export interface Session {
  id: string;
  name: string;
  status: SessionStatus;
  startedAt: number | null;
  endedAt: number | null;
}

export type SpeakerID = "SPK_0" | "SPK_1";

export type EventFlagType =
  | "decision"
  | "action_item"
  | "question"
  | "disagreement"
  | "none";

export interface EventFlag {
  flagType: EventFlagType;
  confidence: number;
  evidenceText: string;
}

export interface Utterance {
  nodeId: string;
  speakerId: SpeakerID;
  speakerName: string;
  text: string;
  timestamp: number;
  confidence: number;
  eventFlags: EventFlag[];
}

export interface ActionItem {
  id: string;
  ownerSpeaker: SpeakerID;
  ownerName: string;
  taskText: string;
  deadlineHint: string | null;
  sourceNodeId: string;
  createdAt: number;
}

export interface QAEntry {
  id: string;
  question: string;
  answer: string;
  citationNodeIds: string[];
  latencyMs: number;
  askedAt: number;
}

export interface GraphNode {
  id: string;
  speakerId: SpeakerID;
  speakerName: string;
  text: string;
  timestamp: number;
  eventFlags: EventFlagType[];
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  type: "semantic" | "turn" | "reference";
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface AudioLevels {
  SPK_0: number; // 0.0 – 1.0
  SPK_1: number;
}

export interface SpeakerNames {
  SPK_0: string;
  SPK_1: string;
}

// ─── Slice Interfaces ─────────────────────────────────────────────────────────

interface SessionSlice {
  session: Session;
  setSession: (session: Partial<Session>) => void;
  resetSession: () => void;
}

interface SpeakerSlice {
  speakerNames: SpeakerNames;
  setSpeakerName: (speakerId: SpeakerID, name: string) => void;
}

interface TranscriptSlice {
  utterances: Utterance[];
  highlightedNodeId: string | null;
  appendUtterance: (utterance: Utterance) => void;
  setHighlightedNodeId: (nodeId: string | null) => void;
  clearTranscript: () => void;
}

interface ActionItemSlice {
  actionItems: ActionItem[];
  appendActionItem: (item: ActionItem) => void;
  clearActionItems: () => void;
}

interface QASlice {
  qaHistory: QAEntry[];
  isQALoading: boolean;
  appendQAEntry: (entry: QAEntry) => void;
  setQALoading: (loading: boolean) => void;
  clearQAHistory: () => void;
}

interface GraphSlice {
  graphData: GraphData;
  setGraphData: (data: GraphData) => void;
  appendGraphNode: (node: GraphNode) => void;
  appendGraphEdge: (edge: GraphEdge) => void;
  clearGraph: () => void;
}

interface AudioSlice {
  audioLevels: AudioLevels;
  setAudioLevel: (speakerId: SpeakerID, level: number) => void;
}

interface UISlice {
  isQADrawerOpen: boolean;
  isActionPanelOpen: boolean;
  isGraphViewOpen: boolean;
  isSpeakerModalOpen: boolean;
  lastPipelineLatencyMs: number | null;
  setQADrawerOpen: (open: boolean) => void;
  setActionPanelOpen: (open: boolean) => void;
  setGraphViewOpen: (open: boolean) => void;
  setSpeakerModalOpen: (open: boolean) => void;
  setLastPipelineLatency: (ms: number) => void;
}

// ─── Default Values ───────────────────────────────────────────────────────────

const defaultSession: Session = {
  id: "",
  name: "",
  status: "idle",
  startedAt: null,
  endedAt: null,
};

const defaultSpeakerNames: SpeakerNames = {
  SPK_0: "Speaker A",
  SPK_1: "Speaker B",
};

const defaultAudioLevels: AudioLevels = {
  SPK_0: 0,
  SPK_1: 0,
};

const defaultGraphData: GraphData = {
  nodes: [],
  edges: [],
};

// ─── Store ────────────────────────────────────────────────────────────────────

type StoreState = SessionSlice &
  SpeakerSlice &
  TranscriptSlice &
  ActionItemSlice &
  QASlice &
  GraphSlice &
  AudioSlice &
  UISlice;

export const useStore = create<StoreState>()((set) => ({
  // ── Session ──────────────────────────────────────────────────────────────
  session: defaultSession,

  setSession: (partial) =>
    set((state) => ({ session: { ...state.session, ...partial } })),

  resetSession: () =>
    set({
      session: defaultSession,
      utterances: [],
      actionItems: [],
      qaHistory: [],
      graphData: defaultGraphData,
      highlightedNodeId: null,
      lastPipelineLatencyMs: null,
    }),

  // ── Speakers ─────────────────────────────────────────────────────────────
  speakerNames: defaultSpeakerNames,

  setSpeakerName: (speakerId, name) =>
    set((state) => ({
      speakerNames: { ...state.speakerNames, [speakerId]: name },
    })),

  // ── Transcript ────────────────────────────────────────────────────────────
  utterances: [],
  highlightedNodeId: null,

  appendUtterance: (utterance) =>
    set((state) => ({ utterances: [...state.utterances, utterance] })),

  setHighlightedNodeId: (nodeId) => set({ highlightedNodeId: nodeId }),

  clearTranscript: () => set({ utterances: [], highlightedNodeId: null }),

  // ── Action Items ──────────────────────────────────────────────────────────
  actionItems: [],

  appendActionItem: (item) =>
    set((state) => ({ actionItems: [...state.actionItems, item] })),

  clearActionItems: () => set({ actionItems: [] }),

  // ── Q&A ───────────────────────────────────────────────────────────────────
  qaHistory: [],
  isQALoading: false,

  appendQAEntry: (entry) =>
    set((state) => ({ qaHistory: [...state.qaHistory, entry] })),

  setQALoading: (loading) => set({ isQALoading: loading }),

  clearQAHistory: () => set({ qaHistory: [], isQALoading: false }),

  // ── Graph ─────────────────────────────────────────────────────────────────
  graphData: defaultGraphData,

  setGraphData: (data) => set({ graphData: data }),

  appendGraphNode: (node) =>
    set((state) => ({
      graphData: {
        ...state.graphData,
        nodes: [...state.graphData.nodes, node],
      },
    })),

  appendGraphEdge: (edge) =>
    set((state) => ({
      graphData: {
        ...state.graphData,
        edges: [...state.graphData.edges, edge],
      },
    })),

  clearGraph: () => set({ graphData: defaultGraphData }),

  // ── Audio Levels ──────────────────────────────────────────────────────────
  audioLevels: defaultAudioLevels,

  setAudioLevel: (speakerId, level) =>
    set((state) => ({
      audioLevels: { ...state.audioLevels, [speakerId]: level },
    })),

  // ── UI State ──────────────────────────────────────────────────────────────
  isQADrawerOpen: false,
  isActionPanelOpen: true,
  isGraphViewOpen: false,
  isSpeakerModalOpen: false,
  lastPipelineLatencyMs: null,

  setQADrawerOpen: (open) => set({ isQADrawerOpen: open }),
  setActionPanelOpen: (open) => set({ isActionPanelOpen: open }),
  setGraphViewOpen: (open) => set({ isGraphViewOpen: open }),
  setSpeakerModalOpen: (open) => set({ isSpeakerModalOpen: open }),
  setLastPipelineLatency: (ms) => set({ lastPipelineLatencyMs: ms }),
}));

// ─── Convenience Selectors ────────────────────────────────────────────────────
// Use these in components instead of selecting the whole store.
// e.g. const utterances = useUtterances();

export const useSession = () => useStore((s) => s.session);
export const useSessionActions = () =>
  useStore((s) => ({ setSession: s.setSession, resetSession: s.resetSession }));

export const useSpeakerNames = () => useStore((s) => s.speakerNames);
export const useSetSpeakerName = () => useStore((s) => s.setSpeakerName);

export const useUtterances = () => useStore((s) => s.utterances);
export const useHighlightedNodeId = () => useStore((s) => s.highlightedNodeId);
export const useTranscriptActions = () =>
  useStore((s) => ({
    appendUtterance: s.appendUtterance,
    setHighlightedNodeId: s.setHighlightedNodeId,
    clearTranscript: s.clearTranscript,
  }));

export const useActionItems = () => useStore((s) => s.actionItems);
export const useAppendActionItem = () => useStore((s) => s.appendActionItem);

export const useQAHistory = () => useStore((s) => s.qaHistory);
export const useIsQALoading = () => useStore((s) => s.isQALoading);
export const useQAActions = () =>
  useStore((s) => ({
    appendQAEntry: s.appendQAEntry,
    setQALoading: s.setQALoading,
    clearQAHistory: s.clearQAHistory,
  }));

export const useGraphData = () => useStore((s) => s.graphData);
export const useGraphActions = () =>
  useStore((s) => ({
    setGraphData: s.setGraphData,
    appendGraphNode: s.appendGraphNode,
    appendGraphEdge: s.appendGraphEdge,
    clearGraph: s.clearGraph,
  }));

export const useAudioLevels = () => useStore((s) => s.audioLevels);
export const useSetAudioLevel = () => useStore((s) => s.setAudioLevel);

export const useUIState = () =>
  useStore((s) => ({
    isQADrawerOpen: s.isQADrawerOpen,
    isActionPanelOpen: s.isActionPanelOpen,
    isGraphViewOpen: s.isGraphViewOpen,
    isSpeakerModalOpen: s.isSpeakerModalOpen,
    lastPipelineLatencyMs: s.lastPipelineLatencyMs,
  }));

export const useUIActions = () =>
  useStore((s) => ({
    setQADrawerOpen: s.setQADrawerOpen,
    setActionPanelOpen: s.setActionPanelOpen,
    setGraphViewOpen: s.setGraphViewOpen,
    setSpeakerModalOpen: s.setSpeakerModalOpen,
    setLastPipelineLatency: s.setLastPipelineLatency,
  }));