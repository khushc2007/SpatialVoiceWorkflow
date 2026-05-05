"""
backend/models/events.py
------------------------
Pydantic models for every WebSocket event SpatialVoiceAI emits.

All events share the same outer envelope (WSEvent).
The `data` field is typed as one of the concrete payload models below.

Wire order:
  AgentCoordinator → NLPEngine → broadcast() → frontend useWebSocket.ts

Frontend discriminates on the `event` field string.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Event type enum
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    UTTERANCE   = "utterance"
    EVENT_FLAG  = "event_flag"
    ACTION_ITEM = "action_item"
    QA_RESPONSE = "qa_response"
    SESSION_END = "session_end"
    AUDIO_LEVEL = "audio_level"


# ---------------------------------------------------------------------------
# Flag type enum  (mirrors EventDetectionAgent output)
# ---------------------------------------------------------------------------

class FlagType(str, Enum):
    DECISION     = "decision"
    ACTION_ITEM  = "action_item"
    QUESTION     = "question"
    DISAGREEMENT = "disagreement"
    NONE         = "none"


# ---------------------------------------------------------------------------
# Concrete payload models
# ---------------------------------------------------------------------------

class UtterancePayload(BaseModel):
    """Emitted by TranscriptionAgent after every 3-second ASR chunk."""
    node_id:      str   = Field(..., description="UUID of the graph node for this utterance")
    speaker_id:   str   = Field(..., description="SPK_0 or SPK_1")
    speaker_name: str   = Field(..., description="Human-assigned name, e.g. 'Khush'")
    text:         str   = Field(..., description="Whisper transcript text")
    confidence:   float = Field(..., ge=0.0, le=1.0, description="Whisper token-level confidence, 0–1")
    event_flags:  List[FlagType] = Field(default_factory=list,
                                         description="Flags detected on this utterance (may be empty)")


class EventFlagPayload(BaseModel):
    """Emitted by EventDetectionAgent when a non-none flag is detected."""
    node_id:       str      = Field(..., description="UUID of the utterance node being flagged")
    flag_type:     FlagType = Field(..., description="Detected flag type")
    confidence:    float    = Field(..., ge=0.0, le=1.0)
    evidence_text: str      = Field(..., description="Short excerpt from utterance that triggered the flag")


class ActionItemPayload(BaseModel):
    """Emitted by ActionItemAgent when an action_item flag is confirmed."""
    id:             str           = Field(..., description="UUID for this action item row in SQLite")
    owner_speaker:  str           = Field(..., description="Speaker name inferred as owner, e.g. 'Satvik'")
    task_text:      str           = Field(..., description="Extracted task description")
    deadline_hint:  Optional[str] = Field(None, description="Deadline string if mentioned, e.g. 'Friday'")
    source_node_id: str           = Field(..., description="UUID of the utterance this was extracted from")


class QAResponsePayload(BaseModel):
    """Emitted by QAAgent in response to a user question via POST /qa."""
    question:          str       = Field(..., description="The original question text")
    answer:            str       = Field(..., description="Phi-3 generated answer")
    citation_node_ids: List[str] = Field(..., description="Graph node UUIDs cited in the answer")
    latency_ms:        int       = Field(..., description="End-to-end latency: FAISS retrieval + Phi-3 generation")


class SessionEndPayload(BaseModel):
    """Emitted by session router when POST /session/{id}/end is called."""
    total_utterances:  int = Field(..., description="Total utterance nodes in final graph")
    action_item_count: int = Field(..., description="Total action items extracted during session")
    graph_edge_count:  int = Field(..., description="Total edges in final conversation graph")


class AudioLevelPayload(BaseModel):
    """Emitted by AudioEngine at ~30fps. Used to drive orb pulse animation on frontend."""
    SPK_0: float = Field(..., ge=0.0, le=1.0, description="Normalised RMS level for speaker 0")
    SPK_1: float = Field(..., ge=0.0, le=1.0, description="Normalised RMS level for speaker 1")


# ---------------------------------------------------------------------------
# Discriminated union of all payload types
# ---------------------------------------------------------------------------

AnyPayload = Union[
    UtterancePayload,
    EventFlagPayload,
    ActionItemPayload,
    QAResponsePayload,
    SessionEndPayload,
    AudioLevelPayload,
]


# ---------------------------------------------------------------------------
# Outer envelope  — every WS message uses this shape
# ---------------------------------------------------------------------------

class WSEvent(BaseModel):
    """
    Universal WebSocket event envelope.

    Every message broadcast to frontend clients has this structure:

        {
          "event":      "utterance",
          "session_id": "3fa85f64-...",
          "timestamp":  1716900000.123,
          "data":       { ... payload fields ... }
        }

    Frontend useWebSocket.ts discriminates on the `event` field and
    dispatches the matching Zustand action.

    Usage in Python:
        from backend.models.events import WSEvent, EventType, UtterancePayload

        event = WSEvent(
            event=EventType.UTTERANCE,
            session_id=session_id,
            timestamp=time.time(),
            data=UtterancePayload(
                node_id=str(node.id),
                speaker_id=node.speaker_id,
                speaker_name=node.speaker_name,
                text=node.text,
                confidence=node.confidence,
                event_flags=[],
            )
        )
        await broadcast(session_id, event.model_dump())
    """

    event:      EventType  = Field(..., description="Discriminator field — matches EventType enum")
    session_id: str        = Field(..., description="UUID of the active session")
    timestamp:  float      = Field(..., description="Unix timestamp (time.time()) at point of emission")
    data:       AnyPayload = Field(..., description="Event-specific payload — shape depends on `event`")
