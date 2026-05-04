"""
db/session_utils.py — CRUD helpers for sessions, utterances, edges,
action_items, and qa_history.

All functions are async and expect an open aiosqlite.Connection passed in
(connection pooling is the caller's responsibility — see main.py lifespan).

Teammate: fill in SQL bodies where marked TODO.
Signatures and docstrings are locked — do not change them.
"""

from __future__ import annotations

import json
import logging
import uuid
from time import time
from typing import Any

import aiosqlite
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases (thin wrappers over plain dicts for IDE help)
# ---------------------------------------------------------------------------

SessionRow      = dict[str, Any]
UtteranceRow    = dict[str, Any]
EdgeRow         = dict[str, Any]
ActionItemRow   = dict[str, Any]
QAHistoryRow    = dict[str, Any]


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

async def create_session(
    db: aiosqlite.Connection,
    speaker_names: dict[str, str] | None = None,
) -> SessionRow:
    """Insert a new session row and return it.

    Args:
        db: Open aiosqlite connection.
        speaker_names: Optional mapping of speaker_id → display name,
                       e.g. ``{"SPK_0": "Alice", "SPK_1": "Bob"}``.

    Returns:
        Dict matching the ``sessions`` table schema with ``id`` populated.
    """
    session_id = str(uuid.uuid4())
    now = time()
    names_json = json.dumps(speaker_names or {})

    # TODO: INSERT INTO sessions (id, created_at, ended_at, speaker_names, meta)
    #       VALUES (?, ?, NULL, ?, '{}')
    raise NotImplementedError

    return {
        "id": session_id,
        "created_at": now,
        "ended_at": None,
        "speaker_names": speaker_names or {},
        "meta": {},
    }


async def get_session(
    db: aiosqlite.Connection,
    session_id: str,
) -> SessionRow | None:
    """Fetch a single session by id.

    Returns:
        Dict or ``None`` if not found.
    """
    # TODO: SELECT * FROM sessions WHERE id = ?
    raise NotImplementedError


async def end_session(
    db: aiosqlite.Connection,
    session_id: str,
) -> None:
    """Stamp ``ended_at`` on a session to mark it closed.

    Args:
        session_id: UUID of the session to close.
    """
    # TODO: UPDATE sessions SET ended_at = ? WHERE id = ?
    raise NotImplementedError


async def update_speaker_names(
    db: aiosqlite.Connection,
    session_id: str,
    speaker_names: dict[str, str],
) -> None:
    """Persist the user-assigned speaker name mapping.

    Args:
        speaker_names: Full replacement mapping ``{"SPK_0": "Alice", ...}``.
    """
    # TODO: UPDATE sessions SET speaker_names = ? WHERE id = ?
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Utterances
# ---------------------------------------------------------------------------

async def insert_utterance(
    db: aiosqlite.Connection,
    session_id: str,
    speaker_id: str,
    speaker_name: str,
    text: str,
    timestamp: float,
    confidence: float = 1.0,
    event_flags: list[str] | None = None,
    embedding: np.ndarray | None = None,
    audio_segment: str | None = None,
) -> UtteranceRow:
    """Insert a transcribed utterance (graph node) and return it.

    The embedding ndarray is stored as raw bytes (float32, C-order).
    Retrieve and restore with ``np.frombuffer(blob, dtype=np.float32)``.

    Args:
        session_id:     Parent session UUID.
        speaker_id:     ``"SPK_0"``, ``"SPK_1"``, etc.
        speaker_name:   Human-readable name assigned by user.
        text:           Whisper transcript text.
        timestamp:      Unix epoch of utterance start.
        confidence:     Whisper segment-level confidence (0–1).
        event_flags:    List of EventType strings, e.g. ``["action_item"]``.
        embedding:      384-dim MiniLM embedding as numpy float32 array.
        audio_segment:  Relative path to the saved 3-second WAV chunk.

    Returns:
        Dict matching the ``utterances`` table schema.
    """
    node_id = str(uuid.uuid4())
    flags_json = json.dumps(event_flags or [])
    embedding_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None

    # TODO: INSERT INTO utterances
    #       (id, session_id, speaker_id, speaker_name, text, timestamp,
    #        confidence, event_flags, embedding_blob, audio_segment)
    #       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    raise NotImplementedError

    return {
        "id": node_id,
        "session_id": session_id,
        "speaker_id": speaker_id,
        "speaker_name": speaker_name,
        "text": text,
        "timestamp": timestamp,
        "confidence": confidence,
        "event_flags": event_flags or [],
        "embedding_blob": embedding_blob,
        "audio_segment": audio_segment,
    }


async def get_utterances(
    db: aiosqlite.Connection,
    session_id: str,
    limit: int = 200,
) -> list[UtteranceRow]:
    """Return utterances for a session ordered by timestamp ascending.

    Args:
        limit: Cap at this many rows (most recent ``limit`` utterances).
    """
    # TODO: SELECT * FROM utterances WHERE session_id = ?
    #       ORDER BY timestamp ASC LIMIT ?
    raise NotImplementedError


async def get_utterance(
    db: aiosqlite.Connection,
    node_id: str,
) -> UtteranceRow | None:
    """Fetch a single utterance by its UUID.

    Returns:
        Dict or ``None`` if not found.
    """
    # TODO: SELECT * FROM utterances WHERE id = ?
    raise NotImplementedError


async def update_utterance_flags(
    db: aiosqlite.Connection,
    node_id: str,
    event_flags: list[str],
) -> None:
    """Overwrite the event_flags for a node after agent post-processing.

    Called by EventDetectionAgent once classification is complete.
    """
    # TODO: UPDATE utterances SET event_flags = ? WHERE id = ?
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

async def insert_edge(
    db: aiosqlite.Connection,
    session_id: str,
    source_node: str,
    target_node: str,
    weight: float,
    edge_type: str,
) -> EdgeRow:
    """Insert a directed edge between two utterance nodes.

    Args:
        source_node:    UUID of the source utterance.
        target_node:    UUID of the target utterance.
        weight:         Cosine similarity (0–1).
        edge_type:      One of ``"semantic"``, ``"turn"``, ``"reference"``.

    Returns:
        Dict matching the ``edges`` table schema.
    """
    edge_id = str(uuid.uuid4())

    # TODO: INSERT INTO edges (id, session_id, source_node, target_node, weight, edge_type)
    #       VALUES (?, ?, ?, ?, ?, ?)
    raise NotImplementedError

    return {
        "id": edge_id,
        "session_id": session_id,
        "source_node": source_node,
        "target_node": target_node,
        "weight": weight,
        "edge_type": edge_type,
    }


async def get_edges(
    db: aiosqlite.Connection,
    session_id: str,
) -> list[EdgeRow]:
    """Return all edges for a session.

    Args:
        session_id: Filter to edges belonging to this session.
    """
    # TODO: SELECT * FROM edges WHERE session_id = ?
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Action Items
# ---------------------------------------------------------------------------

async def insert_action_item(
    db: aiosqlite.Connection,
    session_id: str,
    source_node_id: str,
    owner_speaker: str,
    task_text: str,
    deadline_hint: str | None = None,
) -> ActionItemRow:
    """Persist an action item extracted by ActionItemAgent.

    Args:
        source_node_id: UUID of the utterance that triggered extraction.
        owner_speaker:  speaker_id or resolved speaker_name.
        task_text:      Extracted task description.
        deadline_hint:  Raw deadline string (``"by Friday"``) or ``None``.

    Returns:
        Dict matching the ``action_items`` table schema.
    """
    item_id = str(uuid.uuid4())
    now = time()

    # TODO: INSERT INTO action_items
    #       (id, session_id, source_node_id, owner_speaker, task_text, deadline_hint, created_at)
    #       VALUES (?, ?, ?, ?, ?, ?, ?)
    raise NotImplementedError

    return {
        "id": item_id,
        "session_id": session_id,
        "source_node_id": source_node_id,
        "owner_speaker": owner_speaker,
        "task_text": task_text,
        "deadline_hint": deadline_hint,
        "created_at": now,
    }


async def get_action_items(
    db: aiosqlite.Connection,
    session_id: str,
) -> list[ActionItemRow]:
    """Return all action items for a session ordered by creation time.

    Args:
        session_id: Filter to action items belonging to this session.
    """
    # TODO: SELECT * FROM action_items WHERE session_id = ?
    #       ORDER BY created_at ASC
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Q&A History
# ---------------------------------------------------------------------------

async def insert_qa(
    db: aiosqlite.Connection,
    session_id: str,
    question: str,
    answer: str,
    citation_nodes: list[str],
    latency_ms: float | None = None,
) -> QAHistoryRow:
    """Persist a Q&A exchange after QAAgent responds.

    Args:
        question:       The user's question text.
        answer:         Phi-3 generated answer.
        citation_nodes: List of utterance UUIDs cited in the answer.
        latency_ms:     End-to-end latency for the exchange.

    Returns:
        Dict matching the ``qa_history`` table schema.
    """
    qa_id = str(uuid.uuid4())
    now = time()
    citations_json = json.dumps(citation_nodes)

    # TODO: INSERT INTO qa_history
    #       (id, session_id, question, answer, citation_nodes, latency_ms, asked_at)
    #       VALUES (?, ?, ?, ?, ?, ?, ?)
    raise NotImplementedError

    return {
        "id": qa_id,
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "citation_nodes": citation_nodes,
        "latency_ms": latency_ms,
        "asked_at": now,
    }


async def get_qa_history(
    db: aiosqlite.Connection,
    session_id: str,
) -> list[QAHistoryRow]:
    """Return full Q&A history for a session ordered by time ascending.

    Args:
        session_id: Filter to Q&A exchanges belonging to this session.
    """
    # TODO: SELECT * FROM qa_history WHERE session_id = ?
    #       ORDER BY asked_at ASC
    raise NotImplementedError
