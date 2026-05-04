"""
db/schema.py — SQLite schema for SpatialVoiceAI.

Tables
------
sessions        — one row per call session
utterances      — one row per transcribed utterance (graph node)
edges           — semantic / turn / reference edges between utterances
action_items    — extracted action items from ActionItemAgent
qa_history      — log of every Q&A exchange (question + answer + citations)

Initialise with:
    await init_db(db_path)

All tables use TEXT primary keys (UUIDs stored as strings).
Timestamps are REAL (Unix epoch, seconds).
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,           -- UUID
    created_at      REAL NOT NULL,              -- Unix epoch
    ended_at        REAL,                       -- NULL while live
    speaker_names   TEXT NOT NULL DEFAULT '{}', -- JSON: {"SPK_0": "Alice", ...}
    meta            TEXT NOT NULL DEFAULT '{}'  -- JSON blob for future use
);
"""

_CREATE_UTTERANCES = """
CREATE TABLE IF NOT EXISTS utterances (
    id              TEXT PRIMARY KEY,           -- UUID (= graph node id)
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    speaker_id      TEXT NOT NULL,              -- "SPK_0" | "SPK_1" | ...
    speaker_name    TEXT NOT NULL DEFAULT '',
    text            TEXT NOT NULL,
    timestamp       REAL NOT NULL,              -- Unix epoch of utterance start
    confidence      REAL NOT NULL DEFAULT 1.0,  -- Whisper segment confidence
    event_flags     TEXT NOT NULL DEFAULT '[]', -- JSON array of EventType strings
    embedding_blob  BLOB,                       -- 384-dim float32 numpy array (raw bytes)
    audio_segment   TEXT                        -- relative path to 3s WAV chunk
);
"""

_CREATE_EDGES = """
CREATE TABLE IF NOT EXISTS edges (
    id              TEXT PRIMARY KEY,           -- UUID
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source_node     TEXT NOT NULL REFERENCES utterances(id) ON DELETE CASCADE,
    target_node     TEXT NOT NULL REFERENCES utterances(id) ON DELETE CASCADE,
    weight          REAL NOT NULL DEFAULT 0.0,  -- cosine similarity (0–1)
    edge_type       TEXT NOT NULL               -- "semantic" | "turn" | "reference"
);
"""

_CREATE_ACTION_ITEMS = """
CREATE TABLE IF NOT EXISTS action_items (
    id              TEXT PRIMARY KEY,           -- UUID
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source_node_id  TEXT NOT NULL REFERENCES utterances(id) ON DELETE CASCADE,
    owner_speaker   TEXT NOT NULL,              -- speaker_id or speaker_name
    task_text       TEXT NOT NULL,
    deadline_hint   TEXT,                       -- raw string, e.g. "by Friday", NULL if absent
    created_at      REAL NOT NULL               -- Unix epoch
);
"""

_CREATE_QA_HISTORY = """
CREATE TABLE IF NOT EXISTS qa_history (
    id              TEXT PRIMARY KEY,           -- UUID
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    citation_nodes  TEXT NOT NULL DEFAULT '[]', -- JSON array of utterance UUIDs
    latency_ms      REAL,
    asked_at        REAL NOT NULL               -- Unix epoch
);
"""

# Fast lookup patterns
_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_utterances_session ON utterances(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_utterances_timestamp ON utterances(session_id, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_edges_session ON edges(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node);",
    "CREATE INDEX IF NOT EXISTS idx_action_items_session ON action_items(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_qa_history_session ON qa_history(session_id);",
]

_ALL_DDL: list[str] = [
    _CREATE_SESSIONS,
    _CREATE_UTTERANCES,
    _CREATE_EDGES,
    _CREATE_ACTION_ITEMS,
    _CREATE_QA_HISTORY,
    *_CREATE_INDEXES,
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def init_db(db_path: str | Path) -> None:
    """Create all tables (idempotent — safe to call on every startup).

    Args:
        db_path: Filesystem path to the SQLite file.  Parent directory must
                 exist.  The file is created if it does not exist.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Initialising database at %s", db_path)

    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA foreign_keys=ON;")
        for stmt in _ALL_DDL:
            await db.execute(stmt)
        await db.commit()

    logger.info("Database ready.")


async def drop_all(db_path: str | Path) -> None:
    """Drop all SpatialVoiceAI tables.  **Test / reset only — never call in prod.**"""
    async with aiosqlite.connect(db_path) as db:
        for table in ("qa_history", "action_items", "edges", "utterances", "sessions"):
            await db.execute(f"DROP TABLE IF EXISTS {table};")
        await db.commit()
    logger.warning("All tables dropped from %s", db_path)
