"""
api/session.py — Session lifecycle endpoints.

Routes
------
POST   /session/start              Create a new session
GET    /session/{session_id}       Fetch session metadata
POST   /session/{session_id}/end   End a live session
PATCH  /session/{session_id}/speakers  Update speaker name assignments
"""

from __future__ import annotations

from time import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/session", tags=["session"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class StartSessionRequest(BaseModel):
    """Body for POST /session/start."""

    speaker_names: dict[str, str] = Field(
        default_factory=dict,
        description='Map speaker_id → display name. E.g. {"SPK_0": "Alice"}. '
                    "Can be empty at call start and updated mid-call via PATCH.",
        examples=[{"SPK_0": "Alice", "SPK_1": "Bob"}],
    )


class SessionResponse(BaseModel):
    """Returned by start, get, and end endpoints."""

    id: str
    created_at: float
    ended_at: float | None
    speaker_names: dict[str, str]
    ws_url: str = Field(description="WebSocket URL for this session's live stream.")


class SessionSummaryResponse(BaseModel):
    """Richer payload returned by GET /session/{id} once the session has ended."""

    id: str
    created_at: float
    ended_at: float | None
    speaker_names: dict[str, str]
    total_utterances: int
    action_item_count: int
    graph_edge_count: int


class UpdateSpeakersRequest(BaseModel):
    """Body for PATCH /session/{id}/speakers."""

    speaker_names: dict[str, str] = Field(
        description='Full replacement map. E.g. {"SPK_0": "Alice", "SPK_1": "Bob"}.'
    )


class EndSessionResponse(BaseModel):
    ok: bool
    total_utterances: int
    action_item_count: int
    graph_edge_count: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/start",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new call session",
)
async def start_session(body: StartSessionRequest, request: Request) -> SessionResponse:
    """Create a session row in SQLite and return the session id + WebSocket URL.

    The WebSocket URL is constructed from the incoming request's base URL so
    this works behind a reverse proxy without extra config.

    TODO (backend):
        1. Pull ``db`` from ``request.app.state.db``.
        2. Call ``await create_session(db, body.speaker_names)``.
        3. Return real session data.
    """
    # STUB — returns hardcoded test data
    _fake_id = "00000000-0000-0000-0000-000000000001"
    _base = str(request.base_url).rstrip("/")
    _ws_url = _base.replace("http", "ws") + f"/ws/{_fake_id}"

    return SessionResponse(
        id=_fake_id,
        created_at=time(),
        ended_at=None,
        speaker_names=body.speaker_names,
        ws_url=_ws_url,
    )


@router.get(
    "/{session_id}",
    response_model=SessionSummaryResponse,
    summary="Fetch session metadata and stats",
)
async def get_session(session_id: str, request: Request) -> SessionSummaryResponse:
    """Return metadata for an existing session.

    If the session is still live, ``ended_at`` will be ``None`` and the
    utterance/edge counts reflect the current live state.

    TODO (backend):
        1. Pull ``db`` from ``request.app.state.db``.
        2. Call ``await get_session(db, session_id)``; 404 if None.
        3. Aggregate utterance + edge counts from DB.
    """
    # STUB
    if session_id != "00000000-0000-0000-0000-000000000001":
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionSummaryResponse(
        id=session_id,
        created_at=time() - 60,
        ended_at=None,
        speaker_names={"SPK_0": "Alice", "SPK_1": "Bob"},
        total_utterances=0,
        action_item_count=0,
        graph_edge_count=0,
    )


@router.post(
    "/{session_id}/end",
    response_model=EndSessionResponse,
    summary="End a live session",
)
async def end_session(session_id: str, request: Request) -> EndSessionResponse:
    """Stamp ``ended_at`` on the session and return final stats.

    Broadcasting the ``session_end`` WebSocket event is handled by the
    WebSocket router (ws.py) after the coordinator flushes any pending work.

    TODO (backend):
        1. Pull ``db`` from ``request.app.state.db``.
        2. Call ``await end_session(db, session_id)``; 404 if not found.
        3. Aggregate and return real counts.
        4. Trigger coordinator shutdown + WS session_end broadcast.
    """
    # STUB
    return EndSessionResponse(
        ok=True,
        total_utterances=0,
        action_item_count=0,
        graph_edge_count=0,
    )


@router.patch(
    "/{session_id}/speakers",
    response_model=SessionResponse,
    summary="Update speaker name assignments mid-call",
)
async def update_speakers(
    session_id: str,
    body: UpdateSpeakersRequest,
    request: Request,
) -> SessionResponse:
    """Allow the user to (re)assign display names to speaker IDs at any time.

    This fires a ``speakers_updated`` broadcast to all WS subscribers so the
    frontend can re-label orbs and transcript cards without a reload.

    TODO (backend):
        1. Pull ``db`` from ``request.app.state.db``.
        2. Call ``await update_speaker_names(db, session_id, body.speaker_names)``.
        3. Broadcast ``speakers_updated`` event over WS.
        4. Return updated session row.
    """
    # STUB
    _base = str(request.base_url).rstrip("/")
    _ws_url = _base.replace("http", "ws") + f"/ws/{session_id}"

    return SessionResponse(
        id=session_id,
        created_at=time() - 120,
        ended_at=None,
        speaker_names=body.speaker_names,
        ws_url=_ws_url,
    )
