"""
api/qa.py — Q&A endpoint (HTTP fallback + history).

The primary Q&A path is over WebSocket (QARequestMessage → QAResponseEvent).
These HTTP endpoints exist for:
  - Direct API access / testing
  - Fetching historical Q&A after the session ends

Routes
------
POST /qa                              Submit a question (triggers QAAgent)
GET  /qa/{session_id}/history         Return full Q&A history for a session
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/qa", tags=["qa"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QARequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=1000)


class CitationNode(BaseModel):
    """Minimal node info attached to each cited utterance."""

    node_id: str
    speaker_name: str
    text: str
    timestamp: float


class QAResponse(BaseModel):
    question: str
    answer: str
    citation_nodes: list[CitationNode] = Field(
        description="Full node details for each cited utterance, "
                    "in the order they appear in the answer."
    )
    citation_node_ids: list[str] = Field(
        description="Plain list of node UUIDs for quick frontend highlighting."
    )
    latency_ms: float | None = None


class QAHistoryResponse(BaseModel):
    session_id: str
    exchanges: list[QAResponse]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=QAResponse,
    summary="Submit a question against the live conversation graph",
)
async def ask_question(body: QARequest, request: Request) -> QAResponse:
    """Invoke QAAgent synchronously and return the grounded answer.

    Flow:
        1. FAISS retrieves top-K utterance nodes most similar to the question.
        2. Phi-3 generates an answer grounded in those nodes.
        3. Response includes citation node IDs so the frontend can highlight
           the relevant utterance cards and graph nodes.

    TODO (backend):
        1. Pull ``coordinator`` from ``request.app.state.coordinator``.
        2. Call ``await coordinator.qa_agent.answer(body.session_id, body.question)``.
        3. Fetch full node details for each citation UUID.
        4. Persist to qa_history via ``insert_qa()``.
        5. Return real QAResponse.
    """
    # STUB
    return QAResponse(
        question=body.question,
        answer="TODO — QAAgent not yet wired.",
        citation_nodes=[],
        citation_node_ids=[],
        latency_ms=None,
    )


@router.get(
    "/{session_id}/history",
    response_model=QAHistoryResponse,
    summary="Fetch Q&A history for a session",
)
async def get_qa_history(session_id: str, request: Request) -> QAHistoryResponse:
    """Return all Q&A exchanges for the session ordered chronologically.

    Used by the post-call summary page to re-render the Q&A drawer.

    TODO (backend):
        1. Pull ``db`` from ``request.app.state.db``.
        2. Call ``await get_qa_history(db, session_id)``.
        3. For each exchange, resolve citation node details.
        4. Return populated QAHistoryResponse.
    """
    # STUB
    return QAHistoryResponse(session_id=session_id, exchanges=[])
