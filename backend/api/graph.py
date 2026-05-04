"""
api/graph.py — Conversation graph endpoint.

Routes
------
GET /graph/{session_id}   Return the full serialised graph (nodes + edges)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/graph", tags=["graph"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class NodeModel(BaseModel):
    """Serialised graph node (= utterance)."""

    id: str
    speaker_id: str
    speaker_name: str
    text: str
    timestamp: float
    confidence: float
    event_flags: list[str] = []
    # Embedding intentionally omitted from HTTP response (too large).
    # Returned only when individual node is fetched for Q&A citation highlighting.


class EdgeModel(BaseModel):
    """Serialised directed graph edge."""

    id: str
    source: str = Field(description="Source node UUID")
    target: str = Field(description="Target node UUID")
    weight: float = Field(ge=0.0, le=1.0, description="Cosine similarity")
    edge_type: str = Field(description='"semantic" | "turn" | "reference"')


class GraphResponse(BaseModel):
    """Full session graph payload."""

    session_id: str
    nodes: list[NodeModel]
    edges: list[EdgeModel]
    node_count: int
    edge_count: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/{session_id}",
    response_model=GraphResponse,
    summary="Fetch full conversation graph for a session",
)
async def get_graph(session_id: str, request: Request) -> GraphResponse:
    """Return all utterance nodes and semantic/turn/reference edges.

    This endpoint is used by:
    - ``GraphView.tsx`` for force-directed visualization
    - Post-call summary page
    - Offline export / Samsung Notes integration

    TODO (backend):
        1. Pull ``db`` from ``request.app.state.db``.
        2. Verify session exists; raise 404 if not.
        3. Call ``await get_utterances(db, session_id)`` → nodes list.
        4. Call ``await get_edges(db, session_id)`` → edges list.
        5. Deserialise event_flags from JSON string to list.
        6. Return populated GraphResponse.
    """
    # STUB — empty graph
    return GraphResponse(
        session_id=session_id,
        nodes=[],
        edges=[],
        node_count=0,
        edge_count=0,
    )
