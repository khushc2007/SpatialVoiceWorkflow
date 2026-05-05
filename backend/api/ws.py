"""
backend/api/ws.py
-----------------
WebSocket route for SpatialVoiceAI.

Connection registry: dict[session_id -> set[WebSocket]]
Allows multiple clients (e.g. two browser tabs) to subscribe to the same session.

TEAMMATE: Implement the three TODO sections below.
  1. _register / _unregister helpers
  2. broadcast() — serialises event dict and sends to all sockets in a session
  3. websocket_endpoint() — accept, register, keep-alive loop, unregister on disconnect
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Connection registry
# ---------------------------------------------------------------------------
# Maps  session_id (str)  →  set of active WebSocket connections
# Multiple browser tabs can connect to the same session simultaneously.
# Access is single-threaded (asyncio event loop), so no lock is needed.

_registry: Dict[str, Set[WebSocket]] = {}


# ---------------------------------------------------------------------------
# Internal helpers  (TEAMMATE: implement these)
# ---------------------------------------------------------------------------

def _register(session_id: str, ws: WebSocket) -> None:
    """Add *ws* to the registry for *session_id*.

    TODO: Create the set if this is the first connection for the session.
    """
    raise NotImplementedError


def _unregister(session_id: str, ws: WebSocket) -> None:
    """Remove *ws* from the registry.

    TODO: If the set becomes empty after removal, delete the session key
    so the registry doesn't accumulate stale entries.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Public broadcast helper  (TEAMMATE: implement this)
# ---------------------------------------------------------------------------

async def broadcast(session_id: str, event: dict) -> None:
    """Serialise *event* to JSON and send to every connected client for *session_id*.

    Called from NLPEngine (via asyncio.run_coroutine_threadsafe) and from
    FastAPI route handlers (directly with await).

    TODO:
      - Serialise event with json.dumps
      - Iterate over _registry.get(session_id, set()) — iterate over a COPY
        (set(…)) so removal during iteration doesn't raise RuntimeError
      - Call await ws.send_text(payload) inside a try/except; on any
        exception log a warning and call _unregister(session_id, ws)
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Route  (TEAMMATE: implement this)
# ---------------------------------------------------------------------------

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """Accept a WebSocket connection and keep it alive until the client disconnects.

    TODO:
      1. await websocket.accept()
      2. _register(session_id, websocket)
      3. Log: f"WS client connected — session {session_id}"
      4. Enter a keep-alive loop:
           while True:
               data = await websocket.receive_text()
               # Clients don't send data in normal usage, but handling
               # keep-alive pings here prevents the connection timing out.
               # Optionally echo: await websocket.send_text(data)
      5. Catch WebSocketDisconnect:
           _unregister(session_id, websocket)
           Log: f"WS client disconnected — session {session_id}"
    """
    raise NotImplementedError
