"""
backend/agents/coordinator.py
------------------------------
AgentCoordinator — routes utterance events through the agent pipeline.

Architecture: thin tool-chaining dispatcher. No LangChain. No AutoGen.
Pure Python, deterministic, debuggable.

Pipeline per utterance:
  1. TranscriptionAgent  — already called by NLPEngine before coordinator
  2. EventDetectionAgent — classify utterance → flag
  3. ActionItemAgent     — only if flag == "action_item"
  4. WebSocket broadcast — utterance + event_flag + action_item events

Q&A is demand-driven (POST /qa), not part of the utterance pipeline.
QAAgent is held here and called directly by the /qa route handler.

TEAMMATE: Implement the TODO sections.
  - __init__:            wire in all agent instances + ws broadcast coroutine
  - process_utterance(): run the pipeline, return broadcast-ready event dicts
  - All agents are already implemented — just call their methods.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Coroutine, Dict, List, Optional

from config import cfg
from models.events import (
    ActionItemPayload,
    AudioLevelPayload,
    EventFlagPayload,
    EventType,
    FlagType,
    QAResponsePayload,
    SessionEndPayload,
    UtterancePayload,
    WSEvent,
)

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """Routes utterance data through the multi-agent pipeline.

    Instantiated once at server startup (FastAPI lifespan).
    Shared across all active sessions.

    Parameters
    ----------
    phi3_runner : Phi3Runner
        Pre-warmed Phi-3 inference wrapper.
    graph_builder : GraphBuilder
        Live conversation graph. Shared instance.
    db_session_utils : module
        session_utils module for SQLite writes.
    broadcast_fn : async callable
        broadcast(session_id, event_dict) coroutine from api/ws.py.
        Coordinator calls this to push events to connected clients.
    loop : asyncio.AbstractEventLoop
        The asyncio event loop running the FastAPI server.
        Used by the NLPEngine thread to schedule broadcast coroutines.

    Tool registry
    -------------
    _tools maps event flag strings to handler methods.
    Add new agents here without changing process_utterance().

    Example
    -------
    >>> coordinator = AgentCoordinator(
    ...     phi3_runner=runner,
    ...     graph_builder=graph,
    ...     db_session_utils=session_utils,
    ...     broadcast_fn=broadcast,
    ...     loop=asyncio.get_event_loop(),
    ... )
    >>> await coordinator.process_utterance(
    ...     session_id="uuid",
    ...     speaker_id="SPK_0",
    ...     speaker_name="Khush",
    ...     text="Let's go with Whisper small.",
    ...     confidence=0.94,
    ...     timestamp=142.3,
    ... )
    """

    def __init__(
        self,
        phi3_runner: Any,
        graph_builder: Any,
        db_session_utils: Any,
        broadcast_fn: Callable[..., Coroutine],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._phi3 = phi3_runner
        self._graph = graph_builder
        self._db = db_session_utils
        self._broadcast = broadcast_fn
        self._loop = loop

        # Lazy import to avoid circular deps — agents import from models/prompts only
        from agents.event_agent import EventDetectionAgent
        from agents.action_agent import ActionItemAgent
        from agents.qa_agent import QAAgent
        from agents.transcription_agent import TranscriptionAgent

        self._transcription_agent = TranscriptionAgent(graph_builder=self._graph)
        self._event_agent = EventDetectionAgent(phi3_runner=self._phi3)
        self._action_agent = ActionItemAgent(
            phi3_runner=self._phi3,
            db_session_utils=self._db,
        )
        self._qa_agent = QAAgent(
            phi3_runner=self._phi3,
            graph_builder=self._graph,
        )

        # Tool registry: flag value → handler coroutine
        # Add new flag handlers here; process_utterance() calls them automatically.
        self._tools: Dict[str, Callable] = {
            FlagType.ACTION_ITEM: self._handle_action_item,
            # Future: FlagType.DECISION: self._handle_decision,
        }

        logger.info("AgentCoordinator initialised with %d tools", len(self._tools))

    # ------------------------------------------------------------------
    # Main pipeline entry point
    # Called by NLPEngine thread for every transcribed utterance.
    # ------------------------------------------------------------------

    async def process_utterance(
        self,
        session_id: str,
        speaker_id: str,
        speaker_name: str,
        text: str,
        confidence: float,
        timestamp: float,
    ) -> None:
        """Run the full per-utterance agent pipeline and broadcast results.

        Steps:
          1. Add utterance node to graph (TranscriptionAgent)
          2. Run EventDetectionAgent
          3. If flag is non-none → broadcast event_flag event
          4. Run flag-specific tool handler (e.g. ActionItemAgent)
          5. Broadcast utterance event with event_flags attached

        All broadcast calls are fire-and-forget. Errors are logged, not raised.

        TEAMMATE: TODO — implement this method body.
        The skeleton below shows the intended call sequence.
        """

        # TODO:
        #
        # Step 1 — write utterance node to graph
        # node = self._transcription_agent.write_node(
        #     session_id=session_id,
        #     speaker_id=speaker_id,
        #     speaker_name=speaker_name,
        #     text=text,
        #     confidence=confidence,
        #     timestamp=timestamp,
        # )
        #
        # Step 2 — get recent context for EventDetectionAgent
        # recent_nodes = self._graph.get_recent_nodes(session_id, n=cfg.agent.EVENT_CONTEXT_NODES)
        # recent_context = [f"{n['speaker_name']}: {n['text']}" for n in recent_nodes]
        #
        # Step 3 — classify utterance
        # detection = self._event_agent.detect(
        #     utterance=text,
        #     speaker=speaker_name,
        #     recent_context=recent_context,
        # )
        # flag = detection["flag"]
        # event_flags = [] if flag == FlagType.NONE else [flag]
        #
        # Step 4 — broadcast event_flag if non-none
        # if flag != FlagType.NONE:
        #     flag_event = WSEvent(
        #         event=EventType.EVENT_FLAG,
        #         session_id=session_id,
        #         timestamp=time.time(),
        #         data=EventFlagPayload(
        #             node_id=node["id"],
        #             flag_type=flag,
        #             confidence=detection["confidence"],
        #             evidence_text=detection["evidence_text"],
        #         ),
        #     )
        #     await self._broadcast(session_id, flag_event.model_dump())
        #
        # Step 5 — run tool handler for this flag
        # handler = self._tools.get(flag)
        # if handler:
        #     await handler(session_id=session_id, node=node, detection=detection)
        #
        # Step 6 — broadcast utterance event
        # utterance_event = WSEvent(
        #     event=EventType.UTTERANCE,
        #     session_id=session_id,
        #     timestamp=time.time(),
        #     data=UtterancePayload(
        #         node_id=node["id"],
        #         speaker_id=speaker_id,
        #         speaker_name=speaker_name,
        #         text=text,
        #         confidence=confidence,
        #         event_flags=event_flags,
        #     ),
        # )
        # await self._broadcast(session_id, utterance_event.model_dump())

        raise NotImplementedError("Teammate: implement process_utterance() body")

    # ------------------------------------------------------------------
    # Tool handlers — called by process_utterance() via _tools registry
    # ------------------------------------------------------------------

    async def _handle_action_item(
        self,
        session_id: str,
        node: Dict[str, Any],
        detection: Dict[str, Any],
    ) -> None:
        """Extract action item fields and broadcast to frontend.

        TEAMMATE: TODO — implement this method body.
        """
        # TODO:
        #
        # result = self._action_agent.extract(
        #     utterance=node["text"],
        #     speaker=node["speaker_name"],
        #     other_speaker=<resolve other speaker name from session>,
        #     recent_context=<last 3 utterances>,
        # )
        # if result.get("error"):
        #     return
        #
        # item_id = str(uuid.uuid4())
        # await self._db.insert_action_item(
        #     session_id=session_id,
        #     id=item_id,
        #     owner_speaker=result["owner_speaker"],
        #     task_text=result["task_text"],
        #     deadline_hint=result.get("deadline_hint"),
        #     source_node_id=node["id"],
        # )
        #
        # action_event = WSEvent(
        #     event=EventType.ACTION_ITEM,
        #     session_id=session_id,
        #     timestamp=time.time(),
        #     data=ActionItemPayload(
        #         id=item_id,
        #         owner_speaker=result["owner_speaker"],
        #         task_text=result["task_text"],
        #         deadline_hint=result.get("deadline_hint"),
        #         source_node_id=node["id"],
        #     ),
        # )
        # await self._broadcast(session_id, action_event.model_dump())

        raise NotImplementedError("Teammate: implement _handle_action_item()")

    # ------------------------------------------------------------------
    # Q&A  — called directly by POST /qa route handler
    # ------------------------------------------------------------------

    async def answer_question(
        self,
        session_id: str,
        question: str,
    ) -> Dict[str, Any]:
        """Run QAAgent and return a broadcast-ready QAResponsePayload dict.

        Called by api/qa.py route handler. Broadcasts qa_response event
        AND returns the payload so the HTTP response can echo it.

        TEAMMATE: TODO — implement this method body.
        """
        # TODO:
        #
        # t0 = time.perf_counter()
        # result = self._qa_agent.answer(
        #     question=question,
        #     session_id=session_id,
        # )
        # latency_ms = int((time.perf_counter() - t0) * 1000)
        #
        # qa_event = WSEvent(
        #     event=EventType.QA_RESPONSE,
        #     session_id=session_id,
        #     timestamp=time.time(),
        #     data=QAResponsePayload(
        #         question=question,
        #         answer=result["answer"],
        #         citation_node_ids=result["citation_node_ids"],
        #         latency_ms=latency_ms,
        #     ),
        # )
        # await self._broadcast(session_id, qa_event.model_dump())
        # return qa_event.data.model_dump()

        raise NotImplementedError("Teammate: implement answer_question()")

    # ------------------------------------------------------------------
    # Async bridge  — called from NLPEngine thread (non-async context)
    # ------------------------------------------------------------------

    def schedule_utterance(
        self,
        session_id: str,
        speaker_id: str,
        speaker_name: str,
        text: str,
        confidence: float,
        timestamp: float,
    ) -> None:
        """Thread-safe bridge: schedules process_utterance() on the asyncio loop.

        NLPEngine runs in a daemon thread and cannot directly await coroutines.
        This method uses run_coroutine_threadsafe to hand off to the event loop.

        Called by NLPEngine after every successful ASR + diarization result.
        """
        asyncio.run_coroutine_threadsafe(
            self.process_utterance(
                session_id=session_id,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                text=text,
                confidence=confidence,
                timestamp=timestamp,
            ),
            self._loop,
        )
