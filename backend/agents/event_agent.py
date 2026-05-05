"""
backend/agents/event_agent.py
------------------------------
EventDetectionAgent — classifies every utterance node as one of five event types.

Fires on every utterance. Uses Phi-3 with a 3-shot prompt.
Returns a structured result dict. Coordinator reads flag and routes accordingly.

Dependencies:
  - Phi3Runner (llm/phi3_runner.py)
  - build_event_detection_prompt (agents/prompts/event_detection.py)
  - FlagType enum (models/events.py)
  - cfg (config.py)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config import cfg
from models.events import FlagType
from agents.prompts.event_detection import (
    EVENT_DETECTION_SYSTEM,
    build_event_detection_prompt,
)

logger = logging.getLogger(__name__)

# Sentinel result returned when Phi-3 fails or returns an invalid flag.
# Coordinator treats this as flag = "none" and does not route further.
DETECTION_ERROR: Dict[str, Any] = {
    "flag": FlagType.NONE,
    "confidence": 0.0,
    "evidence_text": "",
    "reasoning": "EventDetectionAgent: inference failed or invalid output",
    "error": True,
}

# Accepted flag values — any Phi-3 output outside this set is rejected.
_VALID_FLAGS = {f.value for f in FlagType}


class EventDetectionAgent:
    """Classifies a single utterance into one of five event types.

    Instantiated once at startup by AgentCoordinator. Shared across all sessions.
    Thread-safe because Phi3Runner.run() is synchronous and stateless.

    Parameters
    ----------
    phi3_runner : Phi3Runner
        Pre-warmed Phi-3 inference wrapper. Injected by AgentCoordinator.

    Example
    -------
    >>> agent = EventDetectionAgent(phi3_runner=runner)
    >>> result = agent.detect(
    ...     utterance="Let's go with Whisper small, final decision.",
    ...     speaker="Khush",
    ...     recent_context=["Satvik: Medium won't hit 400ms."],
    ... )
    >>> result["flag"]
    'decision'
    """

    def __init__(self, phi3_runner: Any) -> None:
        self._runner = phi3_runner
        logger.info("EventDetectionAgent initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        utterance: str,
        speaker: str,
        recent_context: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Classify a single utterance.

        Parameters
        ----------
        utterance : str
            The transcript text to classify.
        speaker : str
            Human-assigned speaker name, e.g. "Khush".
        recent_context : list[str] | None
            Last N utterances as "Speaker: text" strings, most recent last.
            Pass None or [] if this is the first utterance of the session.
            N is bounded by cfg.agent.EVENT_CONTEXT_NODES.

        Returns
        -------
        dict with keys:
            flag          : str  — one of FlagType values
            confidence    : float
            evidence_text : str
            reasoning     : str  (debug only)
            error         : bool (only present on failure)
        """
        context = list(recent_context or [])[-cfg.agent.EVENT_CONTEXT_NODES:]

        user_prompt = build_event_detection_prompt(
            utterance=utterance,
            speaker=speaker,
            recent_context=context,
        )

        raw = self._runner.run(
            system_prompt=EVENT_DETECTION_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=128,  # classification output is short
        )

        # Phi3Runner already returns ERROR_RESPONSE on inference failure
        if raw.get("error"):
            logger.warning(
                "EventDetectionAgent: phi3_runner returned error for utterance: %r",
                utterance[:60],
            )
            return DETECTION_ERROR

        return self._validate(raw, utterance)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, raw: Dict[str, Any], utterance: str) -> Dict[str, Any]:
        """Validate and normalise the parsed JSON from Phi-3.

        Rejects outputs with missing required keys or invalid flag values.
        Falls back to DETECTION_ERROR so the coordinator always gets a safe dict.
        """
        flag = raw.get("flag", "")
        if flag not in _VALID_FLAGS:
            logger.warning(
                "EventDetectionAgent: invalid flag %r for utterance %r — defaulting to none",
                flag,
                utterance[:60],
            )
            return {**DETECTION_ERROR, "reasoning": f"Invalid flag value: {flag!r}"}

        confidence = float(raw.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]

        return {
            "flag": flag,
            "confidence": confidence,
            "evidence_text": str(raw.get("evidence_text", ""))[:200],
            "reasoning": str(raw.get("reasoning", ""))[:300],
        }
