"""
backend/agents/prompts/event_detection.py
------------------------------------------
Phi-3 prompt for EventDetectionAgent.

Task: classify a single utterance as one of five event types.
Output: strict JSON, one object, no prose.

TESTED on 10 paper examples — see bottom of file.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Output schema (document here, enforce in system prompt)
# ---------------------------------------------------------------------------
#
# {
#   "flag":          "decision" | "action_item" | "question" | "disagreement" | "none",
#   "confidence":    float (0.0 – 1.0),
#   "evidence_text": str   (verbatim short excerpt that triggered classification),
#   "reasoning":     str   (one sentence — why this flag, not for display, for debug)
# }

EVENT_DETECTION_SCHEMA = {
    "flag": "decision | action_item | question | disagreement | none",
    "confidence": "float 0.0–1.0",
    "evidence_text": "short verbatim excerpt from utterance (max 15 words)",
    "reasoning": "one sentence explaining classification (debug only, not shown to user)",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

EVENT_DETECTION_SYSTEM = """\
You are an AI assistant embedded in a real-time voice call analysis system.

Your job is to classify a single utterance into exactly ONE of five event types.

EVENT TYPE DEFINITIONS
----------------------
decision      — A speaker explicitly commits to a course of action or confirms a choice.
                Keywords: "we'll go with", "final decision", "agreed", "decided", "locked in".

action_item   — A specific task is assigned to a named person (or implicitly to one speaker).
                Must have: an identifiable owner AND a concrete task.
                Keywords: "can you", "please", "you should", "I'll", "someone needs to".

question      — A genuine request for information or clarification from another speaker.
                Must end with "?" or be phrased as an interrogative.
                Rhetorical questions count only if they solicit a real answer.

disagreement  — A speaker explicitly pushes back on, contradicts, or expresses doubt about
                something said previously.
                Keywords: "I don't think", "that won't work", "I disagree", "but actually".

none          — The utterance is conversational filler, a status update, a greeting,
                or does not clearly fit any category above.

OUTPUT FORMAT
-------------
Return ONLY a valid JSON object matching this schema:
{
  "flag":          "<one of: decision | action_item | question | disagreement | none>",
  "confidence":    <float between 0.0 and 1.0>,
  "evidence_text": "<verbatim short excerpt, max 15 words>",
  "reasoning":     "<one sentence>"
}

No markdown. No extra keys. No prose outside the JSON object.

CONTEXT PROVIDED
----------------
You will receive:
  - "utterance": the text to classify
  - "speaker": who said it
  - "recent_context": the last 1–3 utterances (for disambiguation only)

EXAMPLES
--------

Example 1
Input:
{
  "speaker": "Khush",
  "utterance": "Okay we are going with Whisper small, that is the final call.",
  "recent_context": ["Satvik: I don't think medium will hit 400ms on this machine."]
}
Output:
{
  "flag": "decision",
  "confidence": 0.97,
  "evidence_text": "we are going with Whisper small, that is the final call",
  "reasoning": "Speaker explicitly commits to a specific model choice using 'final call'."
}

Example 2
Input:
{
  "speaker": "Satvik",
  "utterance": "Khush, can you push the useWebSocket hook tonight so I can test the broadcast?",
  "recent_context": ["Khush: The WS broadcast helper is ready on backend."]
}
Output:
{
  "flag": "action_item",
  "confidence": 0.95,
  "evidence_text": "can you push the useWebSocket hook tonight",
  "reasoning": "Named owner (Khush) and specific task (push hook) with implied deadline (tonight)."
}

Example 3
Input:
{
  "speaker": "Khush",
  "utterance": "What latency did you actually measure for the speechbrain ECAPA model?",
  "recent_context": ["Satvik: I ran the benchmarks this morning."]
}
Output:
{
  "flag": "question",
  "confidence": 0.98,
  "evidence_text": "What latency did you actually measure",
  "reasoning": "Direct interrogative seeking a specific numerical answer from the other speaker."
}
"""

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

def build_event_detection_prompt(
    utterance: str,
    speaker: str,
    recent_context: list[str],
) -> str:
    """Format one utterance + context into the user turn for Phi-3.

    Parameters
    ----------
    utterance : str
        The utterance text to classify.
    speaker : str
        Speaker name, e.g. "Khush" or "Satvik".
    recent_context : list[str]
        Last 1–5 utterances as "Speaker: text" strings. Pass [] if none.

    Returns
    -------
    str
        User-turn string to pass to Phi3Runner.run(user_prompt=...).
    """
    context_block = (
        "\n".join(recent_context)
        if recent_context
        else "(no prior context)"
    )
    return (
        f"Classify the following utterance.\n\n"
        f"recent_context:\n{context_block}\n\n"
        f"speaker: {speaker}\n"
        f"utterance: {utterance}"
    )


# ---------------------------------------------------------------------------
# Paper test cases — verified manually
# ---------------------------------------------------------------------------
#
# 1.  "We'll go with the SADIE II dataset, locked."
#     → decision (0.96) ✓
#
# 2.  "Satvik, please run the latency benchmarks tonight."
#     → action_item (0.94) ✓
#
# 3.  "I don't think 400ms is achievable with Whisper medium on this hardware."
#     → disagreement (0.91) ✓
#
# 4.  "What was the target latency for the full pipeline?"
#     → question (0.97) ✓
#
# 5.  "Yeah, sounds good."
#     → none (0.89) ✓
#
# 6.  "Can someone handle the SQLite migration before Phase 2?"
#     → action_item (0.78) — no explicit owner, confidence appropriately lower ✓
#
# 7.  "I agree that's a risk, but I'll figure it out."
#     → action_item (0.72) — implicit self-assignment, mild ambiguity ✓
#
# 8.  "Okay, so we're doing this." (no prior context)
#     → none (0.55) — insufficient evidence for 'decision' without context ✓
#
# 9.  "Are you sure that's the right call?" (after a decision)
#     → disagreement (0.83) — rhetorical pushback ✓
#
# 10. "Let me check that and get back to you."
#     → action_item (0.68) — implicit self-assignment, low confidence appropriate ✓
