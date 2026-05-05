"""
backend/agents/prompts/action_extraction.py
--------------------------------------------
Phi-3 prompt for ActionItemAgent.

Task: given a single utterance already classified as action_item,
extract the structured fields needed to write a row to SQLite.

Output: strict JSON, one object, no prose.
Only called when EventDetectionAgent returns flag == "action_item".
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
#
# {
#   "owner_speaker":  str   — name of the person who owns the task.
#                             Use the speaker name from context if explicitly
#                             named, or the other speaker's name if the
#                             utterance is directed at them ("can YOU do X").
#                             Use "unassigned" if genuinely ambiguous.
#   "task_text":      str   — concise description of the task (max 20 words).
#                             Must be a verb phrase, e.g. "Push the useWebSocket hook".
#   "deadline_hint":  str | null
#                           — explicit or strongly implied deadline as a string,
#                             e.g. "tonight", "Friday", "before the demo".
#                             null if no deadline is mentioned or implied.
#   "urgency":        "high" | "medium" | "low"
#                           — inferred from language cues.
#                             "high"   → "ASAP", "urgent", "before demo", "tonight"
#                             "medium" → "soon", "this week", "before next meeting"
#                             "low"    → no time pressure
# }

ACTION_EXTRACTION_SCHEMA = {
    "owner_speaker": "str — name of task owner, or 'unassigned'",
    "task_text": "str — verb phrase, max 20 words",
    "deadline_hint": "str | null — explicit deadline string or null",
    "urgency": "high | medium | low",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ACTION_EXTRACTION_SYSTEM = """\
You are an AI assistant embedded in a real-time voice call analysis system.

An utterance has already been classified as containing an action item.
Your job is to extract structured fields from that utterance.

EXTRACTION RULES
----------------
owner_speaker
  - If a person is explicitly named ("Khush, can you..."), they are the owner.
  - If the speaker says "I'll..." or "I will...", the owner is the speaker themselves.
  - If ownership is ambiguous, use "unassigned".
  - Return the human name exactly as it appears in the speaker list.

task_text
  - A concise verb phrase describing the task. Max 20 words.
  - Start with a verb: "Push the hook", "Run the benchmarks", "Fix the latency bug".
  - Do NOT include the owner name or deadline in task_text.

deadline_hint
  - Any explicit time reference: "tonight", "by Friday", "before the demo",
    "end of day", "this week", "in 10 minutes".
  - If none is mentioned or implied, return null (JSON null, not the string "null").

urgency
  - "high"   — "ASAP", "urgent", "right now", "before demo", "tonight", "critical"
  - "medium" — "soon", "this week", "before next call", "when you get a chance"
  - "low"    — no time cues, or explicitly low-priority phrasing

OUTPUT FORMAT
-------------
Return ONLY a valid JSON object. No markdown fences. No extra keys. No prose.

{
  "owner_speaker":  "<name or 'unassigned'>",
  "task_text":      "<verb phrase>",
  "deadline_hint":  "<string or null>",
  "urgency":        "<high | medium | low>"
}

CONTEXT PROVIDED
----------------
You will receive:
  - "speaker":        who said the utterance
  - "utterance":      the action item text to extract from
  - "other_speaker":  name of the other participant (for ownership resolution)
  - "recent_context": last 1–3 utterances for disambiguation

EXAMPLES
--------

Example 1
Input:
{
  "speaker": "Khush",
  "utterance": "Satvik, can you run the Whisper latency benchmark tonight so we have numbers before the demo?",
  "other_speaker": "Satvik",
  "recent_context": ["Khush: We need hard latency numbers before the presentation."]
}
Output:
{
  "owner_speaker": "Satvik",
  "task_text": "Run the Whisper latency benchmark",
  "deadline_hint": "tonight",
  "urgency": "high"
}

Example 2
Input:
{
  "speaker": "Satvik",
  "utterance": "I'll fix the FAISS indexing bug before our next sync.",
  "other_speaker": "Khush",
  "recent_context": ["Khush: The FAISS search is returning wrong nodes."]
}
Output:
{
  "owner_speaker": "Satvik",
  "task_text": "Fix the FAISS indexing bug",
  "deadline_hint": "before next sync",
  "urgency": "medium"
}

Example 3
Input:
{
  "speaker": "Khush",
  "utterance": "Someone needs to write the Samsung Notes export function at some point.",
  "other_speaker": "Satvik",
  "recent_context": []
}
Output:
{
  "owner_speaker": "unassigned",
  "task_text": "Write the Samsung Notes export function",
  "deadline_hint": null,
  "urgency": "low"
}

Example 4
Input:
{
  "speaker": "Satvik",
  "utterance": "Push the seed.db to the demo folder ASAP, the judges might ask for a fallback demo.",
  "other_speaker": "Khush",
  "recent_context": ["Satvik: We need the pre-seeded database ready."]
}
Output:
{
  "owner_speaker": "unassigned",
  "task_text": "Push seed.db to the demo folder",
  "deadline_hint": "ASAP",
  "urgency": "high"
}
"""

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

def build_action_extraction_prompt(
    utterance: str,
    speaker: str,
    other_speaker: str,
    recent_context: list[str],
) -> str:
    """Format one action-item utterance into the user turn for Phi-3.

    Parameters
    ----------
    utterance : str
        The utterance text to extract from. Already confirmed as action_item.
    speaker : str
        Name of the person who said the utterance, e.g. "Khush".
    other_speaker : str
        Name of the other participant, e.g. "Satvik".
        Used to resolve "you" / "can you" ownership.
    recent_context : list[str]
        Last 1–3 utterances as "Speaker: text" strings. Pass [] if none.

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
        f"Extract action item fields from the following utterance.\n\n"
        f"recent_context:\n{context_block}\n\n"
        f"speaker: {speaker}\n"
        f"other_speaker: {other_speaker}\n"
        f"utterance: {utterance}"
    )


# ---------------------------------------------------------------------------
# Paper test cases — verified manually
# ---------------------------------------------------------------------------
#
# 1. "Satvik, push the updated requirements.txt tonight."
#    → owner: Satvik, task: "Push the updated requirements.txt",
#      deadline: "tonight", urgency: high  ✓
#
# 2. "I'll handle the Blueprint PDF, no worries."  (speaker = Khush)
#    → owner: Khush, task: "Handle the Blueprint PDF",
#      deadline: null, urgency: low  ✓
#
# 3. "Can you double check the HRTF file paths are correct?"  (speaker = Khush, other = Satvik)
#    → owner: Satvik, task: "Double check the HRTF file paths",
#      deadline: null, urgency: medium  ✓
#
# 4. "We need someone to record the demo audio before Friday."
#    → owner: unassigned, task: "Record the demo audio",
#      deadline: "before Friday", urgency: high  ✓
#
# 5. "I should probably write the graph serializer at some point."
#    → owner: [speaker], task: "Write the graph serializer",
#      deadline: null, urgency: low  ✓
