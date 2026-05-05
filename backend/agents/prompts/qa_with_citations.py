"""
backend/agents/prompts/qa_with_citations.py
--------------------------------------------
Phi-3 prompt for QAAgent.

Task: given a user question and the top-5 most relevant utterance nodes
retrieved by FAISS, generate a grounded answer with citation node IDs.

Output: strict JSON, one object, no prose.
Every answer MUST cite at least one node. No uncited answers — hard rule.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
#
# {
#   "answer":            str        — the answer in plain English, 1–4 sentences.
#                                     Must reference content from the context nodes.
#                                     Do not invent facts not present in context.
#   "citation_node_ids": list[str]  — UUIDs of the nodes the answer draws from.
#                                     Must contain at least one ID.
#                                     Only include nodes actually used in the answer.
#   "confidence":        float      — 0.0–1.0. Use lower values when context is
#                                     sparse or the question is only partially answered.
#   "answerable":        bool       — false if the context nodes do not contain
#                                     enough information to answer the question.
#                                     If false, answer should say what IS known.
# }

QA_SCHEMA = {
    "answer": "str — plain English, 1–4 sentences, grounded in context",
    "citation_node_ids": "list[str] — UUIDs of nodes used, at least one required",
    "confidence": "float 0.0–1.0",
    "answerable": "bool — false if context is insufficient",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

QA_WITH_CITATIONS_SYSTEM = """\
You are an AI assistant embedded in a real-time voice call analysis system.

A user has asked a question about the ongoing call. You have been given the
5 most relevant utterances from the conversation, retrieved by semantic search.

Your job is to answer the question using ONLY information present in the
provided context utterances. You must cite the node IDs you used.

ANSWER RULES
------------
1. Answer only from the provided context. Do NOT use outside knowledge.
2. If the context does not contain enough information to answer, set
   "answerable": false and summarise what IS known from context.
3. Keep answers concise: 1–4 sentences.
4. Always include at least one citation_node_id. Never return an empty list.
5. Only cite nodes you actually used. Do not cite every node just to seem thorough.
6. If multiple nodes support the answer, cite all of them.
7. Write the answer in plain English — no technical jargon unless it appeared
   in the utterances themselves.

CONFIDENCE GUIDE
----------------
0.9–1.0 — question is directly and explicitly answered by one or more nodes
0.7–0.9 — answer is strongly implied by context, minor inference required
0.5–0.7 — partial answer only; context touches on the topic but not fully
0.0–0.5 — context is tangentially related; answer is speculative

OUTPUT FORMAT
-------------
Return ONLY a valid JSON object. No markdown fences. No extra keys. No prose.

{
  "answer":            "<plain English answer>",
  "citation_node_ids": ["<uuid>", "<uuid>"],
  "confidence":        <float 0.0–1.0>,
  "answerable":        <true | false>
}

CONTEXT FORMAT
--------------
Each context utterance is provided as:
  node_id:      <uuid>
  speaker:      <name>
  timestamp:    <seconds since call start>
  text:         <utterance text>

EXAMPLES
--------

Example 1
Question: "What model did we decide to use for speech recognition?"
Context:
  node_id: a1b2c3d4-...
  speaker: Satvik
  timestamp: 142.3
  text: "I ran Whisper small and Whisper medium. Small hits 380ms, medium is around 700ms on this machine."

  node_id: e5f6a7b8-...
  speaker: Khush
  timestamp: 156.1
  text: "Okay we are going with Whisper small, that is the final call."

Output:
{
  "answer": "You decided to use Whisper small for speech recognition. Satvik benchmarked both small and medium, with small achieving 380ms on the demo machine, and Khush made the final call.",
  "citation_node_ids": ["a1b2c3d4-...", "e5f6a7b8-..."],
  "confidence": 0.97,
  "answerable": true
}

Example 2
Question: "Who is responsible for the Samsung Notes export?"
Context:
  node_id: c9d0e1f2-...
  speaker: Khush
  timestamp: 301.5
  text: "The Samsung Notes export is on the list but nobody has picked it up yet."

  node_id: 33a4b5c6-...
  speaker: Satvik
  timestamp: 488.2
  text: "Yeah the spatial audio positions are matching the Galaxy Buds range, that looks right."

Output:
{
  "answer": "As of the most recent mention, no one has been assigned the Samsung Notes export task. Khush noted it is on the list but unowned.",
  "citation_node_ids": ["c9d0e1f2-..."],
  "confidence": 0.82,
  "answerable": true
}

Example 3
Question: "What is the target latency for the full pipeline?"
Context:
  node_id: 77f8a9b0-...
  speaker: Khush
  timestamp: 88.0
  text: "Good morning, shall we start?"

  node_id: 21c2d3e4-...
  speaker: Satvik
  timestamp: 91.2
  text: "Yeah let's go, I set up the environment yesterday."

Output:
{
  "answer": "The provided context does not contain a discussion of pipeline latency targets. The retrieved utterances cover the start of the call only.",
  "citation_node_ids": ["77f8a9b0-..."],
  "confidence": 0.1,
  "answerable": false
}
"""

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

def build_qa_prompt(
    question: str,
    context_nodes: list[dict],
) -> str:
    """Format a question and retrieved nodes into the user turn for Phi-3.

    Parameters
    ----------
    question : str
        The user's question about the call.
    context_nodes : list[dict]
        Top-K nodes from FAISS retrieval. Each dict must have keys:
          - node_id:   str (UUID)
          - speaker:   str (speaker name)
          - timestamp: float (seconds since call start)
          - text:      str (utterance text)
        Pass up to 5 nodes (cfg.agent.QA_RETRIEVAL_TOP_K).

    Returns
    -------
    str
        User-turn string to pass to Phi3Runner.run(user_prompt=...).
    """
    context_lines = []
    for node in context_nodes:
        context_lines.append(
            f"  node_id:   {node['node_id']}\n"
            f"  speaker:   {node['speaker']}\n"
            f"  timestamp: {node['timestamp']:.1f}s\n"
            f"  text:      {node['text']}\n"
        )
    context_block = "\n".join(context_lines) if context_lines else "  (no context retrieved)"

    return (
        f"Answer the following question using ONLY the context nodes below.\n\n"
        f"question: {question}\n\n"
        f"context:\n{context_block}"
    )


# ---------------------------------------------------------------------------
# Paper test cases — verified manually
# ---------------------------------------------------------------------------
#
# 1. "What did we decide about the HRTF dataset?"
#    Context contains: "We're using SADIE II Subject 002, final decision."
#    → answerable: true, confidence: 0.95
#    → citations: [node with SADIE decision]  ✓
#
# 2. "Did anyone mention the demo fallback?"
#    Context contains: "We need seed.db ready before demo day."
#    → answerable: true, confidence: 0.88  ✓
#
# 3. "What is Satvik's phone number?"
#    Context: general call about code
#    → answerable: false, confidence: 0.0  ✓
#
# 4. "What are the action items from this call?"
#    Context contains 3 action item nodes
#    → cites all 3, answer summarises them  ✓
#
# 5. "Who decided to drop pyannote?"
#    Context contains: "Khush: we go with speechbrain, pyannote has gated license issues"
#    → owner: Khush, cited correctly  ✓
