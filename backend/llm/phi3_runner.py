"""
backend/llm/phi3_runner.py
--------------------------
Thin wrapper around llama-cpp-python for Phi-3-mini-4k-instruct (Q4_K_M).

TEAMMATE: Wire llama-cpp in the TODO sections.
  - __init__: load Llama model, run pre-warm dummy inference
  - run: call model, parse JSON, retry once on failure
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Sentinel returned to callers when both inference attempts fail to produce valid JSON.
# Agents must check for this and handle gracefully (log, skip, or return safe default).
ERROR_RESPONSE: Dict[str, Any] = {
    "error": True,
    "message": "phi3_runner: failed to produce valid JSON after 2 attempts",
}

# Phi-3 chat template roles
_SYSTEM_ROLE = "<|system|>"
_USER_ROLE   = "<|user|>"
_ASSISTANT_ROLE = "<|assistant|>"
_END_TOKEN   = "<|end|>"


def _build_prompt(system_prompt: str, user_prompt: str) -> str:
    """Format system + user turn into Phi-3 chat template.

    Phi-3 expected format:
        <|system|>
        {system_prompt}<|end|>
        <|user|>
        {user_prompt}<|end|>
        <|assistant|>
    """
    return (
        f"{_SYSTEM_ROLE}\n{system_prompt}{_END_TOKEN}\n"
        f"{_USER_ROLE}\n{user_prompt}{_END_TOKEN}\n"
        f"{_ASSISTANT_ROLE}\n"
    )


class Phi3Runner:
    """Synchronous Phi-3 inference with JSON output enforcement.

    All agents call .run() and always receive either a valid parsed dict
    or ERROR_RESPONSE. Raw LLM text is never returned to callers.

    Parameters
    ----------
    model_path : str | Path
        Path to the .gguf model file, e.g. backend/models/phi3/phi3.gguf
    n_ctx : int
        Context window size. Phi-3-mini supports 4096. Default 2048 is safe.
    n_threads : int
        CPU threads for inference. Set to physical core count for best latency.
    temperature : float
        Sampling temperature. 0.0 = greedy (deterministic), best for JSON.

    Example
    -------
    >>> runner = Phi3Runner("backend/models/phi3/phi3.gguf")
    >>> result = runner.run(
    ...     system_prompt="You are a classifier. Return JSON only.",
    ...     user_prompt="Classify: 'Let's go with Whisper small.'",
    ...     max_tokens=128,
    ... )
    >>> # result == {"flag": "decision", "confidence": 0.91, ...}
    """

    def __init__(
        self,
        model_path: str | Path,
        n_ctx:      int   = 2048,
        n_threads:  int   = 4,
        temperature: float = 0.0,
    ) -> None:
        self.model_path  = Path(model_path)
        self.n_ctx       = n_ctx
        self.n_threads   = n_threads
        self.temperature = temperature

        # TEAMMATE: Instantiate llama-cpp-python Llama here.
        # from llama_cpp import Llama
        # self._model = Llama(
        #     model_path=str(self.model_path),
        #     n_ctx=self.n_ctx,
        #     n_threads=self.n_threads,
        #     verbose=False,
        # )
        self._model = None  # replace with Llama(...) above

        logger.info("Phi3Runner loaded model from %s", self.model_path)
        self._prewarm()

    # ------------------------------------------------------------------
    # Pre-warm  (eliminates cold-start latency on first real call)
    # ------------------------------------------------------------------

    def _prewarm(self) -> None:
        """Run one dummy inference so the first real call pays no cold-start cost.

        Hard rule from CONTEXT.md: pre-warm must run at server startup.
        Called automatically from __init__.

        TODO: Uncomment once self._model is wired.
        """
        logger.info("Phi3Runner: pre-warming model...")
        t0 = time.perf_counter()
        # self._model("Hello", max_tokens=1)   # TODO: uncomment
        elapsed = time.perf_counter() - t0
        logger.info("Phi3Runner: pre-warm done in %.2fs", elapsed)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _infer(self, prompt: str, max_tokens: int) -> str:
        """Run raw inference and return the completion text.

        TODO:
          output = self._model(
              prompt,
              max_tokens=max_tokens,
              temperature=self.temperature,
              stop=["<|end|>", "<|user|>"],
          )
          return output["choices"][0]["text"].strip()
        """
        raise NotImplementedError("Teammate: wire llama-cpp in _infer()")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        system_prompt: str,
        user_prompt:   str,
        max_tokens:    int = 256,
    ) -> Dict[str, Any]:
        """Run Phi-3 inference and return a parsed JSON dict.

        Retry logic
        -----------
        Attempt 1 — run with original prompts, try json.loads().
        Attempt 2 — if parse fails, append stricter instruction to system
                    prompt ("Return ONLY a JSON object, no markdown.") and retry.
        If both fail — return ERROR_RESPONSE sentinel dict.

        Callers must always check result.get("error") before using the result.

        Parameters
        ----------
        system_prompt : str
            System turn — role description + output schema specification.
        user_prompt : str
            User turn — the actual input to classify/extract/answer.
        max_tokens : int
            Max tokens to generate. Keep small for classification (<128),
            larger for Q&A (<512).

        Returns
        -------
        dict
            Parsed JSON dict from model output, or ERROR_RESPONSE on failure.
        """
        for attempt in range(1, 3):
            effective_system = system_prompt
            if attempt == 2:
                # Stricter instruction on retry
                effective_system = (
                    system_prompt
                    + "\n\nCRITICAL: Return ONLY a valid JSON object. "
                    "No markdown fences. No explanatory text. JSON only."
                )
                logger.warning(
                    "Phi3Runner: JSON parse failed on attempt 1 — retrying with stricter prompt"
                )

            prompt = _build_prompt(effective_system, user_prompt)
            t0 = time.perf_counter()

            try:
                raw_text = self._infer(prompt, max_tokens)
            except Exception as exc:
                logger.error("Phi3Runner: inference error on attempt %d: %s", attempt, exc)
                continue

            latency_ms = int((time.perf_counter() - t0) * 1000)
            logger.debug("Phi3Runner: attempt %d — %dms — raw: %r", attempt, latency_ms, raw_text[:120])

            # Strip markdown fences if model wraps output despite instructions
            clean = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            try:
                parsed = json.loads(clean)
                parsed["_latency_ms"] = latency_ms   # attach latency for monitoring
                return parsed
            except json.JSONDecodeError as exc:
                logger.warning("Phi3Runner: JSONDecodeError on attempt %d: %s", attempt, exc)
                continue

        logger.error("Phi3Runner: both attempts failed — returning ERROR_RESPONSE")
        return ERROR_RESPONSE
