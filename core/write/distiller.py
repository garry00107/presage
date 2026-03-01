"""
MemoryDistiller — decides what from a conversation turn is worth storing,
then emits structured MemoryCandidates for the write pipeline.

Uses a small, fast LLM call (not the full conversation LLM) to extract
signal from noise. The prompt is tightly constrained to return JSON only.

Design goals:
  - Fast: uses a cheap model (haiku / gpt-4o-mini)
  - Precise: only stores genuinely new, durable knowledge
  - Structured: output is always parseable JSON, never free text
  - Fault-tolerant: falls back to heuristic extraction on LLM failure
"""

import json
import re
import time
import uuid
from dataclasses import dataclass, field

from core.types import MemoryID


@dataclass
class MemoryCandidate:
    """A proposed memory before conflict resolution and chunking."""
    id: MemoryID = field(default_factory=lambda: MemoryID(str(uuid.uuid4())))
    content: str = ""
    source: str = ""
    source_type: str = "prose"
    forward_contexts: list[str] = field(default_factory=list)
    graph_edges: list[dict] = field(default_factory=list)
    confidence: float = 1.0   # how sure we are this is worth storing


# ── Distillation prompt ────────────────────────────────────────────────────────
#
# Constraints enforced in the prompt:
#   1. JSON only — no preamble, no markdown fences
#   2. worth_storing=false → empty memories list (no hallucinated content)
#   3. forward_contexts must be from the allowed namespace
#   4. Short content only (< 500 tokens per memory)

_DISTILL_PROMPT = """\
You are a memory distillation engine. Analyze this conversation turn and extract ONLY information worth storing as long-term memory.

Criteria for storing:
- A fact, decision, or code pattern that will be needed in FUTURE turns
- A user preference or constraint that affects future responses
- A bug discovered or a fix applied (with the specific detail)
- A new understanding about the codebase structure

DO NOT store:
- Information already covered by the conversation history
- Transient state or intermediate reasoning steps
- Simple clarifications or acknowledgements

Allowed source_types: code, prose, json, yaml, md
Allowed forward_context prefixes: topic:, intent:, file:, symbol:, type:, lang:, dir:

Respond with ONLY valid JSON (no markdown, no preamble):
{
  "worth_storing": true | false,
  "memories": [
    {
      "content": "string — the exact knowledge to store",
      "source_type": "code | prose | json | yaml | md",
      "forward_contexts": ["topic:auth", "intent:DEBUG", ...],
      "rationale": "one sentence why this is worth storing"
    }
  ]
}

Conversation turn:
USER: {user_message}
ASSISTANT: {assistant_message}
"""


class MemoryDistiller:
    """
    Extracts memory candidates from a conversation turn.

    Two extraction paths:
      1. LLM path: sends a constrained prompt, parses JSON response
      2. Heuristic fallback: regex-based extraction for when LLM fails

    The LLM path uses a cheap, fast model — NOT the same model serving
    the conversation. This keeps distillation off the critical path.
    """

    # Model for distillation — cheapest capable model
    DISTILL_MODEL_ANTHROPIC = "claude-haiku-4-5-20251001"
    DISTILL_MODEL_OPENAI    = "gpt-4o-mini"
    MAX_TOKENS = 1024

    def __init__(self, llm_backend: str = "anthropic",
                 api_key: str = ""):
        self._backend = llm_backend
        self._api_key = api_key

    async def distill(
        self,
        user_message: str,
        assistant_message: str,
        source: str = "",
    ) -> list[MemoryCandidate]:
        """
        Main entry point. Returns 0–3 memory candidates per turn.
        Never raises — falls back to heuristics on any failure.
        """
        try:
            raw = await self._call_llm(user_message, assistant_message)
            return self._parse_response(raw, source)
        except Exception:
            return self._heuristic_extract(user_message, assistant_message, source)

    # ── LLM call ──────────────────────────────────────────────────────────────

    async def _call_llm(self, user_msg: str, asst_msg: str) -> str:
        prompt = _DISTILL_PROMPT.format(
            user_message=user_msg[:2000],       # guard against huge inputs
            assistant_message=asst_msg[:2000],
        )
        if self._backend == "anthropic":
            return await self._call_anthropic(prompt)
        return await self._call_openai(prompt)

    async def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        msg = await client.messages.create(
            model=self.DISTILL_MODEL_ANTHROPIC,
            max_tokens=self.MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    async def _call_openai(self, prompt: str) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self._api_key)
        resp = await client.chat.completions.create(
            model=self.DISTILL_MODEL_OPENAI,
            max_tokens=self.MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or "{}"

    # ── Response parsing ───────────────────────────────────────────────────────

    def _parse_response(self, raw: str, source: str) -> list[MemoryCandidate]:
        # Strip markdown fences defensively
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            return []

        if not data.get("worth_storing", False):
            return []

        candidates = []
        for item in data.get("memories", [])[:3]:   # cap at 3 per turn
            content = item.get("content", "").strip()
            if not content or len(content) < 10:
                continue
            candidates.append(MemoryCandidate(
                content=content,
                source=source,
                source_type=item.get("source_type", "prose"),
                forward_contexts=item.get("forward_contexts", []),
                confidence=1.0,
            ))
        return candidates

    # ── Heuristic fallback ─────────────────────────────────────────────────────

    def _heuristic_extract(
        self, user_msg: str, asst_msg: str, source: str
    ) -> list[MemoryCandidate]:
        """
        Rule-based extraction when LLM call fails.
        Looks for code blocks and explicit decision statements.
        Lower confidence than LLM path.
        """
        candidates = []

        # Extract code blocks from assistant response
        for m in re.finditer(r'```(\w*)\n(.*?)```', asst_msg, re.DOTALL):
            lang = m.group(1) or "code"
            code = m.group(2).strip()
            if len(code) > 50:   # skip trivial snippets
                candidates.append(MemoryCandidate(
                    content=code,
                    source=source,
                    source_type="code",
                    forward_contexts=[f"lang:{lang}" if lang else "type:code"],
                    confidence=0.6,
                ))

        # Extract decision/conclusion sentences from user message
        decision_patterns = [
            r'(?:we decided|I want|the fix is|solution is|use|prefer)[^.!?]{10,100}[.!?]',
            r'(?:don\'t|never|always|must|should)[^.!?]{5,80}[.!?]',
        ]
        for pattern in decision_patterns:
            for m in re.finditer(pattern, user_msg, re.IGNORECASE):
                candidates.append(MemoryCandidate(
                    content=m.group(0).strip(),
                    source=source,
                    source_type="prose",
                    forward_contexts=["intent:IMPLEMENT"],
                    confidence=0.4,
                ))
            if candidates:
                break   # one pattern match is enough

        return candidates[:2]   # cap heuristic output

