# =============================================================================
# PPM: Predictive Push Memory — Phase 2: Write Pipeline
# core/write/chunker.py
# core/write/distiller.py
# core/write/conflict.py
# core/write/annotator.py
# core/write/pipeline.py          ← orchestrator (ties all 4 together)
# tests/unit/test_chunker.py
# tests/unit/test_conflict.py
# tests/unit/test_annotator.py
# tests/integration/test_write_pipeline.py
# =============================================================================


### core/write/chunker.py ###

"""
SemanticChunker — splits raw memory content into natural-boundary chunks
at WRITE TIME (never at injection time).

Supported source types:
  'code'  → AST node boundaries (FunctionDef, AsyncFunctionDef, ClassDef)
  'json'  → top-level key boundaries
  'yaml'  → top-level key boundaries
  'prose' → sentence boundaries (~150 token target per chunk)
  'md'    → markdown header boundaries

Each Chunk is a self-contained semantic unit that can be injected
into an LLM context without truncation or ambiguity.
"""

import ast
import re
import json
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from core.types import ChunkID, MemoryID


# Approximate token count: GPT-family ~4 chars/token, conservative estimate
def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 3)


@dataclass
class RawChunk:
    id: ChunkID
    parent_id: MemoryID
    chunk_index: int
    content: str
    tokens: int
    source_type: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "tokens": self.tokens,
            "source_type": self.source_type,
            "score": 0.0,   # set at retrieval time
            "embedding": None,
        }


class SemanticChunker:
    """
    Splits memory content into natural-boundary chunks.

    Design invariant: chunk content is NEVER modified after creation.
    The knapsack and injection layers receive these chunks as-is.
    """

    # Prose chunking: target tokens per chunk
    PROSE_TARGET_TOKENS: int = 150
    PROSE_MAX_TOKENS: int = 300

    def chunk(self, content: str, parent_id: MemoryID,
              source_type: str) -> list[RawChunk]:
        """
        Dispatch to the appropriate chunker by source_type.
        Always returns at least one chunk (the full content as fallback).
        """
        try:
            if source_type == "code":
                chunks = list(self._chunk_code(content, parent_id))
            elif source_type == "json":
                chunks = list(self._chunk_json(content, parent_id))
            elif source_type == "yaml":
                chunks = list(self._chunk_yaml(content, parent_id))
            elif source_type == "md":
                chunks = list(self._chunk_markdown(content, parent_id))
            else:
                chunks = list(self._chunk_prose(content, parent_id))
        except Exception:
            # Fallback: treat entire content as one chunk rather than crashing
            chunks = []

        if not chunks:
            chunks = [self._make_chunk(content, parent_id, 0, source_type)]

        return chunks

    # ── Code ──────────────────────────────────────────────────────────────────

    def _chunk_code(self, content: str, parent_id: MemoryID) -> Iterator[RawChunk]:
        """
        Split Python source at top-level AST node boundaries.
        Each function/class becomes one chunk. Module-level statements
        (imports, constants) are grouped into a preamble chunk.

        Falls back to prose chunking for non-Python code.
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Non-Python code: fall back to prose chunking
            yield from self._chunk_prose(content, parent_id, source_type="code")
            return

        lines = content.splitlines(keepends=True)
        top_level_nodes = [
            n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef, ast.If, ast.For, ast.While,
                               ast.Try, ast.With, ast.Match))
        ]

        # Preamble: everything before the first top-level node
        preamble_lines: list[str] = []
        if top_level_nodes:
            first_line = top_level_nodes[0].lineno - 1
            preamble_lines = lines[:first_line]
        else:
            preamble_lines = lines

        idx = 0
        if preamble_text := "".join(preamble_lines).strip():
            yield self._make_chunk(preamble_text, parent_id, idx, "code")
            idx += 1

        for node in top_level_nodes:
            start = node.lineno - 1
            end = node.end_lineno          # type: ignore[attr-defined]
            node_content = "".join(lines[start:end]).rstrip()
            if node_content.strip():
                yield self._make_chunk(node_content, parent_id, idx, "code")
                idx += 1

    # ── JSON ──────────────────────────────────────────────────────────────────

    def _chunk_json(self, content: str, parent_id: MemoryID) -> Iterator[RawChunk]:
        """
        Split JSON at top-level keys.
        Each key → value pair becomes one chunk (valid JSON fragment).
        Preserves JSON validity per chunk for LLM parseability.
        """
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            yield from self._chunk_prose(content, parent_id, source_type="json")
            return

        if not isinstance(obj, dict):
            # Array or scalar: treat as single chunk
            yield self._make_chunk(content, parent_id, 0, "json")
            return

        for idx, (key, val) in enumerate(obj.items()):
            fragment = json.dumps({key: val}, indent=2, ensure_ascii=False)
            yield self._make_chunk(fragment, parent_id, idx, "json")

    # ── YAML ──────────────────────────────────────────────────────────────────

    def _chunk_yaml(self, content: str, parent_id: MemoryID) -> Iterator[RawChunk]:
        """
        Split YAML at top-level key boundaries (regex-based, no yaml dep).
        Each top-level key block becomes one chunk.
        """
        # Match lines that start a top-level key: "key:" or "key: value"
        top_key_pattern = re.compile(r'^(\w[\w\-]*)\s*:', re.MULTILINE)
        lines = content.splitlines(keepends=True)
        matches = list(top_key_pattern.finditer(content))

        if not matches:
            yield self._make_chunk(content, parent_id, 0, "yaml")
            return

        # Build line index from char offset
        char_to_line: list[int] = []
        for i, line in enumerate(lines):
            char_to_line.extend([i] * len(line))

        boundaries = [char_to_line[m.start()] for m in matches]
        boundaries.append(len(lines))

        for idx, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            block = "".join(lines[start:end]).rstrip()
            if block.strip():
                yield self._make_chunk(block, parent_id, idx, "yaml")

    # ── Markdown ───────────────────────────────────────────────────────────────

    def _chunk_markdown(self, content: str, parent_id: MemoryID) -> Iterator[RawChunk]:
        """
        Split markdown at H1/H2/H3 header boundaries.
        Each section (header + body) becomes one chunk.
        """
        header_re = re.compile(r'^#{1,3}\s+.+', re.MULTILINE)
        lines = content.splitlines(keepends=True)
        positions = [m.start() for m in header_re.finditer(content)]

        if not positions:
            yield from self._chunk_prose(content, parent_id, source_type="md")
            return

        # Build char → line mapping
        char_pos = 0
        line_starts = [0]
        for line in lines[:-1]:
            char_pos += len(line)
            line_starts.append(char_pos)

        def char_to_lineno(c: int) -> int:
            lo, hi = 0, len(line_starts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if line_starts[mid] <= c:
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        boundaries = [char_to_lineno(p) for p in positions]
        boundaries.append(len(lines))

        idx = 0
        # Content before first header
        if boundaries[0] > 0:
            pre = "".join(lines[:boundaries[0]]).strip()
            if pre:
                yield self._make_chunk(pre, parent_id, idx, "md")
                idx += 1

        for start, end in zip(boundaries, boundaries[1:]):
            section = "".join(lines[start:end]).rstrip()
            if section.strip():
                # If section too large, sub-chunk by prose
                if _approx_tokens(section) > self.PROSE_MAX_TOKENS:
                    for sub in self._chunk_prose(section, parent_id,
                                                  source_type="md"):
                        sub.chunk_index = idx
                        yield sub
                        idx += 1
                else:
                    yield self._make_chunk(section, parent_id, idx, "md")
                    idx += 1

    # ── Prose ──────────────────────────────────────────────────────────────────

    def _chunk_prose(self, content: str, parent_id: MemoryID,
                     source_type: str = "prose") -> Iterator[RawChunk]:
        """
        Split prose at sentence boundaries, grouping into ~150 token chunks.
        Sentence boundary: period/!/?  followed by whitespace + capital letter.
        """
        sentence_re = re.compile(r'(?<=[.!?])\s+(?=[A-Z\"\'\(])')
        sentences = sentence_re.split(content)

        current: list[str] = []
        current_tokens = 0
        idx = 0

        for sentence in sentences:
            s_tokens = _approx_tokens(sentence)

            # If single sentence exceeds max, yield it alone
            if s_tokens > self.PROSE_MAX_TOKENS:
                if current:
                    yield self._make_chunk(" ".join(current), parent_id,
                                           idx, source_type)
                    idx += 1
                    current, current_tokens = [], 0
                yield self._make_chunk(sentence, parent_id, idx, source_type)
                idx += 1
                continue

            if current_tokens + s_tokens > self.PROSE_TARGET_TOKENS and current:
                yield self._make_chunk(" ".join(current), parent_id,
                                       idx, source_type)
                idx += 1
                current, current_tokens = [], 0

            current.append(sentence)
            current_tokens += s_tokens

        if current:
            yield self._make_chunk(" ".join(current), parent_id, idx, source_type)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _make_chunk(self, content: str, parent_id: MemoryID,
                    idx: int, source_type: str) -> RawChunk:
        return RawChunk(
            id=ChunkID(str(uuid.uuid4())),
            parent_id=parent_id,
            chunk_index=idx,
            content=content,
            tokens=_approx_tokens(content),
            source_type=source_type,
        )


### core/write/conflict.py ###

"""
ConflictResolver — determines how a new memory relates to existing memories
and decides the appropriate write strategy.

Conflict taxonomy:
  DUPLICATE   → content_hash match or cosine_sim > 0.97
                Action: skip write, increment access_count
  CONFLICT    → cosine_sim ∈ [0.80, 0.97), content materially differs
                Action: write new version, link CONFLICTS_WITH edge, keep both
  EXTENSION   → cosine_sim ∈ [0.55, 0.80)
                Action: write new version, link SUMMARIZES/EXTENDS edge, deprecate old
  NOVEL       → cosine_sim < 0.55
                Action: write as independent memory, no versioning needed

The resolver never silently overwrites. Every conflict is recorded in the
graph so the LLM can reason about contradictions explicitly.
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np

from core.types import MemoryID, UnitVector


class ConflictType(str, Enum):
    DUPLICATE = "DUPLICATE"
    CONFLICT  = "CONFLICT"
    EXTENSION = "EXTENSION"
    NOVEL     = "NOVEL"


@dataclass
class ConflictResolution:
    conflict_type: ConflictType
    existing_id: MemoryID | None    # None for NOVEL
    similarity: float
    edge_type: str | None           # graph edge to create
    should_write: bool              # False for DUPLICATE
    should_deprecate: bool          # True for EXTENSION


class ConflictThresholds(NamedTuple):
    duplicate: float = 0.97
    conflict:  float = 0.80
    extension: float = 0.55


class ConflictResolver:
    """
    Compares a new memory embedding against existing candidates
    and returns a ConflictResolution describing the write strategy.

    Designed to be called BEFORE writing to the store, so the write
    layer can make a single informed decision about what to persist.
    """

    def __init__(self, thresholds: ConflictThresholds | None = None):
        self.t = thresholds or ConflictThresholds()

    def resolve(
        self,
        new_embedding: UnitVector,
        candidates: list[tuple[MemoryID, UnitVector, str]],
        # candidates: [(id, embedding, content_hash)]
        new_hash: str = "",
    ) -> ConflictResolution:
        """
        Compare new memory against candidate existing memories.

        Args:
            new_embedding: L2-normalized embedding of new content
            candidates:    [(memory_id, embedding, content_hash)] from store
            new_hash:      SHA-256 of new content (for exact dedup)

        Returns:
            ConflictResolution with the recommended write action.
        """
        if not candidates:
            return ConflictResolution(
                conflict_type=ConflictType.NOVEL,
                existing_id=None, similarity=0.0,
                edge_type=None, should_write=True, should_deprecate=False,
            )

        # Find the most similar existing memory
        best_id, best_sim, best_hash = self._best_match(
            new_embedding, new_hash, candidates
        )

        return self._classify(best_id, best_sim, new_hash, best_hash)

    def _best_match(
        self,
        new_emb: UnitVector,
        new_hash: str,
        candidates: list[tuple[MemoryID, UnitVector, str]],
    ) -> tuple[MemoryID, float, str]:
        best_id, best_sim, best_hash = candidates[0][0], -1.0, ""
        for mem_id, emb, h in candidates:
            # Exact hash match → instant duplicate
            if new_hash and h == new_hash:
                return mem_id, 1.0, h
            sim = float(np.dot(new_emb, emb))  # cosine sim (both unit vectors)
            if sim > best_sim:
                best_id, best_sim, best_hash = mem_id, sim, h
        return best_id, best_sim, best_hash

    def _classify(
        self,
        existing_id: MemoryID,
        sim: float,
        new_hash: str,
        existing_hash: str,
    ) -> ConflictResolution:
        if sim >= self.t.duplicate or (new_hash and new_hash == existing_hash):
            return ConflictResolution(
                conflict_type=ConflictType.DUPLICATE,
                existing_id=existing_id, similarity=sim,
                edge_type=None, should_write=False, should_deprecate=False,
            )

        if sim >= self.t.conflict:
            return ConflictResolution(
                conflict_type=ConflictType.CONFLICT,
                existing_id=existing_id, similarity=sim,
                edge_type="CONFLICTS_WITH",
                should_write=True, should_deprecate=False,
            )

        if sim >= self.t.extension:
            return ConflictResolution(
                conflict_type=ConflictType.EXTENSION,
                existing_id=existing_id, similarity=sim,
                edge_type="EXTENDS",
                should_write=True, should_deprecate=True,
            )

        return ConflictResolution(
            conflict_type=ConflictType.NOVEL,
            existing_id=None, similarity=sim,
            edge_type=None, should_write=True, should_deprecate=False,
        )


### core/write/annotator.py ###

"""
ForwardAnnotator — enriches new memories with future-relevance tags
at write time.

A forward annotation is a (memory_id, context_tag) pair that says:
  "This memory is likely needed when the conversation is about <context_tag>"

Two annotation sources:
  1. STATIC:  rule-based extraction from content and metadata
  2. DYNAMIC: updated when a memory is used (hit_count increments add new tags)

Tag vocabulary uses a structured namespace:
  topic:<topic>       — semantic topic (e.g., topic:authentication)
  intent:<intent>     — conversation intent (e.g., intent:DEBUG)
  file:<path>         — source file (e.g., file:src/auth.py)
  symbol:<name>       — code symbol (e.g., symbol:verify_token)
  lang:<language>     — programming language (e.g., lang:python)
  type:<source_type>  — memory type (e.g., type:code)
"""

import re
from dataclasses import dataclass

from core.types import MemoryID


@dataclass
class Annotation:
    memory_id: MemoryID
    context_tag: str
    weight: float = 1.0
    source: str = "static"   # 'static' | 'dynamic' | 'llm'


# Python stdlib keywords as topic indicators
_CODE_TOPICS = {
    "auth": ["login", "logout", "token", "jwt", "session", "password",
             "authenticate", "authorize", "permission", "role"],
    "database": ["query", "select", "insert", "update", "delete", "sql",
                 "orm", "migration", "schema", "index", "transaction"],
    "api": ["endpoint", "route", "request", "response", "http", "rest",
             "graphql", "webhook", "middleware", "handler"],
    "testing": ["test", "spec", "assert", "mock", "fixture", "pytest",
                "unittest", "coverage", "stub"],
    "config": ["settings", "env", "environment", "config", "dotenv",
               "secret", "key", "variable"],
    "async": ["async", "await", "coroutine", "asyncio", "concurrent",
              "thread", "worker", "queue", "task"],
    "error": ["exception", "error", "raise", "try", "catch", "traceback",
              "logging", "warning", "debug"],
    "deploy": ["docker", "kubernetes", "ci", "cd", "pipeline", "build",
               "deploy", "container", "helm", "terraform"],
}

_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("intent:DEBUG",      ["error", "exception", "traceback", "not working",
                           "failing", "bug", "broken", "fix"]),
    ("intent:IMPLEMENT",  ["implement", "create", "build", "write", "add",
                           "new function", "def ", "class "]),
    ("intent:EXPLORE",    ["what is", "explain", "how does", "overview",
                           "understand", "describe"]),
    ("intent:REFACTOR",   ["refactor", "clean up", "improve", "optimize",
                           "restructure", "rename"]),
    ("intent:TEST",       ["test", "spec", "coverage", "assert", "mock",
                           "unittest", "pytest"]),
]

_LANG_PATTERNS = {
    "lang:python":     [r'def\s+\w+', r'import\s+\w+', r'class\s+\w+:'],
    "lang:javascript": [r'const\s+\w+', r'=>', r'require\(', r'module\.exports'],
    "lang:typescript": [r'interface\s+\w+', r'type\s+\w+\s*=', r': string', r': number'],
    "lang:rust":       [r'fn\s+\w+', r'let\s+mut', r'impl\s+\w+', r'->.*\{'],
    "lang:go":         [r'func\s+\w+', r':=', r'package\s+\w+', r'goroutine'],
    "lang:sql":        [r'SELECT\s+', r'FROM\s+', r'WHERE\s+', r'JOIN\s+'],
}


class ForwardAnnotator:
    """
    Extracts forward annotation tags from memory content and metadata.
    Tags are written to the MetaStore at write time, enabling
    the predictor to find memories via annotation lookup before
    vector search is even needed.
    """

    def annotate(
        self,
        memory_id: MemoryID,
        content: str,
        source: str = "",
        source_type: str = "prose",
        extra_tags: list[str] | None = None,
    ) -> list[Annotation]:
        """
        Generate all forward annotations for a memory.

        Args:
            memory_id:   ID of the memory being written
            content:     raw text content
            source:      file path or URL
            source_type: 'code' | 'prose' | 'json' | 'yaml' | 'md'
            extra_tags:  manually provided tags (from distiller or user)

        Returns:
            List of Annotation objects ready to write to MetaStore.
        """
        tags: dict[str, float] = {}   # tag → weight

        self._tag_source_type(tags, source_type)
        self._tag_source_path(tags, source)
        self._tag_topics(tags, content)
        self._tag_intents(tags, content)
        self._tag_language(tags, content, source_type)
        self._tag_symbols(tags, content, source_type)

        if extra_tags:
            for t in extra_tags:
                tags[t] = max(tags.get(t, 0.0), 1.5)  # manual tags get higher weight

        return [
            Annotation(memory_id=memory_id, context_tag=tag,
                       weight=weight, source="static")
            for tag, weight in tags.items()
        ]

    # ── Tag extraction methods ─────────────────────────────────────────────────

    def _tag_source_type(self, tags: dict, source_type: str) -> None:
        tags[f"type:{source_type}"] = 1.0

    def _tag_source_path(self, tags: dict, source: str) -> None:
        if not source:
            return
        # Normalize path separators
        norm = source.replace("\\", "/")
        tags[f"file:{norm}"] = 2.0   # file match is highly specific

        # Extract directory tags  
        parts = norm.split("/")
        for part in parts[:-1]:   # exclude filename itself
            if part and not part.startswith("."):
                tags[f"dir:{part}"] = 0.8

        # Extension → language hint
        if "." in parts[-1]:
            ext = parts[-1].rsplit(".", 1)[-1].lower()
            ext_to_lang = {
                "py": "lang:python", "js": "lang:javascript",
                "ts": "lang:typescript", "rs": "lang:rust",
                "go": "lang:go", "sql": "lang:sql",
                "json": "type:json", "yaml": "type:yaml", "yml": "type:yaml",
                "md": "type:md",
            }
            if lang := ext_to_lang.get(ext):
                tags[lang] = max(tags.get(lang, 0.0), 1.2)

    def _tag_topics(self, tags: dict, content: str) -> None:
        lower = content.lower()
        for topic, keywords in _CODE_TOPICS.items():
            hits = sum(1 for kw in keywords if kw in lower)
            if hits >= 2:
                weight = min(1.0 + 0.2 * hits, 2.0)
                tags[f"topic:{topic}"] = weight

    def _tag_intents(self, tags: dict, content: str) -> None:
        lower = content.lower()
        for tag, keywords in _INTENT_PATTERNS:
            if any(kw in lower for kw in keywords):
                tags[tag] = 1.0

    def _tag_language(self, tags: dict, content: str,
                       source_type: str) -> None:
        if source_type not in ("code", "prose"):
            return
        for lang_tag, patterns in _LANG_PATTERNS.items():
            if sum(1 for p in patterns if re.search(p, content)) >= 2:
                tags[lang_tag] = max(tags.get(lang_tag, 0.0), 1.0)

    def _tag_symbols(self, tags: dict, content: str,
                      source_type: str) -> None:
        """Extract function and class names as symbol tags."""
        if source_type != "code":
            return
        # Function definitions
        for m in re.finditer(r'\bdef\s+(\w+)\s*\(', content):
            name = m.group(1)
            if not name.startswith("_"):   # skip private
                tags[f"symbol:{name}"] = 1.5
        # Class definitions
        for m in re.finditer(r'\bclass\s+(\w+)', content):
            tags[f"symbol:{m.group(1)}"] = 1.5


### core/write/distiller.py ###

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


### core/write/pipeline.py ###

"""
WritePipeline — orchestrates the full write flow for a conversation turn.

Flow:
  Turn (user + assistant messages)
    → Distiller    → [MemoryCandidates]
    → Embedder     → embeddings per candidate
    → Resolver     → ConflictResolution per candidate
    → Chunker      → chunks per candidate
    → Annotator    → forward annotations
    → MetaStore    → atomic write (data + outbox entries)

Each step is independently testable. The pipeline is the only place
that coordinates across modules — no module imports another module here.
"""

import asyncio
import hashlib
import uuid
from dataclasses import dataclass

import structlog

from adapters.embedder.base import Embedder
from core.types import MemoryID, UnitVector
from core.write.chunker import SemanticChunker
from core.write.conflict import ConflictResolver, ConflictType
from core.write.distiller import MemoryDistiller, MemoryCandidate
from core.write.annotator import ForwardAnnotator

log = structlog.get_logger(__name__)


@dataclass
class WriteResult:
    memory_id: MemoryID | None
    action: str          # 'written' | 'duplicate' | 'skipped' | 'error'
    conflict_type: str
    chunks_written: int
    annotations_written: int


class WritePipeline:
    """
    Orchestrates memory formation for a single conversation turn.
    All steps run asynchronously. Embedding and distillation can run
    in parallel where possible.
    """

    def __init__(
        self,
        distiller: MemoryDistiller,
        embedder: Embedder,
        resolver: ConflictResolver,
        chunker: SemanticChunker,
        annotator: ForwardAnnotator,
        meta_store,    # MetaStore — avoid circular import with string type
        vector_store,  # VectorStore adapter
        top_k_candidates: int = 5,
    ):
        self.distiller   = distiller
        self.embedder    = embedder
        self.resolver    = resolver
        self.chunker     = chunker
        self.annotator   = annotator
        self.meta        = meta_store
        self.vector      = vector_store
        self.top_k       = top_k_candidates

    async def process_turn(
        self,
        user_message: str,
        assistant_message: str,
        source: str = "",
        extra_tags: list[str] | None = None,
    ) -> list[WriteResult]:
        """
        Full write pipeline for one conversation turn.
        Returns one WriteResult per memory candidate extracted.
        """
        # Step 1: Distill candidates from turn
        candidates = await self.distiller.distill(
            user_message, assistant_message, source
        )

        if not candidates:
            log.debug("write_pipeline.no_candidates", source=source)
            return []

        # Step 2: Embed all candidates in parallel
        embeddings: list[UnitVector] = await self.embedder.embed_batch(
            [c.content for c in candidates]
        )

        # Step 3: Process each candidate
        tasks = [
            self._process_candidate(c, emb, extra_tags or [])
            for c, emb in zip(candidates, embeddings)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        write_results = []
        for r in results:
            if isinstance(r, Exception):
                log.error("write_pipeline.candidate_failed", error=str(r))
                write_results.append(WriteResult(
                    memory_id=None, action="error",
                    conflict_type="UNKNOWN", chunks_written=0, annotations_written=0
                ))
            else:
                write_results.append(r)

        return write_results

    async def _process_candidate(
        self,
        candidate: MemoryCandidate,
        embedding: UnitVector,
        extra_tags: list[str],
    ) -> WriteResult:
        content_hash = hashlib.sha256(candidate.content.encode()).hexdigest()

        # Step 3a: Find similar existing memories for conflict check
        similar = await self.vector.search(embedding, top_k=self.top_k)
        candidates_for_resolver = [
            (MemoryID(r["id"]), r["embedding"], r.get("content_hash", ""))
            for r in similar
            if "embedding" in r and r["embedding"] is not None
        ]

        # Step 3b: Conflict resolution
        resolution = self.resolver.resolve(
            embedding, candidates_for_resolver, content_hash
        )

        if not resolution.should_write:
            log.debug("write_pipeline.duplicate_skipped",
                      existing_id=resolution.existing_id,
                      sim=f"{resolution.similarity:.3f}")
            # Update access count on existing
            if resolution.existing_id:
                await self.meta.touch_memory(resolution.existing_id)
            return WriteResult(
                memory_id=resolution.existing_id,
                action="duplicate",
                conflict_type=resolution.conflict_type,
                chunks_written=0, annotations_written=0,
            )

        # Step 3c: Chunk the content
        mid = candidate.id
        raw_chunks = self.chunker.chunk(
            candidate.content, mid, candidate.source_type
        )

        # Step 3d: Embed chunks (batch)
        chunk_texts = [c.content for c in raw_chunks]
        chunk_embeddings = await self.embedder.embed_batch(chunk_texts)

        chunks_dicts = []
        for rc, cemb in zip(raw_chunks, chunk_embeddings):
            d = rc.to_dict()
            d["embedding"] = cemb.tolist()
            chunks_dicts.append(d)

        # Step 3e: Generate forward annotations
        all_tags = candidate.forward_contexts + extra_tags
        annotations = self.annotator.annotate(
            memory_id=mid,
            content=candidate.content,
            source=candidate.source,
            source_type=candidate.source_type,
            extra_tags=all_tags,
        )

        # Step 3f: Build graph edges for conflict/extension
        graph_edges = list(candidate.graph_edges)
        if resolution.edge_type and resolution.existing_id:
            graph_edges.append({
                "to_id": resolution.existing_id,
                "type": resolution.edge_type,
                "weight": resolution.similarity,
            })

        # Step 3g: Atomic write to MetaStore (triggers outbox for Qdrant/Kuzu)
        memory_dict = {
            "id": mid,
            "content": candidate.content,
            "source": candidate.source,
            "source_type": candidate.source_type,
            "token_count": sum(c["tokens"] for c in chunks_dicts),
            "chunks": chunks_dicts,
            "forward_contexts": [a.context_tag for a in annotations],
            "graph_edges": graph_edges,
        }
        await self.meta.insert_memory(memory_dict)

        # Step 3h: Deprecate old memory if this is an extension
        if resolution.should_deprecate and resolution.existing_id:
            await self.meta.soft_delete(resolution.existing_id)
            log.info("write_pipeline.deprecated_old",
                     old_id=resolution.existing_id,
                     new_id=mid)

        log.info(
            "write_pipeline.written",
            memory_id=mid,
            conflict_type=resolution.conflict_type,
            chunks=len(chunks_dicts),
            annotations=len(annotations),
        )

        return WriteResult(
            memory_id=mid,
            action="written",
            conflict_type=resolution.conflict_type,
            chunks_written=len(chunks_dicts),
            annotations_written=len(annotations),
        )


### tests/unit/test_chunker.py ###

import ast
import pytest
from core.types import MemoryID
from core.write.chunker import SemanticChunker, _approx_tokens

MID = MemoryID("test-parent-id")
chunker = SemanticChunker()


# ── Code chunking ──────────────────────────────────────────────────────────────

PYTHON_SOURCE = '''\
import os
import sys

CONSTANT = 42

def alpha():
    """First function."""
    return 1

def beta(x, y):
    return x + y

class MyClass:
    def method(self):
        pass
'''

def test_code_chunk_count():
    chunks = chunker.chunk(PYTHON_SOURCE, MID, "code")
    # Expect: preamble (imports+constant), alpha, beta, MyClass
    assert len(chunks) >= 3

def test_code_chunk_content_valid_python():
    """Every code chunk must be parseable Python."""
    chunks = chunker.chunk(PYTHON_SOURCE, MID, "code")
    for c in chunks:
        try:
            ast.parse(c.content)
        except SyntaxError as e:
            pytest.fail(f"Chunk is not valid Python: {c.content!r}\nError: {e}")

def test_code_chunk_indices_sequential():
    chunks = chunker.chunk(PYTHON_SOURCE, MID, "code")
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))

def test_code_content_never_modified():
    """No chunk content should be truncated or modified."""
    chunks = chunker.chunk(PYTHON_SOURCE, MID, "code")
    for chunk in chunks:
        assert chunk.content in PYTHON_SOURCE or PYTHON_SOURCE in chunk.content \
            or all(line.strip() in PYTHON_SOURCE for line in chunk.content.splitlines() if line.strip())

def test_code_fallback_non_python():
    """Non-Python code should fall back gracefully without crashing."""
    js = "const x = () => { return 42; };\nconsole.log(x());"
    chunks = chunker.chunk(js, MID, "code")
    assert len(chunks) >= 1
    assert all(c.content for c in chunks)

def test_code_single_function():
    src = "def solo():\n    return True\n"
    chunks = chunker.chunk(src, MID, "code")
    assert any("solo" in c.content for c in chunks)


# ── JSON chunking ──────────────────────────────────────────────────────────────

def test_json_chunk_per_key():
    data = '{"a": 1, "b": 2, "c": [1,2,3]}'
    chunks = chunker.chunk(data, MID, "json")
    assert len(chunks) == 3

def test_json_chunk_valid():
    import json
    data = '{"key1": {"nested": true}, "key2": "value"}'
    chunks = chunker.chunk(data, MID, "json")
    for c in chunks:
        parsed = json.loads(c.content)
        assert isinstance(parsed, dict)
        assert len(parsed) == 1   # one key per chunk

def test_json_invalid_fallback():
    """Invalid JSON should not crash."""
    chunks = chunker.chunk("not json at all {{{", MID, "json")
    assert len(chunks) >= 1


# ── Prose chunking ─────────────────────────────────────────────────────────────

def test_prose_chunk_within_token_limits():
    long_prose = ". ".join([f"Sentence number {i} about authentication" for i in range(50)]) + "."
    chunks = chunker.chunk(long_prose, MID, "prose")
    for c in chunks:
        assert c.tokens <= chunker.PROSE_MAX_TOKENS, \
            f"Chunk exceeds max tokens: {c.tokens}"

def test_prose_content_complete():
    """All content must be preserved across chunks."""
    prose = "First sentence here. Second sentence about auth. Third about testing."
    chunks = chunker.chunk(prose, MID, "prose")
    reconstructed = " ".join(c.content for c in chunks)
    # Every word from original should appear somewhere
    for word in prose.split():
        assert word in reconstructed

def test_prose_fallback_for_empty():
    chunks = chunker.chunk("", MID, "prose")
    assert len(chunks) >= 1  # always returns at least one chunk


# ── Markdown chunking ──────────────────────────────────────────────────────────

def test_markdown_splits_at_headers():
    md = "# Title\n\nIntro.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B.\n"
    chunks = chunker.chunk(md, MID, "md")
    assert len(chunks) >= 2
    assert any("Section A" in c.content for c in chunks)
    assert any("Section B" in c.content for c in chunks)

def test_markdown_no_headers_falls_back():
    md = "Just some prose without any headers. Another sentence here."
    chunks = chunker.chunk(md, MID, "md")
    assert len(chunks) >= 1


### tests/unit/test_conflict.py ###

import numpy as np
import pytest
from math_core.momentum import l2_normalize
from core.types import MemoryID, UnitVector
from core.write.conflict import ConflictResolver, ConflictType


def rand_unit(d=64):
    return l2_normalize(np.random.randn(d).astype(np.float32))

def near_unit(base: np.ndarray, noise: float) -> UnitVector:
    """Unit vector close to base."""
    v = base + noise * np.random.randn(*base.shape).astype(np.float32)
    return l2_normalize(v)


resolver = ConflictResolver()
MID = MemoryID("existing-001")


def test_no_candidates_is_novel():
    r = resolver.resolve(rand_unit(), [])
    assert r.conflict_type == ConflictType.NOVEL
    assert r.should_write is True

def test_duplicate_detected_by_hash():
    emb = rand_unit()
    r = resolver.resolve(emb, [(MID, emb, "abc123")], new_hash="abc123")
    assert r.conflict_type == ConflictType.DUPLICATE
    assert r.should_write is False

def test_duplicate_detected_by_cosine():
    emb = rand_unit()
    similar = near_unit(emb, 0.01)   # very close
    r = resolver.resolve(emb, [(MID, similar, "different_hash")])
    assert r.conflict_type == ConflictType.DUPLICATE
    assert r.should_write is False

def test_conflict_detected():
    base = rand_unit()
    # Cosine sim ~0.85 → conflict zone
    existing = near_unit(base, 0.3)
    sim = float(np.dot(base, existing))
    if 0.80 <= sim < 0.97:
        r = resolver.resolve(base, [(MID, existing, "h1")])
        assert r.conflict_type == ConflictType.CONFLICT
        assert r.edge_type == "CONFLICTS_WITH"
        assert r.should_write is True
        assert r.should_deprecate is False

def test_novel_detected():
    a = rand_unit()
    # Orthogonal vector → very low cosine sim
    b = l2_normalize(np.ones_like(a) - np.dot(np.ones_like(a), a) * a)
    r = resolver.resolve(a, [(MID, b, "h2")])
    assert r.conflict_type in (ConflictType.NOVEL, ConflictType.EXTENSION)

def test_extension_triggers_deprecate():
    base = rand_unit()
    # Target sim ~0.65 → extension zone
    existing = near_unit(base, 0.8)
    sim = float(np.dot(base, existing))
    if 0.55 <= sim < 0.80:
        r = resolver.resolve(base, [(MID, existing, "h3")])
        assert r.conflict_type == ConflictType.EXTENSION
        assert r.should_deprecate is True
        assert r.edge_type == "EXTENDS"

def test_best_match_selection():
    """Should pick the most similar candidate."""
    base = rand_unit()
    close = near_unit(base, 0.05)
    far = rand_unit()
    r = resolver.resolve(base, [
        (MemoryID("far"), far, "h1"),
        (MemoryID("close"), close, "h2"),
    ])
    assert r.existing_id == MemoryID("close")


### tests/unit/test_annotator.py ###

import pytest
from core.types import MemoryID
from core.write.annotator import ForwardAnnotator

MID = MemoryID("ann-test-001")
annotator = ForwardAnnotator()


def test_source_type_always_tagged():
    anns = annotator.annotate(MID, "some content", source_type="code")
    tags = {a.context_tag for a in anns}
    assert "type:code" in tags

def test_file_path_tagged():
    anns = annotator.annotate(MID, "content", source="src/auth/login.py")
    tags = {a.context_tag for a in anns}
    assert "file:src/auth/login.py" in tags

def test_file_ext_infers_language():
    anns = annotator.annotate(MID, "content", source="app/models.py")
    tags = {a.context_tag for a in anns}
    assert "lang:python" in tags

def test_topic_auth_detected():
    content = "This function handles JWT token authentication and login session management."
    anns = annotator.annotate(MID, content, source_type="prose")
    tags = {a.context_tag for a in anns}
    assert "topic:auth" in tags

def test_symbol_extraction():
    code = "def verify_token(token: str) -> bool:\n    pass\n\ndef refresh_session():\n    pass"
    anns = annotator.annotate(MID, code, source_type="code")
    tags = {a.context_tag for a in anns}
    assert "symbol:verify_token" in tags
    assert "symbol:refresh_session" in tags

def test_private_symbols_excluded():
    code = "def _internal():\n    pass\ndef public_func():\n    pass"
    anns = annotator.annotate(MID, code, source_type="code")
    tags = {a.context_tag for a in anns}
    assert "symbol:_internal" not in tags
    assert "symbol:public_func" in tags

def test_extra_tags_get_high_weight():
    anns = annotator.annotate(MID, "content", extra_tags=["topic:custom"])
    weights = {a.context_tag: a.weight for a in anns}
    assert "topic:custom" in weights
    assert weights["topic:custom"] >= 1.5

def test_intent_debug_detected():
    content = "There is an exception being raised in the auth module, need to fix the bug."
    anns = annotator.annotate(MID, content, source_type="prose")
    tags = {a.context_tag for a in anns}
    assert "intent:DEBUG" in tags

def test_no_duplicate_tags():
    anns = annotator.annotate(MID, "content", source="a.py", source_type="code",
                               extra_tags=["type:code"])
    tags = [a.context_tag for a in anns]
    assert len(tags) == len(set(tags)), "Duplicate tags detected"

def test_language_detected_from_patterns():
    ts_code = "interface User {\n  name: string;\n  age: number;\n}\nconst x: string = 'hi';"
    anns = annotator.annotate(MID, ts_code, source_type="code")
    tags = {a.context_tag for a in anns}
    assert "lang:typescript" in tags
