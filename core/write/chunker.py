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

