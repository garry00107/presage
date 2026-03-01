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

