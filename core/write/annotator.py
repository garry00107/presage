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

