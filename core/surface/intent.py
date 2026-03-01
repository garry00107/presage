"""
IntentClassifier — maps conversation text to IntentSignal.

Phase 3 uses a heuristic classifier (rules + keyword matching).
This module is designed to be swapped for a fine-tuned classifier
in Phase 7 without changing any downstream code.

The classifier runs synchronously — it must be < 1ms on the hot path.
"""

import re
from core.nerve.models import IntentSignal


# Each rule is (IntentSignal, trigger_patterns, min_matches)
# Patterns are applied to lowercased text.
_RULES: list[tuple[IntentSignal, list[str], int]] = [
    (IntentSignal.DEBUG, [
        "error", "exception", "traceback", "not working", "failing",
        "bug", "broken", "fix", "wrong", "unexpected", "crash",
        "undefined", "null", "none", "attributeerror", "typeerror",
        "syntaxerror", "why does", "why is", "doesn't work"
    ], 1),

    (IntentSignal.IMPLEMENT, [
        "implement", "create", "build", "write a", "add a", "make a",
        "generate", "scaffold", "new function", "new class", "new endpoint",
        "how do i", "how to", "can you write", "can you create"
    ], 1),

    (IntentSignal.NAVIGATE, [
        "where is", "find", "show me", "which file", "locate",
        "where can i", "what file", "what module", "where does"
    ], 1),

    (IntentSignal.COMPARE, [
        " vs ", " versus ", "difference between", "better than",
        "compared to", "which is better", "pros and cons",
        "tradeoffs", "when to use"
    ], 1),

    (IntentSignal.REFLECT, [
        "earlier", "before", "previously", "we decided", "you said",
        "last time", "what did we", "remember when", "as we discussed",
        "going back to"
    ], 1),

    (IntentSignal.EXPLORE, [
        "what is", "what are", "explain", "tell me about", "describe",
        "overview", "how does", "understand", "what does", "meaning of"
    ], 1),
]

# Regex for extracting Python symbols from text
_SYMBOL_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\s*(?:\(|\.)', re.MULTILINE)

# Regex for file paths
_FILE_RE = re.compile(
    r'(?:^|[\s\'"`])([a-zA-Z0-9_\-./]+\.(?:py|js|ts|go|rs|json|yaml|yml|md|txt))',
    re.MULTILINE
)


class IntentClassifier:
    """
    Fast, rule-based intent classifier.
    Returns the highest-priority matching intent.
    Priority order mirrors _RULES list (DEBUG first — most actionable).
    """

    def classify(self, text: str) -> IntentSignal:
        """
        Classify text into an IntentSignal.
        O(n·k) where n=len(text), k=total keywords. Runs in < 0.5ms.
        """
        lower = text.lower()
        for signal, patterns, min_matches in _RULES:
            hits = sum(1 for p in patterns if p in lower)
            if hits >= min_matches:
                return signal
        return IntentSignal.UNKNOWN

    def extract_symbols(self, text: str) -> list[str]:
        """
        Extract likely code symbol names (function calls, attribute access).
        Filters out common English words and short tokens.
        """
        _STOPWORDS = {
            "the", "and", "for", "this", "that", "with", "from",
            "import", "class", "return", "print", "true", "false",
            "none", "self", "args", "kwargs", "str", "int", "list",
            "dict", "bool", "type", "any", "not", "new", "get",
        }
        matches = _SYMBOL_RE.findall(text)
        return [m for m in matches
                if m.lower() not in _STOPWORDS and len(m) > 3][:10]

    def extract_files(self, text: str) -> list[str]:
        """Extract file path references from text."""
        return [m.strip(" '\"`") for m in _FILE_RE.findall(text)][:5]

