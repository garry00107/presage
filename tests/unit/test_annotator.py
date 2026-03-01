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
