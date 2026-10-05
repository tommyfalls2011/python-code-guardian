from pathlib import Path

from codeguardian.ai.ollama import (
    OllamaError,
    build_edit_prompt,
    build_repair_prompt,
    request_ai_edit,
)


def test_repair_prompt_contains_source_and_diagnostics():
    prompt = build_repair_prompt(
        "value = definitely_not_defined\n",
        ["Undefined name: 'definitely_not_defined'"],
    )

    assert "Undefined name: 'definitely_not_defined'" in prompt
    assert "value = definitely_not_defined" in prompt
    assert "complete corrected Python source" in prompt


def test_repair_prompt_requires_source_only():
    prompt = build_repair_prompt(
        "x = 1\n",
        ["Unused definition: 'x'"],
    )

    assert "Do not use Markdown fences" in prompt
    assert "Return the complete corrected source now." in prompt

class FakeResponse:
    def __init__(self, payload):
        import json
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def test_edit_prompt_restricts_authorized_line():
    prompt = build_edit_prompt(
        diagnostic="Undefined name: x",
        line=8,
        context="7: before\n8: x\n9: after\n",
    )

    assert "ONLY line 8" in prompt
    assert '"line" MUST be 8' in prompt
    assert "smallest possible repair" in prompt


def test_request_ai_edit_accepts_bounded_delete(monkeypatch):
    import json
    import codeguardian.ai.ollama as ollama

    response = {
        "response": json.dumps({
            "operation": "delete",
            "line": 8,
            "content": "",
        })
    }

    monkeypatch.setattr(
        ollama,
        "urlopen",
        lambda request, timeout: FakeResponse(response),
    )

    edit = request_ai_edit(
        diagnostic="Undefined name: x",
        line=8,
        context="8: x\n",
        model="test-model",
    )

    assert edit.operation == "delete"
    assert edit.line == 8
    assert edit.content == ""


def test_request_ai_edit_rejects_other_line(monkeypatch):
    import json
    import pytest
    import codeguardian.ai.ollama as ollama

    response = {
        "response": json.dumps({
            "operation": "delete",
            "line": 9,
            "content": "",
        })
    }

    monkeypatch.setattr(
        ollama,
        "urlopen",
        lambda request, timeout: FakeResponse(response),
    )

    with pytest.raises(OllamaError, match="unauthorized line 9"):
        request_ai_edit(
            diagnostic="Undefined name: x",
            line=8,
            context="8: x\n9: y\n",
            model="test-model",
        )


def test_request_ai_edit_rejects_unknown_operation(monkeypatch):
    import json
    import pytest
    import codeguardian.ai.ollama as ollama

    response = {
        "response": json.dumps({
            "operation": "rewrite_file",
            "line": 8,
            "content": "",
        })
    }

    monkeypatch.setattr(
        ollama,
        "urlopen",
        lambda request, timeout: FakeResponse(response),
    )

    with pytest.raises(OllamaError, match="Unsupported"):
        request_ai_edit(
            diagnostic="Undefined name: x",
            line=8,
            context="8: x\n",
            model="test-model",
        )


def test_request_ai_edit_rejects_extra_keys(monkeypatch):
    import json
    import pytest
    import codeguardian.ai.ollama as ollama

    response = {
        "response": json.dumps({
            "operation": "delete",
            "line": 8,
            "content": "",
            "rewrite_everything": True,
        })
    }

    monkeypatch.setattr(
        ollama,
        "urlopen",
        lambda request, timeout: FakeResponse(response),
    )

    with pytest.raises(OllamaError, match="exactly"):
        request_ai_edit(
            diagnostic="Undefined name: x",
            line=8,
            context="8: x\n",
            model="test-model",
        )

