import json
import pytest
from pathlib import Path
from codeguardian.ai import ollama

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

def test_multi_edit_prompt_uses_original_lines():
    prompt = ollama.build_multi_edit_prompt(
        diagnostic="WARNING: mutable default",
        line=6,
        context="6: def example(items=[]):\n",
    )

    assert "original source line 6" in prompt
    assert "ORIGINAL source" in prompt
    assert "between 1 and 3 edits" in prompt


def test_request_ai_edits_accepts_transaction(
    monkeypatch,
):
    response_body = {
        "response": json.dumps(
            {
                "edits": [
                    {
                        "operation": "replace",
                        "line": 6,
                        "content": "def example(items=None):",
                    },
                    {
                        "operation": "insert_after",
                        "line": 6,
                        "content": (
                            "    if items is None:\n"
                            "        items = []"
                        ),
                    },
                ]
            }
        )
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(response_body).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert timeout == 300
        return FakeResponse()

    monkeypatch.setattr(ollama, "urlopen", fake_urlopen)

    edits = ollama.request_ai_edits(
        diagnostic="WARNING: mutable default",
        line=6,
        context="6: def example(items=[]):\n",
        model="test-model",
    )

    assert len(edits) == 2
    assert edits[0].operation == "replace"
    assert edits[0].line == 6
    assert edits[1].operation == "insert_after"
    assert edits[1].line == 6


def test_request_ai_edits_rejects_too_many(
    monkeypatch,
):
    response_body = {
        "response": json.dumps(
            {
                "edits": [
                    {
                        "operation": "replace",
                        "line": number,
                        "content": "value = 1",
                    }
                    for number in range(1, 5)
                ]
            }
        )
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(response_body).encode("utf-8")

    monkeypatch.setattr(
        ollama,
        "urlopen",
        lambda request, timeout: FakeResponse(),
    )

    with pytest.raises(
        ollama.OllamaError,
        match="too many",
    ):
        ollama.request_ai_edits(
            diagnostic="test",
            line=1,
            context="1: value = 0\n",
            model="test-model",
        )


def test_request_ai_edits_rejects_extra_keys(
    monkeypatch,
):
    response_body = {
        "response": json.dumps(
            {
                "edits": [
                    {
                        "operation": "delete",
                        "line": 1,
                        "content": "",
                    }
                ],
                "explanation": "extra",
            }
        )
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(response_body).encode("utf-8")

    monkeypatch.setattr(
        ollama,
        "urlopen",
        lambda request, timeout: FakeResponse(),
    )

    with pytest.raises(
        ollama.OllamaError,
        match="only edits",
    ):
        ollama.request_ai_edits(
            diagnostic="test",
            line=1,
            context="1: bad_name\n",
            model="test-model",
        )


def test_request_ai_edits_rejects_boolean_line(
    monkeypatch,
):
    response_body = {
        "response": json.dumps(
            {
                "edits": [
                    {
                        "operation": "delete",
                        "line": True,
                        "content": "",
                    }
                ]
            }
        )
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(response_body).encode("utf-8")

    monkeypatch.setattr(
        ollama,
        "urlopen",
        lambda request, timeout: FakeResponse(),
    )

    with pytest.raises(
        ollama.OllamaError,
        match="positive integer",
    ):
        ollama.request_ai_edits(
            diagnostic="test",
            line=1,
            context="1: bad_name\n",
            model="test-model",
        )

def test_multi_edit_prompt_groups_multiline_insertions():
    prompt = ollama.build_multi_edit_prompt(
        diagnostic=(
            "WARNING: line 6: Mutable default argument"
        ),
        line=6,
        context=(
            "6: def example(items=[]):\n"
            "7:     return len(items)\n"
        ),
        max_edits=3,
    )

    assert (
        "put ALL of them in ONE insert_before or "
        "insert_after edit"
    ) in prompt
    assert (
        "Do not split a logical multi-line block"
    ) in prompt
    assert "ORIGINAL source" in prompt

