from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .edit import AIEdit
from .proposal import AIRepairProposal


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot produce a proposal."""


def build_repair_prompt(
    source: str,
    diagnostics: list[str],
) -> str:
    diagnostic_text = "\n".join(
        f"- {diagnostic}" for diagnostic in diagnostics
    )

    return f"""You are repairing Python 3 source code.

You MUST return only the complete corrected Python source.
Do not use Markdown fences.
Do not explain the repair outside the source.

The Python Code Guardian diagnostics are:

{diagnostic_text}

Original source:

---BEGIN SOURCE---
{source}
---END SOURCE---

Return the complete corrected source now.
"""


def request_ai_repair(
    *,
    file,
    source: str,
    diagnostics: list[str],
    model: str,
    base_url: str = "http://127.0.0.1:11434",
) -> AIRepairProposal:
    prompt = build_repair_prompt(source, diagnostics)

    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }
    ).encode("utf-8")

    request = Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise OllamaError(
            f"Ollama HTTP error {exc.code}"
        ) from exc
    except URLError as exc:
        raise OllamaError(
            f"Unable to connect to Ollama: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise OllamaError("Ollama request timed out") from exc

    proposed_source = data.get("response")

    if not isinstance(proposed_source, str):
        raise OllamaError(
            "Ollama response did not contain a string 'response' field"
        )

    proposed_source = proposed_source.strip()

    if proposed_source.startswith("```"):
        lines = proposed_source.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        proposed_source = "\n".join(lines).strip() + "\n"

    return AIRepairProposal(
        file=file,
        original_source=source,
        proposed_source=proposed_source,
        reason="AI-generated repair proposal",
        model=model,
    )

def build_edit_prompt(
    *,
    diagnostic: str,
    line: int,
    context: str,
) -> str:
    return f"""You are proposing ONE tiny Python source edit.

Diagnostic:
{diagnostic}

You are authorized to modify ONLY line {line}.

Allowed operations:
delete
replace
insert_before
insert_after

Return exactly one JSON object with these keys:
"operation"
"line"
"content"

Rules:
- "line" MUST be {line}.
- Make the smallest possible repair.
- Preserve unrelated behavior.
- Do not modify unrelated code.
- For delete, content MUST be an empty string.
- Return JSON only.
- No Markdown.
- No explanation.

Context:
{context}
"""


def request_ai_edit(
    *,
    diagnostic: str,
    line: int,
    context: str,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
) -> AIEdit:
    prompt = build_edit_prompt(
        diagnostic=diagnostic,
        line=line,
        context=context,
    )

    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
            },
        }
    ).encode("utf-8")

    request = Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )
    except HTTPError as exc:
        raise OllamaError(
            f"Ollama HTTP error {exc.code}"
        ) from exc
    except URLError as exc:
        raise OllamaError(
            f"Unable to connect to Ollama: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise OllamaError(
            "Ollama request timed out"
        ) from exc

    raw = response_data.get("response")

    if not isinstance(raw, str):
        raise OllamaError(
            "Ollama response did not contain a string 'response' field"
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OllamaError(
            "Ollama did not return valid JSON"
        ) from exc

    required = {"operation", "line", "content"}

    if set(data) != required:
        raise OllamaError(
            "AI edit must contain exactly: content, line, operation"
        )

    operation = data["operation"]
    proposed_line = data["line"]
    content = data["content"]

    if not isinstance(operation, str):
        raise OllamaError("AI edit operation must be a string")

    if operation not in {
        "delete",
        "replace",
        "insert_before",
        "insert_after",
    }:
        raise OllamaError(
            f"Unsupported AI edit operation: {operation}"
        )

    if type(proposed_line) is not int:
        raise OllamaError("AI edit line must be an integer")

    if proposed_line != line:
        raise OllamaError(
            f"AI attempted unauthorized line {proposed_line}"
        )

    if not isinstance(content, str):
        raise OllamaError("AI edit content must be a string")

    if operation == "delete" and content:
        raise OllamaError(
            "AI delete operation must have empty content"
        )

    if operation != "delete" and not content:
        raise OllamaError(
            f"AI {operation} operation requires content"
        )

    return AIEdit(
        operation=operation,
        line=proposed_line,
        content=content,
    )

def build_multi_edit_prompt(
    *,
    diagnostic: str,
    line: int,
    context: str,
    max_edits: int = 3,
) -> str:
    return f"""You are proposing a tiny Python repair transaction.

Diagnostic:
{diagnostic}

The diagnostic is centered on original source line {line}.

Return exactly one JSON object with this shape:
{{
  "edits": [
    {{
      "operation": "replace",
      "line": {line},
      "content": "replacement source"
    }}
  ]
}}

Allowed operations:
delete
replace
insert_before
insert_after

Rules:
- Return between 1 and {max_edits} edits.
- Every line number refers to the ORIGINAL source shown below.
- If several new lines must be inserted together, put ALL of them in ONE insert_before or insert_after edit using \n inside content.
- Do not split a logical multi-line block across different original line anchors.
- Modify only lines necessary to repair the diagnostic.
- Preserve unrelated behavior.
- For delete, content MUST be an empty string.
- For replace/insert operations, content MUST be nonempty.
- Do not return duplicate destructive edits for one line.
- Return JSON only.
- No Markdown.
- No explanation.

Context:
{context}
"""


def request_ai_edits(
    *,
    diagnostic: str,
    line: int,
    context: str,
    model: str,
    max_edits: int = 3,
    base_url: str = "http://127.0.0.1:11434",
) -> list[AIEdit]:
    if max_edits < 1 or max_edits > 3:
        raise ValueError("max_edits must be between 1 and 3")

    prompt = build_multi_edit_prompt(
        diagnostic=diagnostic,
        line=line,
        context=context,
        max_edits=max_edits,
    )

    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
            },
        }
    ).encode("utf-8")

    request = Request(
        f"{base_url.rstrip("/")}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )
    except HTTPError as exc:
        raise OllamaError(
            f"Ollama HTTP error {exc.code}"
        ) from exc
    except URLError as exc:
        raise OllamaError(
            f"Unable to connect to Ollama: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise OllamaError(
            "Ollama request timed out"
        ) from exc

    raw = response_data.get("response")

    if not isinstance(raw, str):
        raise OllamaError(
            "Ollama response does not contain a response string"
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OllamaError(
            "Ollama returned invalid JSON"
        ) from exc

    if not isinstance(data, dict):
        raise OllamaError(
            "AI multi-edit response must be a JSON object"
        )

    if set(data) != {"edits"}:
        raise OllamaError(
            "AI multi-edit response must contain only edits"
        )

    raw_edits = data["edits"]

    if not isinstance(raw_edits, list):
        raise OllamaError("AI edits must be a JSON array")

    if not raw_edits:
        raise OllamaError("AI edits array must not be empty")

    if len(raw_edits) > max_edits:
        raise OllamaError("AI returned too many edits")

    edits: list[AIEdit] = []

    for raw_edit in raw_edits:
        if not isinstance(raw_edit, dict):
            raise OllamaError(
                "each AI edit must be a JSON object"
            )

        if set(raw_edit) != {
            "operation",
            "line",
            "content",
        }:
            raise OllamaError(
                "AI edit contains invalid keys"
            )

        operation = raw_edit["operation"]
        edit_line = raw_edit["line"]
        content = raw_edit["content"]

        if operation not in {
            "delete",
            "replace",
            "insert_before",
            "insert_after",
        }:
            raise OllamaError(
                "AI edit contains an invalid operation"
            )

        if type(edit_line) is not int or edit_line < 1:
            raise OllamaError(
                "AI edit line must be a positive integer"
            )

        if not isinstance(content, str):
            raise OllamaError(
                "AI edit content must be a string"
            )

        if operation == "delete":
            if content:
                raise OllamaError(
                    "delete edit content must be empty"
                )
        elif not content:
            raise OllamaError(
                "non-delete edit content must not be empty"
            )

        edits.append(
            AIEdit(
                operation=operation,
                line=edit_line,
                content=content,
            )
        )

    return edits

