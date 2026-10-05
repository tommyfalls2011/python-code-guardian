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

