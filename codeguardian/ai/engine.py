from __future__ import annotations

import ast
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..analyzer import analyze_file
from ..repair import create_backup, restore_backup
from ..scanner import Diagnostic
from ..validator import RepairValidator
from .edit import (
    AIEdit,
    apply_ai_edit,
    apply_ai_edits,
    normalize_ai_edits,
)
from .localized import extract_source_window
from .ollama import request_ai_edit, request_ai_edits


@dataclass(frozen=True)
class AIEditEvaluation:
    accepted: bool
    reason: str
    edit: AIEdit | None
    candidate_source: str | None
    before_diagnostics: list[Diagnostic]
    after_diagnostics: list[Diagnostic]


AIEditProvider = Callable[..., AIEdit]
AIEditsProvider = Callable[..., list[AIEdit]]


@dataclass(frozen=True)
class AIRepairTransaction:
    applied: bool
    reason: str
    backup: Path | None
    evaluation: AIEditEvaluation | AIEditsEvaluation


def apply_evaluated_ai_edit(
    path: Path,
    evaluation: AIEditEvaluation | AIEditsEvaluation,
) -> AIRepairTransaction:
    path = path.resolve()

    if not evaluation.accepted:
        return AIRepairTransaction(
            applied=False,
            reason="AI edit was not accepted",
            backup=None,
            evaluation=evaluation,
        )

    if evaluation.candidate_source is None:
        return AIRepairTransaction(
            applied=False,
            reason="accepted AI edit has no candidate source",
            backup=None,
            evaluation=evaluation,
        )

    original_source = path.read_text(encoding="utf-8")

    if not path.is_file():
        raise FileNotFoundError(path)

    backup = create_backup(path)

    try:
        path.write_text(
            evaluation.candidate_source,
            encoding="utf-8",
        )

        final_source = path.read_text(encoding="utf-8")

        if final_source != evaluation.candidate_source:
            restore_backup(path, backup)
            return AIRepairTransaction(
                applied=False,
                reason=(
                    "written source did not match candidate; "
                    "repair rolled back"
                ),
                backup=backup,
                evaluation=evaluation,
            )

        syntax = RepairValidator().validate_source(
            final_source,
            filename=str(path),
        )

        if not syntax.passed:
            restore_backup(path, backup)
            return AIRepairTransaction(
                applied=False,
                reason=(
                    "final syntax validation failed; "
                    "repair rolled back"
                ),
                backup=backup,
                evaluation=evaluation,
            )

        final_diagnostics = analyze_file(path).diagnostics

        expected = sorted(
            _diagnostic_key(item)
            for item in evaluation.after_diagnostics
        )
        actual = sorted(
            _diagnostic_key(item)
            for item in final_diagnostics
        )

        if actual != expected:
            restore_backup(path, backup)
            return AIRepairTransaction(
                applied=False,
                reason=(
                    "final diagnostics differed from approved "
                    "candidate; repair rolled back"
                ),
                backup=backup,
                evaluation=evaluation,
            )

    except Exception:
        try:
            restore_backup(path, backup)
        except Exception:
            path.write_text(original_source, encoding="utf-8")
        raise

    return AIRepairTransaction(
        applied=True,
        reason="AI repair applied and validated",
        backup=backup,
        evaluation=evaluation,
    )


def _diagnostic_key(diagnostic: Diagnostic) -> tuple[str, str]:
    return (
        diagnostic.severity.upper(),
        diagnostic.message,
    )


def _constant_false_body_statements(
    source: str,
    line: int,
) -> list[str] | None:
    """Return statements belonging to the targeted if False body."""
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue

        if not (
            isinstance(node.test, ast.Constant)
            and node.test.value is False
        ):
            continue

        end_line = getattr(
            node,
            "end_lineno",
            node.lineno,
        )

        if not (
            node.lineno <= line <= end_line
        ):
            continue

        return [
            ast.dump(
                statement,
                annotate_fields=True,
                include_attributes=False,
            )
            for statement in node.body
        ]

    return None


def _statement_exists_outside_false_block(
    source: str,
    statement_dump: str,
) -> bool:
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue

        current_dump = ast.dump(
            node,
            annotate_fields=True,
            include_attributes=False,
        )

        if current_dump != statement_dump:
            continue

        current: ast.AST = node
        guarded = False

        while current in parents:
            current = parents[current]

            if not isinstance(current, ast.If):
                continue

            if (
                isinstance(current.test, ast.Constant)
                and current.test.value is False
            ):
                guarded = True
                break

        if not guarded:
            return True

    return False


def _exposes_if_false_body(
    source: str,
    candidate: str,
    diagnostic: Diagnostic,
) -> bool:
    if "if False" not in diagnostic.message:
        return False

    body = _constant_false_body_statements(
        source,
        diagnostic.line,
    )

    if body is None:
        return False

    return any(
        _statement_exists_outside_false_block(
            candidate,
            statement,
        )
        for statement in body
    )


def deterministic_duplicate_import_edit(
    source: str,
    diagnostic: Diagnostic,
) -> AIEdit | None:
    """Return a safe deletion for a standalone duplicate import."""
    if not diagnostic.message.startswith(
        "Duplicate import: "
    ):
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    target: ast.Import | ast.ImportFrom | None = None

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Import, ast.ImportFrom),
        ):
            continue

        if node.lineno == diagnostic.line:
            target = node
            break

    if target is None:
        return None

    if len(target.names) != 1:
        return None

    alias = target.names[0]

    if isinstance(target, ast.Import):
        binding = (
            alias.asname
            or alias.name.split(".")[0]
        )
    else:
        if alias.name == "*":
            return None
        binding = alias.asname or alias.name

    expected = f"Duplicate import: '{binding}'"

    if diagnostic.message != expected:
        return None

    lines = source.splitlines()

    if not (
        1 <= diagnostic.line <= len(lines)
    ):
        return None

    line = lines[diagnostic.line - 1]

    try:
        line_tree = ast.parse(line.lstrip())
    except SyntaxError:
        return None

    if len(line_tree.body) != 1:
        return None

    line_node = line_tree.body[0]

    if not isinstance(
        line_node,
        (ast.Import, ast.ImportFrom),
    ):
        return None

    target_dump = ast.dump(
        target,
        include_attributes=False,
    )

    identical_earlier_sibling = False

    for parent in ast.walk(tree):
        for _field, value in ast.iter_fields(parent):
            if not isinstance(value, list):
                continue

            target_index = None

            for index, item in enumerate(value):
                if item is target:
                    target_index = index
                    break

            if target_index is None:
                continue

            identical_earlier_sibling = any(
                isinstance(
                    item,
                    (ast.Import, ast.ImportFrom),
                )
                and ast.dump(
                    item,
                    include_attributes=False,
                ) == target_dump
                for item in value[:target_index]
            )
            break

        if identical_earlier_sibling:
            break

    if not identical_earlier_sibling:
        return None

    return AIEdit(
        operation="delete",
        line=diagnostic.line,
    )




def deterministic_unused_import_edit(
    source: str,
    diagnostic: Diagnostic,
) -> AIEdit | None:
    """Delete a standalone single-name import reported unused."""
    prefix = "Unused import: '"

    if not (
        diagnostic.message.startswith(prefix)
        and diagnostic.message.endswith("'")
    ):
        return None

    binding = diagnostic.message[len(prefix):-1]

    if not binding.isidentifier():
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    target: ast.Import | ast.ImportFrom | None = None

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Import, ast.ImportFrom),
        ):
            continue

        if node.lineno != diagnostic.line:
            continue

        target = node
        break

    if target is None:
        return None

    if len(target.names) != 1:
        return None

    if getattr(target, "end_lineno", target.lineno) != target.lineno:
        return None

    alias = target.names[0]

    if isinstance(target, ast.Import):
        actual_binding = (
            alias.asname
            or alias.name.split(".")[0]
        )
    else:
        if target.module == "__future__":
            return None

        if alias.name == "*":
            return None

        actual_binding = alias.asname or alias.name

    if actual_binding != binding:
        return None

    lines = source.splitlines()

    if not (
        1 <= diagnostic.line <= len(lines)
    ):
        return None

    line = lines[diagnostic.line - 1]

    try:
        line_tree = ast.parse(line.lstrip())
    except SyntaxError:
        return None

    if len(line_tree.body) != 1:
        return None

    line_node = line_tree.body[0]

    if not isinstance(
        line_node,
        (ast.Import, ast.ImportFrom),
    ):
        return None

    if len(line_node.names) != 1:
        return None

    return AIEdit(
        operation="delete",
        line=diagnostic.line,
    )

def _side_effect_free_expression(
    node: ast.AST,
) -> bool:
    """Return whether deleting this expression is conservatively safe."""
    if isinstance(node, ast.Constant):
        return True

    if isinstance(node, (ast.Tuple, ast.List)):
        return all(
            _side_effect_free_expression(item)
            for item in node.elts
        )

    return False


def deterministic_mutable_default_edits(
    source: str,
    diagnostic: Diagnostic,
) -> list[AIEdit] | None:
    """Repair one simple mutable default with a bounded two-edit transaction."""
    if not diagnostic.message.startswith(
        "Mutable default argument"
    ):
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
        and node.lineno == diagnostic.line
    ]

    if len(functions) != 1:
        return None

    function = functions[0]

    if getattr(
        function,
        "end_lineno",
        function.lineno,
    ) == function.lineno:
        return None

    lines = source.splitlines()

    if not (1 <= function.lineno <= len(lines)):
        return None

    header = lines[function.lineno - 1]

    try:
        header_tree = ast.parse(
            header.lstrip() + "\n    pass\n"
        )
    except SyntaxError:
        return None

    if (
        len(header_tree.body) != 1
        or not isinstance(
            header_tree.body[0],
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
    ):
        return None

    positional = (
        list(function.args.posonlyargs)
        + list(function.args.args)
    )

    positional_defaults = list(function.args.defaults)
    positional_with_defaults = positional[
        len(positional) - len(positional_defaults):
    ]

    candidates: list[
        tuple[str, ast.expr, str]
    ] = []

    def mutable_factory(
        node: ast.expr,
    ) -> str | None:
        if isinstance(node, ast.List):
            if node.elts:
                return None
            return "[]"

        if isinstance(node, ast.Dict):
            if node.keys or node.values:
                return None
            return "{}"

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "set"
            and not node.args
            and not node.keywords
        ):
            return "set()"

        return None

    for argument, default in zip(
        positional_with_defaults,
        positional_defaults,
    ):
        factory = mutable_factory(default)
        if factory is not None:
            candidates.append(
                (argument.arg, default, factory)
            )

    for argument, default in zip(
        function.args.kwonlyargs,
        function.args.kw_defaults,
    ):
        if default is None:
            continue

        factory = mutable_factory(default)
        if factory is not None:
            candidates.append(
                (argument.arg, default, factory)
            )

    if len(candidates) != 1:
        return None

    name, default, factory = candidates[0]

    if (
        getattr(default, "lineno", None)
        != function.lineno
        or getattr(default, "end_lineno", None)
        != function.lineno
    ):
        return None

    start = default.col_offset
    end = default.end_col_offset

    leading = len(header) - len(header.lstrip())
    start -= leading
    end -= leading

    stripped = header.lstrip()

    if not (
        0 <= start < end <= len(stripped)
    ):
        return None

    replacement_header = (
        stripped[:start]
        + "None"
        + stripped[end:]
    )

    indentation = header[:leading] + "    "

    initializer = (
        f"{indentation}if {name} is None:\n"
        f"{indentation}    {name} = {factory}"
    )

    return [
        AIEdit(
            operation="replace",
            line=function.lineno,
            content=(
                header[:leading]
                + replacement_header
            ),
        ),
        AIEdit(
            operation="insert_after",
            line=function.lineno,
            content=initializer,
        ),
    ]


def deterministic_unreachable_code_edit(
    source: str,
    diagnostic: Diagnostic,
) -> AIEdit | None:
    """Delete one provably unreachable single-line sibling statement."""
    if diagnostic.message != (
        "Unreachable code after unconditional control transfer."
    ):
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    target: ast.stmt | None = None
    containing_list: list[ast.stmt] | None = None
    target_index: int | None = None

    for parent in ast.walk(tree):
        for _field, value in ast.iter_fields(parent):
            if not isinstance(value, list):
                continue

            for index, item in enumerate(value):
                if not isinstance(item, ast.stmt):
                    continue

                if item.lineno != diagnostic.line:
                    continue

                if target is not None:
                    return None

                target = item
                containing_list = value
                target_index = index

    if (
        target is None
        or containing_list is None
        or target_index is None
    ):
        return None

    if getattr(target, "end_lineno", target.lineno) != target.lineno:
        return None

    if target_index == 0:
        return None

    terminators = (
        ast.Return,
        ast.Raise,
        ast.Break,
        ast.Continue,
    )

    if not any(
        isinstance(item, terminators)
        for item in containing_list[:target_index]
    ):
        return None

    lines = source.splitlines()

    if not (1 <= diagnostic.line <= len(lines)):
        return None

    line = lines[diagnostic.line - 1]

    try:
        line_tree = ast.parse(line.lstrip())
    except SyntaxError:
        return None

    if len(line_tree.body) != 1:
        return None

    return AIEdit(
        operation="delete",
        line=diagnostic.line,
    )


def deterministic_unused_definition_edit(
    source: str,
    diagnostic: Diagnostic,
) -> AIEdit | None:
    """Delete a standalone unused assignment only when its RHS is inert."""
    prefix = "Unused definition: '"

    if not (
        diagnostic.message.startswith(prefix)
        and diagnostic.message.endswith("'")
    ):
        return None

    name = diagnostic.message[
        len(prefix):-1
    ]

    if not name.isidentifier():
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    target: ast.Assign | None = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        if node.lineno != diagnostic.line:
            continue

        if len(node.targets) != 1:
            return None

        assignment_target = node.targets[0]

        if not (
            isinstance(assignment_target, ast.Name)
            and assignment_target.id == name
        ):
            return None

        target = node
        break

    if target is None:
        return None

    if getattr(target, "end_lineno", target.lineno) != target.lineno:
        return None

    if not _side_effect_free_expression(target.value):
        return None

    lines = source.splitlines()

    if not (
        1 <= diagnostic.line <= len(lines)
    ):
        return None

    try:
        line_tree = ast.parse(
            lines[diagnostic.line - 1].lstrip()
        )
    except SyntaxError:
        return None

    if len(line_tree.body) != 1:
        return None

    line_node = line_tree.body[0]

    if not isinstance(line_node, ast.Assign):
        return None

    return AIEdit(
        operation="delete",
        line=diagnostic.line,
    )

def _analyze_source(source: str) -> list[Diagnostic]:
    with tempfile.TemporaryDirectory(
        prefix="codeguardian-ai-"
    ) as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        return analyze_file(path).diagnostics


def _numbered_context(
    source: str,
    line: int,
    *,
    context_lines: int,
) -> str:
    window = extract_source_window(
        source,
        line,
        context=context_lines,
    )

    return "".join(
        f"{number}: {text}"
        for number, text in enumerate(
            window.source.splitlines(keepends=True),
            start=window.start_line,
        )
    )


def evaluate_ai_edit(
    *,
    source: str,
    diagnostic: Diagnostic,
    model: str,
    provider: AIEditProvider = request_ai_edit,
    context_lines: int = 3,
) -> AIEditEvaluation:
    before = _analyze_source(source)

    if diagnostic.message.startswith(
        "Mutable default argument"
    ):
        return AIEditEvaluation(
            accepted=False,
            reason=(
                "mutable-default repair requires a bounded "
                "multi-edit transaction"
            ),
            edit=None,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=before,
        )

    context = _numbered_context(
        source,
        diagnostic.line,
        context_lines=context_lines,
    )

    edit = provider(
        diagnostic=(
            f"{diagnostic.severity}: "
            f"line {diagnostic.line}: "
            f"{diagnostic.message}"
        ),
        line=diagnostic.line,
        context=context,
        model=model,
    )

    candidate = apply_ai_edit(source, edit)

    syntax = RepairValidator().validate_source(candidate)

    if not syntax.passed:
        return AIEditEvaluation(
            accepted=False,
            reason="candidate has invalid Python syntax",
            edit=edit,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=syntax.diagnostics,
        )

    after = _analyze_source(candidate)

    if _exposes_if_false_body(
        source,
        candidate,
        diagnostic,
    ):
        return AIEditEvaluation(
            accepted=False,
            reason=(
                "candidate exposes code previously guarded "
                "by 'if False'"
            ),
            edit=edit,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=after,
        )

    before_errors = {
        _diagnostic_key(item)
        for item in before
        if item.severity.upper() == "ERROR"
    }
    after_errors = {
        _diagnostic_key(item)
        for item in after
        if item.severity.upper() == "ERROR"
    }

    new_errors = after_errors - before_errors

    if new_errors:
        return AIEditEvaluation(
            accepted=False,
            reason="candidate introduces a new error",
            edit=edit,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=after,
        )

    target_key = _diagnostic_key(diagnostic)

    before_target_count = sum(
        _diagnostic_key(item) == target_key
        for item in before
    )
    after_target_count = sum(
        _diagnostic_key(item) == target_key
        for item in after
    )

    duplicate_import_reduction = (
        diagnostic.message.startswith("Duplicate import: ")
        and edit.operation == "delete"
        and edit.line == diagnostic.line
        and after_target_count == before_target_count - 1
    )

    target_still_present = after_target_count > 0

    if (
        target_still_present
        and not duplicate_import_reduction
    ):
        return AIEditEvaluation(
            accepted=False,
            reason="target diagnostic was not repaired",
            edit=edit,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=after,
        )

    if len(after) >= len(before):
        return AIEditEvaluation(
            accepted=False,
            reason="candidate does not reduce diagnostics",
            edit=edit,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=after,
        )

    return AIEditEvaluation(
        accepted=True,
        reason="candidate safely reduces diagnostics",
        edit=edit,
        candidate_source=candidate,
        before_diagnostics=before,
        after_diagnostics=after,
    )

@dataclass(frozen=True)
class AIEditsEvaluation:
    accepted: bool
    reason: str
    edits: list[AIEdit]
    candidate_source: str | None
    before_diagnostics: list[Diagnostic]
    after_diagnostics: list[Diagnostic]


def evaluate_ai_edits(
    *,
    source: str,
    diagnostic: Diagnostic,
    model: str,
    provider: AIEditsProvider = request_ai_edits,
    context_lines: int = 3,
    max_edits: int = 3,
) -> AIEditsEvaluation:
    before = _analyze_source(source)

    window = extract_source_window(
        source,
        diagnostic.line,
        context=context_lines,
    )

    context = "".join(
        f"{number}: {text}"
        for number, text in enumerate(
            window.source.splitlines(keepends=True),
            start=window.start_line,
        )
    )

    edits = provider(
        diagnostic=(
            f"{diagnostic.severity}: "
            f"line {diagnostic.line}: "
            f"{diagnostic.message}"
        ),
        line=diagnostic.line,
        context=context,
        model=model,
        max_edits=max_edits,
    )

    edits = normalize_ai_edits(edits)

    if not edits:
        return AIEditsEvaluation(
            accepted=False,
            reason="AI returned an empty edit transaction",
            edits=[],
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=before,
        )

    if len(edits) > max_edits:
        return AIEditsEvaluation(
            accepted=False,
            reason="AI returned too many edits",
            edits=edits,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=before,
        )

    for edit in edits:
        if not (
            window.start_line
            <= edit.line
            <= window.end_line
        ):
            return AIEditsEvaluation(
                accepted=False,
                reason=(
                    "AI attempted an edit outside the "
                    "authorized source window"
                ),
                edits=edits,
                candidate_source=None,
                before_diagnostics=before,
                after_diagnostics=before,
            )

    try:
        candidate = apply_ai_edits(
            source,
            edits,
            max_edits=max_edits,
        )
    except ValueError as exc:
        return AIEditsEvaluation(
            accepted=False,
            reason=f"invalid AI edit transaction: {exc}",
            edits=edits,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=before,
        )

    syntax = RepairValidator().validate_source(candidate)

    if not syntax.passed:
        return AIEditsEvaluation(
            accepted=False,
            reason="candidate has invalid Python syntax",
            edits=edits,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=syntax.diagnostics,
        )

    after = _analyze_source(candidate)

    if _exposes_if_false_body(
        source,
        candidate,
        diagnostic,
    ):
        return AIEditsEvaluation(
            accepted=False,
            reason=(
                "candidate exposes code previously guarded "
                "by 'if False'"
            ),
            edits=edits,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=after,
        )

    before_errors = {
        _diagnostic_key(item)
        for item in before
        if item.severity.upper() == "ERROR"
    }
    after_errors = {
        _diagnostic_key(item)
        for item in after
        if item.severity.upper() == "ERROR"
    }

    if after_errors - before_errors:
        return AIEditsEvaluation(
            accepted=False,
            reason="candidate introduces a new error",
            edits=edits,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=after,
        )

    target_key = _diagnostic_key(diagnostic)

    if any(
        _diagnostic_key(item) == target_key
        for item in after
    ):
        return AIEditsEvaluation(
            accepted=False,
            reason="target diagnostic was not repaired",
            edits=edits,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=after,
        )

    if len(after) >= len(before):
        return AIEditsEvaluation(
            accepted=False,
            reason="candidate does not reduce diagnostics",
            edits=edits,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=after,
        )

    return AIEditsEvaluation(
        accepted=True,
        reason="candidate safely reduces diagnostics",
        edits=edits,
        candidate_source=candidate,
        before_diagnostics=before,
        after_diagnostics=after,
    )
