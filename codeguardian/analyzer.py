from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .scanner import Diagnostic
from .scope import (
    Scope,
    analyze_scope,
    find_undefined_names,
    find_unused_definitions,
)


@dataclass
class AnalysisResult:
    diagnostics: list[Diagnostic]


class PythonAnalyzer(ast.NodeVisitor):
    """Analyze Python source code using Python's AST."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.diagnostics: list[Diagnostic] = []
        self.imports: dict[str, int] = {}
        self.import_scopes: list[set[str]] = [set()]
        self.definition_scopes: list[set[str]] = [set()]
        self.callable_scopes: list[set[str]] = [set()]
        self.loop_depth = 0

    def diagnostic(
        self,
        node: ast.AST,
        severity: str,
        message: str,
    ) -> None:
        self.diagnostics.append(
            Diagnostic(
                file=self.path,
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                severity=severity,
                message=message,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]

            if name in self.import_scopes[-1]:
                self.diagnostic(
                    node,
                    "WARNING",
                    f"Duplicate import: '{name}'",
                )
            self.import_scopes[-1].add(name)
            self.imports.setdefault(name, node.lineno)

            self.definition_scopes[-1].add(name)

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "__future__":
            self.generic_visit(node)
            return

        for alias in node.names:
            if alias.name == "*":
                self.diagnostic(
                    node,
                    "WARNING",
                    "Wildcard import makes static analysis less reliable.",
                )
                continue

            name = alias.asname or alias.name

            if name in self.import_scopes[-1]:
                self.diagnostic(
                    node,
                    "WARNING",
                    f"Duplicate import: '{name}'",
                )
            self.import_scopes[-1].add(name)
            self.imports.setdefault(name, node.lineno)

            self.definition_scopes[-1].add(name)

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name in self.callable_scopes[-1]:
            self.diagnostic(
                node,
                "WARNING",
                f"Name '{node.name}' is defined more than once.",
            )
        elif node.name in self.definition_scopes[-1]:
            self.diagnostic(
                node,
                "WARNING",
                f"Name '{node.name}' is rebound by a class definition.",
            )

        self.definition_scopes[-1].add(node.name)
        self.callable_scopes[-1].add(node.name)
        self.definition_scopes.append(set())
        self.import_scopes.append(set())
        self.callable_scopes.append(set())
        try:
            self.generic_visit(node)
        finally:
            self.callable_scopes.pop()
            self.import_scopes.pop()
            self.definition_scopes.pop()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.definition_scopes[-1].add(node.id)

        self.generic_visit(node)

    def _check_unreachable_statements(
        self,
        statements: list[ast.stmt],
    ) -> None:
        terminated = False

        for statement in statements:
            if terminated:
                self.diagnostic(
                    statement,
                    "WARNING",
                    "Unreachable code after unconditional control transfer.",
                )
                continue

            if isinstance(
                statement,
                (ast.Return, ast.Raise, ast.Break, ast.Continue),
            ):
                terminated = True

    def visit_Module(self, node: ast.Module) -> None:
        self._check_unreachable_statements(node.body)
        self.generic_visit(node)

    def _check_mutable_defaults(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        defaults = list(node.args.defaults)
        defaults.extend(
            default
            for default in node.args.kw_defaults
            if default is not None
        )

        for default in defaults:
            is_mutable_literal = isinstance(
                default,
                (ast.List, ast.Dict, ast.Set),
            )
            is_set_call = (
                isinstance(default, ast.Call)
                and isinstance(default.func, ast.Name)
                and default.func.id == "set"
                and not default.args
                and not default.keywords
            )

            if is_mutable_literal or is_set_call:
                self.diagnostic(
                    default,
                    "WARNING",
                    "Mutable default argument; use None and create "
                    "the mutable value inside the function.",
                )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name in self.callable_scopes[-1]:
            self.diagnostic(
                node,
                "WARNING",
                f"Name '{node.name}' is defined more than once.",
            )
        elif node.name in self.definition_scopes[-1]:
            self.diagnostic(
                node,
                "WARNING",
                f"Name '{node.name}' is rebound by a function definition.",
            )

        self.definition_scopes[-1].add(node.name)
        self.callable_scopes[-1].add(node.name)
        self._check_mutable_defaults(node)
        self._check_unreachable_statements(node.body)
        self.definition_scopes.append(set())
        self.import_scopes.append(set())
        self.callable_scopes.append(set())
        try:
            self.generic_visit(node)
        finally:
            self.callable_scopes.pop()
            self.import_scopes.pop()
            self.definition_scopes.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_For(self, node: ast.For) -> None:
        self._check_unreachable_statements(node.body)
        self._check_unreachable_statements(node.orelse)
        self.loop_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self.loop_depth -= 1

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_With(self, node: ast.With) -> None:
        self._check_unreachable_statements(node.body)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_If(self, node: ast.If) -> None:
        if isinstance(node.test, ast.Constant) and node.test.value is False:
            self.diagnostic(
                node,
                "WARNING",
                "Code is guarded by 'if False' and will never execute.",
            )

        self._check_unreachable_statements(node.body)
        self._check_unreachable_statements(node.orelse)
        self.generic_visit(node)

    def _loop_has_break(self, statements: list[ast.stmt]) -> bool:
        class BreakFinder(ast.NodeVisitor):
            def __init__(self) -> None:
                self.found = False

            def visit_Break(self, node: ast.Break) -> None:
                self.found = True

            def visit_For(self, node: ast.For) -> None:
                return

            def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
                return

            def visit_While(self, node: ast.While) -> None:
                return

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(
                self,
                node: ast.AsyncFunctionDef,
            ) -> None:
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                return

        finder = BreakFinder()

        for statement in statements:
            finder.visit(statement)
            if finder.found:
                return True

        return False

    def visit_While(self, node: ast.While) -> None:
        if (
            isinstance(node.test, ast.Constant)
            and node.test.value is True
            and not self._loop_has_break(node.body)
        ):
            self.diagnostic(
                node,
                "INFO",
                "Loop uses 'while True' with no obvious reachable exit.",
            )

        self._check_unreachable_statements(node.body)
        self._check_unreachable_statements(node.orelse)
        self.loop_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self.loop_depth -= 1

    def visit_Compare(self, node: ast.Compare) -> None:
        if self.loop_depth > 0:
            for operator, comparator in zip(node.ops, node.comparators):
                if not isinstance(operator, (ast.In, ast.NotIn)):
                    continue

                if not isinstance(comparator, (ast.List, ast.Tuple)):
                    continue

                if not comparator.elts:
                    continue

                if not all(
                    isinstance(element, ast.Constant)
                    for element in comparator.elts
                ):
                    continue

                self.diagnostic(
                    comparator,
                    "INFO",
                    "Constant list/tuple membership test inside a loop; "
                    "consider reusing a set or frozenset outside the loop.",
                )

        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        if isinstance(node.test, ast.Tuple) and node.test.elts:
            self.diagnostic(
                node,
                "WARNING",
                "Assert condition is a non-empty tuple and is always truthy.",
            )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"eval", "exec"}
        ):
            self.diagnostic(
                node,
                "WARNING",
                f"Use of {node.func.id}() can execute arbitrary code; "
                "avoid it unless the input is fully trusted.",
            )

        if (
            self.loop_depth > 0
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compile"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, (str, bytes))
        ):
            self.diagnostic(
                node,
                "INFO",
                "Constant re.compile() inside a loop; "
                "consider compiling the pattern once outside the loop.",
            )

        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.diagnostic(
                node,
                "WARNING",
                "Bare except catches BaseException; catch a specific "
                "exception type instead.",
            )

        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._check_unreachable_statements(node.body)
        self._check_unreachable_statements(node.orelse)
        self._check_unreachable_statements(node.finalbody)
        for handler in node.handlers:
            self._check_unreachable_statements(handler.body)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        for case in node.cases:
            self._check_unreachable_statements(case.body)
        self.generic_visit(node)


def _scope_summary(scope: Scope, indent: int = 0) -> list[str]:
    """Return a readable scope tree for debugging."""

    prefix = " " * indent
    lines = [
        f"{prefix}{scope.kind}: {scope.name}",
        f"{prefix}  defined: {sorted(scope.defined)}",
        f"{prefix}  used: {sorted(scope.used)}",
    ]

    if scope.globals:
        lines.append(f"{prefix}  global: {sorted(scope.globals)}")

    if scope.nonlocals:
        lines.append(f"{prefix}  nonlocal: {sorted(scope.nonlocals)}")

    for child in scope.children:
        lines.extend(_scope_summary(child, indent + 2))

    return lines


def analyze_file(path: Path) -> AnalysisResult:
    """Parse and analyze one Python file."""

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return AnalysisResult(diagnostics=[])

    analyzer = PythonAnalyzer(path)
    analyzer.visit(tree)

    # Build the scope tree and check name resolution.
    scope = analyze_scope(path)

    for name, line, column in find_undefined_names(scope):
        analyzer.diagnostics.append(
            Diagnostic(
                file=path,
                line=line,
                column=column,
                severity="ERROR",
                message=f"Undefined name: '{name}'",
            )
        )

    used_names: set[str] = set()

    def collect_used(current_scope) -> None:
        used_names.update(current_scope.used)

        for child_scope in current_scope.children:
            collect_used(child_scope)

    collect_used(scope)

    exported_names: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue

        if value is None:
            continue

        is_all_assignment = any(
            isinstance(target, ast.Name)
            and target.id == "__all__"
            for target in targets
        )

        if not is_all_assignment:
            continue

        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for element in value.elts:
                if (
                    isinstance(element, ast.Constant)
                    and isinstance(element.value, str)
                ):
                    exported_names.add(element.value)

    used_names.update(exported_names)

    for name, line in analyzer.imports.items():
        if name not in used_names:
            analyzer.diagnostics.append(
                Diagnostic(
                    file=path,
                    line=line,
                    column=1,
                    severity="WARNING",
                    message=f"Unused import: '{name}'",
                )
            )

    for name, line, column in find_unused_definitions(scope):
        if name not in analyzer.imports:
            analyzer.diagnostics.append(
                Diagnostic(
                    file=path,
                    line=line,
                    column=column,
                    severity="WARNING",
                    message=f"Unused definition: '{name}'",
                )
            )

    return AnalysisResult(
        diagnostics=analyzer.diagnostics,
    )
