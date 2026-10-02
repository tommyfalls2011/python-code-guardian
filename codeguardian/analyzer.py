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
        self.defined_names: set[str] = set()

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

            if name in self.imports:
                self.diagnostic(
                    node,
                    "WARNING",
                    f"Duplicate import: '{name}'",
                )
            else:
                self.imports[name] = node.lineno

            self.defined_names.add(name)

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.diagnostic(
                    node,
                    "WARNING",
                    "Wildcard import makes static analysis less reliable.",
                )
                continue

            name = alias.asname or alias.name

            if name in self.imports:
                self.diagnostic(
                    node,
                    "WARNING",
                    f"Duplicate import: '{name}'",
                )
            else:
                self.imports[name] = node.lineno

            self.defined_names.add(name)

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name in self.defined_names:
            self.diagnostic(
                node,
                "WARNING",
                f"Name '{node.name}' is defined more than once.",
            )

        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name in self.defined_names:
            self.diagnostic(
                node,
                "WARNING",
                f"Name '{node.name}' is defined more than once.",
            )

        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.defined_names.add(node.id)

        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if isinstance(node.test, ast.Constant) and node.test.value is False:
            self.diagnostic(
                node,
                "WARNING",
                "Code is guarded by 'if False' and will never execute.",
            )

        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            self.diagnostic(
                node,
                "INFO",
                "Loop uses 'while True'; verify that it has a reachable exit.",
            )

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
