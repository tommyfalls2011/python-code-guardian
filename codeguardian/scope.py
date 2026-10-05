from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Scope:
    """Represent one Python lexical scope."""

    name: str
    kind: str
    parent: Scope | None = None
    children: list[Scope] = field(default_factory=list)

    defined: set[str] = field(default_factory=set)
    used: set[str] = field(default_factory=set)
    usage_locations: dict[str, tuple[int, int]] = field(default_factory=dict)
    definition_locations: dict[str, tuple[int, int]] = field(default_factory=dict)
    globals: set[str] = field(default_factory=set)
    nonlocals: set[str] = field(default_factory=set)
    parameters: set[str] = field(default_factory=set)

    def add_child(self, child: Scope) -> None:
        self.children.append(child)


class ScopeAnalyzer(ast.NodeVisitor):
    """Build a lexical scope tree from Python AST."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.root = Scope(name="<module>", kind="module")
        self.current = self.root

    def push_scope(self, name: str, kind: str) -> Scope:
        scope = Scope(
            name=name,
            kind=kind,
            parent=self.current,
        )
        self.current.add_child(scope)
        self.current = scope
        return scope

    def pop_scope(self) -> None:
        if self.current.parent is not None:
            self.current = self.current.parent

    def define(
        self,
        name: str,
        node: ast.AST | None = None,
    ) -> None:
        target = self.current

        if name in self.current.globals:
            target = self.root
        elif name in self.current.nonlocals:
            parent = self.current.parent
            while parent is not None:
                if name in parent.defined:
                    target = parent
                    break
                parent = parent.parent

        target.defined.add(name)

        if node is not None:
            target.definition_locations.setdefault(
                name,
                (
                    getattr(node, "lineno", 1),
                    getattr(node, "col_offset", 0) + 1,
                ),
            )

    def use(self, name: str, node: ast.Name | None = None) -> None:
        self.current.used.add(name)

        if node is not None:
            self.current.usage_locations.setdefault(
                name,
                (node.lineno, node.col_offset + 1),
            )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.define(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "__future__":
            return

        for alias in node.names:
            if alias.name != "*":
                self.define(alias.asname or alias.name, node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.define(node.id, node)
        elif isinstance(node.ctx, ast.Load):
            self.use(node.id, node)

    @staticmethod
    def _dir_membership_guard_name(node: ast.AST) -> str | None:
        if not isinstance(node, ast.Compare):
            return None
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return None
        if not isinstance(node.ops[0], ast.In):
            return None
        if not (
            isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
        ):
            return None

        comparator = node.comparators[0]
        if not (
            isinstance(comparator, ast.Call)
            and isinstance(comparator.func, ast.Name)
            and comparator.func.id == "dir"
            and not comparator.args
            and not comparator.keywords
        ):
            return None

        return node.left.value

    def visit_IfExp(self, node: ast.IfExp) -> None:
        guarded_name = self._dir_membership_guard_name(node.test)
        self.visit(node.test)

        if guarded_name is None:
            self.visit(node.body)
        else:
            used_before = guarded_name in self.current.used
            location_before = self.current.usage_locations.get(guarded_name)

            self.visit(node.body)

            if not used_before:
                self.current.used.discard(guarded_name)
                self.current.usage_locations.pop(guarded_name, None)
            elif location_before is not None:
                self.current.usage_locations[guarded_name] = location_before

        self.visit(node.orelse)

    def visit_Global(self, node: ast.Global) -> None:
        self.current.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.current.nonlocals.update(node.names)

    def _define_arguments(self, node: ast.arguments) -> None:
        for argument in (
            *node.posonlyargs,
            *node.args,
            *node.kwonlyargs,
        ):
            self.define(argument.arg, argument)
            self.current.parameters.add(argument.arg)

        if node.vararg:
            self.define(node.vararg.arg, node.vararg)
            self.current.parameters.add(node.vararg.arg)

        if node.kwarg:
            self.define(node.kwarg.arg, node.kwarg)
            self.current.parameters.add(node.kwarg.arg)

    def _visit_function_defaults(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)

        for default in node.args.defaults:
            self.visit(default)

        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)

        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)

        if node.args.vararg and node.args.vararg.annotation is not None:
            self.visit(node.args.vararg.annotation)

        if node.args.kwarg and node.args.kwarg.annotation is not None:
            self.visit(node.args.kwarg.annotation)

        if node.returns:
            self.visit(node.returns)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.define(node.name, node)
        self._visit_function_defaults(node)

        self.push_scope(node.name, "function")
        self._define_arguments(node.args)

        for statement in node.body:
            self.visit(statement)

        self.pop_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.define(node.name, node)
        self._visit_function_defaults(node)

        self.push_scope(node.name, "function")
        self._define_arguments(node.args)

        for statement in node.body:
            self.visit(statement)

        self.pop_scope()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.push_scope("<lambda>", "lambda")
        self._define_arguments(node.args)
        self.visit(node.body)
        self.pop_scope()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.define(node.name, node)

        for decorator in node.decorator_list:
            self.visit(decorator)

        for base in node.bases:
            self.visit(base)

        for keyword in node.keywords:
            self.visit(keyword.value)

        self.push_scope(node.name, "class")

        for statement in node.body:
            self.visit(statement)

        self.pop_scope()

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.target)
        self.visit(node.iter)

        for statement in node.body:
            self.visit(statement)

        for statement in node.orelse:
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self.visit(item.optional_vars)

        for statement in node.body:
            self.visit(statement)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type:
            self.visit(node.type)

        if node.name:
            self.define(node.name)

        for statement in node.body:
            self.visit(statement)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.visit(node.iter)
        self.visit(node.target)

        for condition in node.ifs:
            self.visit(condition)

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
    ) -> None:
        self.push_scope("<comprehension>", "comprehension")

        for generator in node.generators:
            self.visit_comprehension(generator)

        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)

        self.pop_scope()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def _define_pattern(self, pattern: ast.pattern) -> None:
        if isinstance(pattern, ast.MatchValue):
            self.visit(pattern.value)

        elif isinstance(pattern, ast.MatchSingleton):
            return

        elif isinstance(pattern, ast.MatchSequence):
            for item in pattern.patterns:
                self._define_pattern(item)

        elif isinstance(pattern, ast.MatchMapping):
            for key in pattern.keys:
                self.visit(key)

            for item in pattern.patterns:
                self._define_pattern(item)

            if pattern.rest:
                self.define(pattern.rest)

        elif isinstance(pattern, ast.MatchClass):
            self.visit(pattern.cls)

            for pattern_node in pattern.patterns:
                self._define_pattern(pattern_node)

            for pattern_node in pattern.kwd_patterns:
                self._define_pattern(pattern_node)

        elif isinstance(pattern, ast.MatchStar):
            if pattern.name:
                self.define(pattern.name)

        elif isinstance(pattern, ast.MatchAs):
            if pattern.pattern:
                self._define_pattern(pattern.pattern)

            if pattern.name:
                self.define(pattern.name)

        elif isinstance(pattern, ast.MatchOr):
            for pattern_node in pattern.patterns:
                self._define_pattern(pattern_node)

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)

        for case in node.cases:
            self._define_pattern(case.pattern)

            if case.guard:
                self.visit(case.guard)

            for statement in case.body:
                self.visit(statement)

    def analyze(self, tree: ast.AST) -> Scope:
        self.visit(tree)
        return self.root


def analyze_scope(path: Path) -> Scope:
    """Parse a Python file and return its scope tree."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    analyzer = ScopeAnalyzer(path)
    return analyzer.analyze(tree)


BUILTIN_NAMES = set(dir(builtins))

# Names normally supplied by Python's module execution environment.
MODULE_RUNTIME_NAMES = {
    "__file__",
    "__name__",
    "__package__",
    "__loader__",
    "__spec__",
    "__cached__",
    "__builtins__",
}


def find_undefined_names(
    scope: Scope,
) -> list[tuple[str, int, int]]:
    """Find names used by a scope that cannot be resolved."""

    undefined: dict[str, tuple[int, int]] = {}

    def resolve(name: str, current: Scope) -> bool:
        if name in current.defined:
            return True

        if name in current.globals:
            return name in scope.root_defined

        if name in current.nonlocals:
            parent = current.parent
            while parent is not None:
                if name in parent.defined:
                    return True
                parent = parent.parent
            return False

        parent = current.parent
        while parent is not None:
            if name in parent.defined:
                return True
            parent = parent.parent

        return (
            name in BUILTIN_NAMES
            or name in MODULE_RUNTIME_NAMES
        )

    def walk(current: Scope) -> None:
        for name in current.used:
            if not resolve(name, current):
                location = current.usage_locations.get(name, (1, 1))
                undefined[name] = location

        for child in current.children:
            walk(child)

    scope.root_defined = scope.defined
    walk(scope)

    return [
        (name, line, column)
        for name, (line, column) in sorted(undefined.items())
    ]


def find_unused_definitions(
    scope: Scope,
) -> list[tuple[str, int, int]]:
    """Find locally defined names that are never used."""

    unused: list[tuple[str, int, int]] = []

    def used_in_scope_or_descendants(name: str, current: Scope) -> bool:
        if name in current.used:
            return True

        for child in current.children:
            if name in child.defined and name not in child.globals and name not in child.nonlocals:
                continue

            if used_in_scope_or_descendants(name, child):
                return True

        return False

    ignored = {
        "__class__",
        "__module__",
        "__qualname__",
        "__annotations__",
    }

    def walk(current: Scope) -> None:
        # Module-level definitions are public API candidates and may be
        # referenced by imports from other modules, so do not flag them here.
        if current.kind in {"function", "lambda"}:
            for name in current.defined:
                if name in ignored:
                    continue

                if name.startswith("_"):
                    continue

                if name in current.parameters:
                    continue

                if not used_in_scope_or_descendants(name, current):
                    location = current.definition_locations.get(name, (1, 1))
                    unused.append((name, location[0], location[1]))

        for child in current.children:
            walk(child)

    walk(scope)

    return sorted(unused, key=lambda item: (item[1], item[2], item[0]))
