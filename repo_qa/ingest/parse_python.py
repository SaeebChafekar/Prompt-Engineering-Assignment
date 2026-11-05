"""Parse Python source into structured facts using the standard `ast` module.

This is the raw material for the knowledge graph: which classes and functions exist,
where they are, what they import, and what they call. Nothing is resolved here. Call
targets are kept as the names written in the source ("self.save", "jwt.decode"), and
the graph builder later decides which definition each one most likely refers to.

Parsing never raises on bad input. A file that cannot be parsed (syntax error, Python 2
code, template placeholders, extreme nesting) comes back as a ParsedFile with `error`
set, and the chunker falls back to plain line windows for it.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import PurePosixPath
from typing import Iterator

from .models import ClassInfo, FunctionInfo, ImportInfo, ParsedFile

MAX_DOCSTRING_CHARS = 500
_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_TRY_TYPES = tuple(t for t in (ast.Try, getattr(ast, "TryStar", None)) if t is not None)
_MATCH_TYPE = getattr(ast, "Match", None)


def module_name_from_path(path: str) -> str:
    """'src/auth/jwt.py' -> 'src.auth.jwt'; 'pkg/__init__.py' -> 'pkg'."""
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def parse_python(path: str, source: str) -> ParsedFile:
    """Parse one file. Never raises; failures are reported in ParsedFile.error."""
    module = module_name_from_path(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # e.g. invalid escape sequences in old code
            tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return ParsedFile(path=path, module=module, error=f"SyntaxError at line {exc.lineno}: {exc.msg}")
    except (ValueError, RecursionError, MemoryError, OverflowError) as exc:
        return ParsedFile(path=path, module=module, error=f"{type(exc).__name__}: {str(exc)[:100]}")

    collector = _Collector(path)
    try:
        collector.visit_scope(tree.body, scope="", parent_class=None, in_function=False)
        module_calls = _unique(
            name for stmt in _flatten(tree.body) if not _is_definition(stmt) for name in _calls_in([stmt])
        )
    except RecursionError:
        return ParsedFile(path=path, module=module, error="RecursionError: code is nested too deeply")

    return ParsedFile(
        path=path,
        module=module,
        docstring=_docstring(tree),
        classes=tuple(collector.classes),
        functions=tuple(collector.functions),
        imports=tuple(collector.imports),
        module_calls=tuple(module_calls),
        assignments=tuple(_module_assignments(tree.body)),
    )


class _Collector:
    """Walks scopes (module, class, function bodies) and records definitions."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.classes: list[ClassInfo] = []
        self.functions: list[FunctionInfo] = []
        self.imports: list[ImportInfo] = []

    def visit_scope(self, body: list[ast.stmt], *, scope: str, parent_class: str | None, in_function: bool) -> None:
        for stmt in _flatten(body):
            if isinstance(stmt, _FUNC_TYPES):
                self._function(stmt, scope, parent_class, in_function)
            elif isinstance(stmt, ast.ClassDef):
                self._class(stmt, scope, in_function)
            elif isinstance(stmt, ast.Import):
                self._import(stmt, scope)
            elif isinstance(stmt, ast.ImportFrom):
                self._import_from(stmt, scope)

    # ---- definitions
    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, scope: str, parent_class: str | None, in_function: bool) -> None:
        qualname = f"{scope}.{node.name}" if scope else node.name
        self.functions.append(
            FunctionInfo(
                name=node.name,
                qualname=qualname,
                path=self.path,
                start_line=_start_line(node),
                def_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=_signature(node),
                docstring=_docstring(node),
                decorators=tuple(_unparse(d) for d in node.decorator_list),
                calls=tuple(_unique(_calls_in(node.body))),
                parent_class=parent_class,
                is_nested=in_function,
            )
        )
        # Nested functions and lazy imports inside the body belong to this function.
        self.visit_scope(node.body, scope=qualname, parent_class=None, in_function=True)

    def _class(self, node: ast.ClassDef, scope: str, in_function: bool) -> None:
        qualname = f"{scope}.{node.name}" if scope else node.name
        methods = tuple(
            f"{qualname}.{s.name}" for s in _flatten(node.body) if isinstance(s, _FUNC_TYPES)
        )
        self.classes.append(
            ClassInfo(
                name=node.name,
                qualname=qualname,
                path=self.path,
                start_line=_start_line(node),
                def_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                bases=tuple(_unparse(b) for b in node.bases),
                docstring=_docstring(node),
                decorators=tuple(_unparse(d) for d in node.decorator_list),
                methods=methods,
                in_function=in_function,
            )
        )
        self.visit_scope(node.body, scope=qualname, parent_class=qualname, in_function=in_function)

    # ---- imports
    def _import(self, node: ast.Import, scope: str) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            target = alias.name if alias.asname else alias.name.split(".")[0]
            self.imports.append(
                ImportInfo(module=alias.name, level=0, bindings=((local, target),), line=node.lineno, scope=scope)
            )

    def _import_from(self, node: ast.ImportFrom, scope: str) -> None:
        module = node.module or ""
        prefix = "." * node.level + module
        bindings = []
        for alias in node.names:
            if alias.name == "*":
                bindings.append(("*", f"{prefix}.*"))
            else:
                joiner = "" if prefix.endswith(".") or not prefix else "."
                bindings.append((alias.asname or alias.name, f"{prefix}{joiner}{alias.name}"))
        self.imports.append(
            ImportInfo(module=module, level=node.level, bindings=tuple(bindings), line=node.lineno, scope=scope)
        )


# ---------------------------------------------------------------- helpers
def _flatten(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Yield statements, looking inside if/try/with/for/while/match blocks.

    Definitions and imports are often wrapped in `if TYPE_CHECKING:` or
    `try: import x except ImportError:`; flattening finds them without treating the
    wrapper as a scope. Function and class bodies are NOT entered here.
    """
    for stmt in body:
        if isinstance(stmt, ast.If):
            yield from _flatten(stmt.body)
            yield from _flatten(stmt.orelse)
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            yield from _flatten(stmt.body)
            yield from _flatten(stmt.orelse)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            yield from _flatten(stmt.body)
        elif _TRY_TYPES and isinstance(stmt, _TRY_TYPES):
            yield from _flatten(stmt.body)
            for handler in stmt.handlers:
                yield from _flatten(handler.body)
            yield from _flatten(stmt.orelse)
            yield from _flatten(stmt.finalbody)
        elif _MATCH_TYPE is not None and isinstance(stmt, _MATCH_TYPE):
            for case in stmt.cases:
                yield from _flatten(case.body)
        else:
            yield stmt


def _module_assignments(body: list[ast.stmt]) -> list[tuple[str, str]]:
    """Module-level `name = Constructor(...)` assignments, as (name, constructor name)."""
    found: list[tuple[str, str]] = []
    for stmt in _flatten(body):
        if isinstance(stmt, ast.Assign):
            targets = [t for t in stmt.targets if isinstance(t, ast.Name)]
            value: ast.expr | None = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            targets, value = [stmt.target], stmt.value
        else:
            continue
        if isinstance(value, ast.Call):
            constructor = _call_name(value.func)
            if constructor and not constructor.startswith("?"):
                found.extend((t.id, constructor) for t in targets)
    return found


def _is_definition(stmt: ast.stmt) -> bool:
    return isinstance(stmt, (*_FUNC_TYPES, ast.ClassDef, ast.Import, ast.ImportFrom))


def _calls_in(body: list[ast.stmt]) -> list[str]:
    """Names of everything called in these statements, in source order.

    Nested function and class bodies are skipped: they are separate scopes with their
    own entries. Lambdas are included because they run in the enclosing scope.
    """
    found: list[tuple[int, int, str]] = []
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (*_FUNC_TYPES, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                found.append((node.lineno, node.col_offset, name))
        stack.extend(ast.iter_child_nodes(node))
    found.sort()
    return [name for _, _, name in found]


def _call_name(func: ast.expr) -> str | None:
    """Render what is being called: 'foo', 'self.save', 'jwt.decode', '?.strip'."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        node: ast.expr = func.value
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        base = node.id if isinstance(node, ast.Name) else "?"  # "?" = result of an expression
        return ".".join([base, *reversed(parts)])
    return None  # calls on subscripts, lambdas, etc. cannot be named


def _unique(names: "Iterator[str] | list[str]") -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _start_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    return min([node.lineno, *(d.lineno for d in node.decorator_list)])


def _docstring(node: ast.AST) -> str:
    try:
        doc = ast.get_docstring(node) or ""  # type: ignore[arg-type]
    except TypeError:
        return ""
    doc = doc.strip()
    return doc[:MAX_DOCSTRING_CHARS] + ("..." if len(doc) > MAX_DOCSTRING_CHARS else "")


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # unparse can fail on unusual nodes; a placeholder is fine
        return "?"


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    returns = f" -> {_unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix}{node.name}({_unparse(node.args)}){returns}"
