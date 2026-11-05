"""Data model shared by all ingestion stages.

Everything here is a plain, frozen dataclass so results are easy to test, print and
serialize. Paths are always repo-relative with forward slashes ("src/auth/jwt.py"),
on every operating system, so they can be used directly as citations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class SourceFile:
    """One file that passed the filters, with its text loaded."""

    path: str
    language: str  # "python", "markdown", "javascript", ...
    kind: str  # "code" | "doc" | "config"
    text: str
    size: int  # bytes on disk
    is_test: bool = False

    @property
    def line_count(self) -> int:
        return self.text.count("\n") + (0 if self.text.endswith("\n") or not self.text else 1)


# ---------------------------------------------------------------- Python parsing
@dataclass(frozen=True)
class ImportInfo:
    """One import statement. `bindings` maps local names to what they refer to.

    import a.b          -> module "a.b", bindings (("a", "a"),)
    import a.b as c     -> module "a.b", bindings (("c", "a.b"),)
    from m import x as y-> module "m",   bindings (("y", "m.x"),)
    from . import x     -> module "",    level 1, bindings (("x", ".x"),)
    from m import *     -> module "m",   bindings (("*", "m.*"),)

    Relative targets keep their leading dots; the graph builder resolves them against
    the importing file's package.
    """

    module: str
    level: int  # number of leading dots for relative imports, else 0
    bindings: tuple[tuple[str, str], ...]
    line: int
    scope: str = ""  # qualified name of the enclosing function/class, "" for module level


@dataclass(frozen=True)
class FunctionInfo:
    name: str
    qualname: str  # "Class.method", or "outer.inner" for nested functions
    path: str
    start_line: int  # includes decorators
    def_line: int  # the line holding "def"
    end_line: int
    signature: str  # e.g. "async verify_token(self, token: str) -> bool"
    docstring: str
    decorators: tuple[str, ...]
    calls: tuple[str, ...]  # raw call targets in source order, e.g. "self.save", "jwt.decode"
    parent_class: str | None = None  # qualname of the enclosing class if this is a method
    is_nested: bool = False  # defined inside another function


@dataclass(frozen=True)
class ClassInfo:
    name: str
    qualname: str
    path: str
    start_line: int  # includes decorators
    def_line: int
    end_line: int
    bases: tuple[str, ...]
    docstring: str
    decorators: tuple[str, ...]
    methods: tuple[str, ...]  # qualnames of directly defined methods
    in_function: bool = False  # defined inside a function body


@dataclass(frozen=True)
class ParsedFile:
    path: str
    module: str  # dotted path from the repo root, e.g. "src.auth.jwt"
    docstring: str = ""
    classes: tuple[ClassInfo, ...] = ()
    functions: tuple[FunctionInfo, ...] = ()
    imports: tuple[ImportInfo, ...] = ()
    module_calls: tuple[str, ...] = ()  # calls made at import time (scripts, __main__)
    # Module-level `name = Constructor(...)` assignments as (name, "Constructor"), e.g.
    # ("db", "SQLAlchemy"). Lets the graph know what `db.session.add(...)` is called on.
    assignments: tuple[tuple[str, str], ...] = ()
    error: str | None = None  # set when the file could not be parsed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- chunks
@dataclass(frozen=True)
class Chunk:
    """A piece of the repository that gets embedded and retrieved."""

    id: str
    kind: str  # "function" | "class" | "module" | "doc" | "block" | "config"
    path: str
    start_line: int
    end_line: int
    name: str  # qualified symbol name, section title, or file name
    language: str
    text: str  # the content shown to the answering model
    context: str  # short header describing where the text comes from
    symbol: str | None = None  # qualname of the function/class this chunk represents
    is_test: bool = False
    part: int = 1  # large chunks are split into parts
    parts: int = 1

    @property
    def embed_text(self) -> str:
        """Text to embed: the header carries path and symbol words into the vector."""
        return f"{self.context}\n{self.text}"

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


# ---------------------------------------------------------------- result
@dataclass
class IngestResult:
    repo_name: str
    root: str  # absolute path of the repository on disk
    source: str  # what the user gave us (URL or folder)
    files: list[SourceFile] = field(default_factory=list)
    parsed: list[ParsedFile] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)  # reason -> count
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def files_by_language(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.files:
            counts[f.language] = counts.get(f.language, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def summary(self) -> str:
        funcs = sum(len(p.functions) for p in self.parsed)
        classes = sum(len(p.classes) for p in self.parsed)
        imports = sum(len(p.imports) for p in self.parsed)
        kinds: dict[str, int] = {}
        for c in self.chunks:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        lines = [
            f"Repository : {self.repo_name}",
            f"Files kept : {len(self.files)} ("
            + ", ".join(f"{n} {lang}" for lang, n in self.files_by_language().items())
            + ")",
            f"Python     : {len(self.parsed)} parsed, {funcs} functions/methods, "
            f"{classes} classes, {imports} imports",
            f"Chunks     : {len(self.chunks)} ("
            + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
            + ")",
        ]
        if self.skipped:
            lines.append(
                "Skipped    : "
                + ", ".join(f"{n} {why}" for why, n in sorted(self.skipped.items()))
            )
        lines.append(f"Time       : {self.elapsed_s:.1f}s")
        for w in self.warnings:
            lines.append(f"Warning    : {w}")
        return "\n".join(lines)
