"""Build the knowledge graph from an ingestion result.

    ingest_repo(...) -> IngestResult  --build_graph-->  KnowledgeGraph

Steps: (1) File nodes, (2) Class and Function nodes with CONTAINS edges, (3) DocSection
nodes, (4) IMPORTS, (5) INHERITS, (6) DECORATED_BY, (7) CALLS and USES, (8) DOCUMENTS.

Nested functions are folded into their top-level parent (a call made inside a nested
helper counts as a call made by the parent), matching how the chunker keeps them inside
the parent's chunk. Every graph node records the ids of the retrieval chunks it owns, so
a vector-search hit can be mapped to its place in the graph.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from repo_qa.ingest.models import Chunk, FunctionInfo, IngestResult, ParsedFile

from .model import KnowledgeGraph
from .resolve import STDLIB_NAMES, CallContext, CallTarget, RepoIndex, decorator_reference
from .schema import (
    CERTAIN,
    EDGE_CALLS,
    EDGE_CONTAINS,
    EDGE_DECORATED_BY,
    EDGE_DOCUMENTS,
    EDGE_IMPORTS,
    EDGE_INHERITS,
    EDGE_USES,
    GUESS,
    LIKELY,
    NODE_CLASS,
    NODE_DOC,
    NODE_EXTERNAL,
    NODE_FILE,
    NODE_FUNCTION,
    class_id,
    doc_id,
    external_id,
    file_id,
    function_id,
)

log = logging.getLogger(__name__)

MAX_DOC_LINKS_PER_SECTION = 30
MIN_DOC_NAME_LENGTH = 3
_BACKTICK_RE = re.compile(r"`([^`\n]{2,100})`")
_PATH_RE = re.compile(r"[\w./-]+\.py\b")
_IDENT_RE = re.compile(r"^[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*$")
_DOC_STOPWORDS = frozenset(
    "python pip git true false none null main self cls args kwargs init test tests app name data".split()
)


@dataclass
class CallStats:
    """How the call mentions found by the parser were resolved."""

    total: int = 0
    certain: int = 0
    likely: int = 0
    guess: int = 0
    external: int = 0  # calls into libraries (USES edges)
    builtin: int = 0  # len(), print(), ...
    skipped: int = 0  # str/list methods, dunders, nested helpers (deliberately ignored)
    unresolved: int = 0  # a plain name we could not find
    ambiguous: int = 0  # a method name shared by too many classes
    foreign: int = 0  # a method name no repository class defines (probably a library call)

    @property
    def resolved(self) -> int:
        return self.certain + self.likely + self.guess


@dataclass
class BuildReport:
    calls: CallStats = field(default_factory=CallStats)
    imports_internal: int = 0
    imports_external: int = 0
    imports_unresolved: int = 0
    doc_links: int = 0
    # Rows for hand-labelling call accuracy: (caller, raw call, target, confidence, via)
    call_rows: list[tuple[str, str, str, str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("call_rows", None)
        data["calls"]["resolved"] = self.calls.resolved
        return data


def build_graph(result: IngestResult, *, with_doc_links: bool = True) -> tuple[KnowledgeGraph, BuildReport]:
    """Build the graph and a report of how the resolution went."""
    graph = KnowledgeGraph(meta={"repo": result.repo_name, "source": result.source})
    report = BuildReport()
    chunk_ids = _chunk_ids_by_owner(result.chunks)
    parsed = {p.path: p for p in result.parsed}
    python_paths = [f.path for f in result.files if f.language == "python"]

    _add_files(graph, result, chunk_ids)
    nodes = _add_definitions(graph, parsed, chunk_ids)
    _add_doc_sections(graph, result)

    index = RepoIndex(python_paths, list(parsed.values()))
    _add_imports(graph, index, parsed, report)
    _add_inheritance(graph, index, parsed)
    _add_decorators(graph, index, parsed)
    _add_calls(graph, index, parsed, nodes, report)
    if with_doc_links:
        _add_doc_links(graph, result, report)

    graph.meta["counts"] = graph.counts()
    graph.meta["report"] = report.to_dict()
    return graph, report


# ---------------------------------------------------------------- nodes
def _chunk_ids_by_owner(chunks: Iterable[Chunk]) -> dict[tuple[str, str, str], list[str]]:
    """(path, kind, owner) -> chunk ids. owner is the symbol, section name or file path."""
    table: dict[tuple[str, str, str], list[str]] = {}
    for chunk in chunks:
        owner = chunk.symbol or chunk.name
        key_kind = "file" if chunk.kind in ("module", "block", "config") else chunk.kind
        if key_kind == "file":
            owner = ""
        table.setdefault((chunk.path, key_kind, owner), []).append(chunk.id)
    return table


def _add_files(graph: KnowledgeGraph, result: IngestResult, chunk_ids: dict) -> None:
    for sf in result.files:
        graph.add_node(
            file_id(sf.path), NODE_FILE, path=sf.path, name=sf.path.rsplit("/", 1)[-1],
            language=sf.language, file_kind=sf.kind, is_test=sf.is_test, lines=sf.line_count,
            chunk_ids=chunk_ids.get((sf.path, "file", ""), []),
        )


@dataclass
class _Nodes:
    functions: set[str] = field(default_factory=set)
    classes: set[str] = field(default_factory=set)


def _add_definitions(graph: KnowledgeGraph, parsed: dict[str, ParsedFile], chunk_ids: dict) -> _Nodes:
    nodes = _Nodes()
    for path, pf in parsed.items():
        if pf.error is not None:
            continue
        file_node = file_id(path)
        class_qualnames = {c.qualname for c in pf.classes if not c.in_function}
        for cls in pf.classes:
            if cls.in_function:
                continue
            cid = class_id(path, cls.qualname)
            graph.add_node(
                cid, NODE_CLASS, name=cls.name, qualname=cls.qualname, path=path,
                start_line=cls.start_line, end_line=cls.end_line, bases=list(cls.bases),
                docstring=cls.docstring, decorators=list(cls.decorators),
                is_test=graph.node(file_node)["is_test"],
                chunk_ids=chunk_ids.get((path, "class", cls.qualname), []),
            )
            nodes.classes.add(cid)
            outer = cls.qualname.rsplit(".", 1)[0] if "." in cls.qualname else None
            parent = class_id(path, outer) if outer in class_qualnames else file_node
            graph.add_edge(parent, cid, EDGE_CONTAINS)
        for fn in pf.functions:
            if fn.is_nested:
                continue
            fid = function_id(path, fn.qualname)
            graph.add_node(
                fid, NODE_FUNCTION, name=fn.name, qualname=fn.qualname, path=path,
                start_line=fn.start_line, end_line=fn.end_line, signature=fn.signature,
                docstring=fn.docstring, decorators=list(fn.decorators),
                is_method=fn.parent_class is not None, is_test=graph.node(file_node)["is_test"],
                chunk_ids=chunk_ids.get((path, "function", fn.qualname), []),
            )
            nodes.functions.add(fid)
            owner = class_id(path, fn.parent_class) if fn.parent_class in class_qualnames else file_node
            graph.add_edge(owner, fid, EDGE_CONTAINS)
    return nodes


def _add_doc_sections(graph: KnowledgeGraph, result: IngestResult) -> None:
    spans: dict[tuple[str, str], list[Chunk]] = {}
    for chunk in result.chunks:
        if chunk.kind == "doc" and chunk.language == "markdown":
            spans.setdefault((chunk.path, chunk.name), []).append(chunk)
    for (path, title), chunks in sorted(spans.items(), key=lambda kv: (kv[0][0], kv[1][0].start_line)):
        start = min(c.start_line for c in chunks)
        end = max(c.end_line for c in chunks)
        node = doc_id(path, start)
        graph.add_node(
            node, NODE_DOC, title=title, name=title, path=path, start_line=start, end_line=end,
            chunk_ids=[c.id for c in chunks],
        )
        graph.add_edge(file_id(path), node, EDGE_CONTAINS)


# ---------------------------------------------------------------- imports, inheritance, decorators
def _ensure_external(graph: KnowledgeGraph, name: str) -> str:
    node = external_id(name)
    if node not in graph:
        graph.add_node(node, NODE_EXTERNAL, name=name, stdlib=name in STDLIB_NAMES)
    return node


def _add_imports(graph: KnowledgeGraph, index: RepoIndex, parsed: dict[str, ParsedFile], report: BuildReport) -> None:
    for path, pf in parsed.items():
        if pf.error is not None:
            continue
        src = file_id(path)
        for imp in pf.imports:
            for local, target in imp.bindings:
                clean = target[:-2] if local == "*" and target.endswith(".*") else target
                binding = index.resolve_target(path, clean)
                label = local if local != "*" else "*"
                if binding.kind == "external":
                    if binding.external:
                        graph.add_edge(src, _ensure_external(graph, binding.external), EDGE_IMPORTS,
                                       via=f"line {imp.line}", names=[label])
                        report.imports_external += 1
                elif binding.kind == "unknown":
                    report.imports_unresolved += 1
                else:
                    dest = binding.module_path or binding.path
                    if dest and dest != path and file_id(dest) in graph:
                        graph.add_edge(src, file_id(dest), EDGE_IMPORTS, via=f"line {imp.line}", names=[label])
                        report.imports_internal += 1


def _add_inheritance(graph: KnowledgeGraph, index: RepoIndex, parsed: dict[str, ParsedFile]) -> None:
    for (path, qualname), cls in index.classes.items():
        src = class_id(path, qualname)
        if src not in graph:
            continue
        for base in cls.bases:
            ref = index.resolve_ref(path, base)
            if ref is None:
                continue
            if ref.kind == "symbol" and ref.symbol_kind == "class":
                dst = class_id(ref.path, ref.symbol)
                if dst in graph and dst != src:
                    graph.add_edge(src, dst, EDGE_INHERITS, names=[base])
            elif ref.kind == "external" and ref.external:
                inferred = ref.symbol_kind == "instance"  # e.g. db.Model, where db = SQLAlchemy()
                graph.add_edge(src, _ensure_external(graph, ref.external), EDGE_INHERITS,
                               confidence=LIKELY if inferred else CERTAIN, via="instance" if inferred else "import",
                               names=[base])


def _add_decorators(graph: KnowledgeGraph, index: RepoIndex, parsed: dict[str, ParsedFile]) -> None:
    def link(src: str, path: str, decorator: str) -> None:
        reference = decorator_reference(decorator)
        if reference is None:
            return
        ctx = CallContext(caller_class=None, caller_name="")
        for target in index.resolve_call(path, ctx, reference, allow_fallback=False):
            if target.kind == "func":
                dst = function_id(target.path, target.qualname)
            elif target.kind == "class":
                dst = class_id(target.path, target.qualname)
            elif target.kind == "external":
                dst = _ensure_external(graph, target.external)
            else:
                continue
            if dst in graph:
                graph.add_edge(src, dst, EDGE_DECORATED_BY, confidence=target.confidence or CERTAIN, names=[decorator])

    for path, pf in parsed.items():
        if pf.error is not None:
            continue
        for fn in pf.functions:
            if not fn.is_nested:
                for deco in fn.decorators:
                    link(function_id(path, fn.qualname), path, deco)
        for cls in pf.classes:
            if not cls.in_function:
                for deco in cls.decorators:
                    link(class_id(path, cls.qualname), path, deco)


# ---------------------------------------------------------------- calls
def _fold_nested(pf: ParsedFile) -> dict[str, list[FunctionInfo]]:
    """Group every function under its top-level owner: {owner qualname: [owner, nested...]}."""
    top = {fn.qualname: fn for fn in pf.functions if not fn.is_nested}
    groups: dict[str, list[FunctionInfo]] = {q: [fn] for q, fn in top.items()}
    for fn in pf.functions:
        if not fn.is_nested:
            continue
        pieces = fn.qualname.split(".")
        for cut in range(len(pieces) - 1, 0, -1):
            owner = ".".join(pieces[:cut])
            if owner in top:
                groups[owner].append(fn)
                break
    return groups


def _add_calls(graph: KnowledgeGraph, index: RepoIndex, parsed: dict[str, ParsedFile], nodes: _Nodes, report: BuildReport) -> None:
    for path, pf in parsed.items():
        if pf.error is not None:
            continue
        for owner_qualname, group in _fold_nested(pf).items():
            owner = group[0]
            local_names = frozenset(
                [f.name for f in group[1:]]
                + [c.name for c in pf.classes if c.in_function and c.qualname.startswith(owner_qualname + ".")]
            )
            ctx = CallContext(owner.parent_class, owner.name, local_names)
            seen: set[str] = set()
            raw_calls = [c for fn in group for c in fn.calls if not (c in seen or seen.add(c))]
            caller = function_id(path, owner_qualname)
            _resolve_calls(graph, index, path, ctx, caller, raw_calls, report)
        _resolve_calls(
            graph, index, path, CallContext(None, "<module>"), file_id(path), list(pf.module_calls), report
        )


def _resolve_calls(
    graph: KnowledgeGraph, index: RepoIndex, path: str, ctx: CallContext, caller: str,
    raw_calls: list[str], report: BuildReport,
) -> None:
    stats = report.calls
    for raw in raw_calls:
        stats.total += 1
        targets = index.resolve_call(path, ctx, raw)
        linked: list[CallTarget] = []
        for target in targets:
            dst = _target_node(graph, target)
            if target.kind in ("func", "class") and dst in graph:
                graph.add_edge(caller, dst, EDGE_CALLS, confidence=target.confidence, via=target.via, names=[raw])
                linked.append(target)
            elif target.kind == "external":
                graph.add_edge(caller, dst, EDGE_USES, confidence=target.confidence, via=target.via, names=[raw])
                linked.append(target)
        if linked:
            best = max(linked, key=lambda t: {GUESS: 1, LIKELY: 2, CERTAIN: 3}[t.confidence])
            if best.kind == "external":
                stats.external += 1
            else:
                setattr(stats, best.confidence, getattr(stats, best.confidence) + 1)
            for t in linked:
                report.call_rows.append((caller, raw, _target_node(graph, t), t.confidence, t.via))
        else:
            kind = targets[0].kind if targets else "unresolved"
            if kind in ("func", "class"):  # resolved on paper, but the node does not exist
                kind = "unresolved"
            setattr(stats, kind, getattr(stats, kind) + 1)
            report.call_rows.append((caller, raw, "", kind, targets[0].via if targets else ""))


def _target_node(graph: KnowledgeGraph, target: CallTarget) -> str:
    if target.kind == "func":
        return function_id(target.path, target.qualname)
    if target.kind == "class":
        return class_id(target.path, target.qualname)
    if target.kind == "external":
        return _ensure_external(graph, target.external)
    return ""


# ---------------------------------------------------------------- docs
def _add_doc_links(graph: KnowledgeGraph, result: IngestResult, report: BuildReport) -> None:
    """Link README/doc sections to the files, classes and functions they mention."""
    texts = {f.path: f.text.splitlines() for f in result.files if f.language == "markdown"}
    paths = [f.path for f in result.files]
    by_name: dict[str, list[str]] = {}
    for node_id, data in graph.nodes():
        if data["type"] in (NODE_FUNCTION, NODE_CLASS):
            for key in {data["name"], data["qualname"]}:
                by_name.setdefault(key, []).append(node_id)

    for doc_node, data in list(graph.nodes(NODE_DOC)):
        lines = texts.get(data["path"], [])[data["start_line"] - 1 : data["end_line"]]
        body = "\n".join(lines)
        found: dict[str, str] = {}  # node id -> confidence
        for token in _PATH_RE.findall(body):
            matches = [p for p in paths if p == token or p.endswith("/" + token)]
            if len(matches) == 1:
                found.setdefault(file_id(matches[0]), CERTAIN if matches[0] == token else LIKELY)
        for span in _BACKTICK_RE.findall(body):
            name = span.strip().rstrip(")").split("(")[0].removeprefix("self.").strip()
            if not _IDENT_RE.match(name) or len(name) < MIN_DOC_NAME_LENGTH or name.lower() in _DOC_STOPWORDS:
                continue
            candidates = by_name.get(name, [])
            if 1 <= len(candidates) <= 3:
                for cid in candidates:
                    found.setdefault(cid, LIKELY if len(candidates) == 1 else GUESS)
        for node_id, confidence in list(found.items())[:MAX_DOC_LINKS_PER_SECTION]:
            graph.add_edge(doc_node, node_id, EDGE_DOCUMENTS, confidence=confidence, via="mention")
            report.doc_links += 1


# ---------------------------------------------------------------- output helpers
def write_call_labeling_csv(report: BuildReport, path: str | Path) -> None:
    """Write every call decision to a CSV so accuracy can be hand-labelled for the report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["caller", "call_as_written", "resolved_to", "confidence_or_outcome", "via", "correct (y/n)"])
        for row in report.call_rows:
            writer.writerow([*row, ""])
