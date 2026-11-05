"""Retrieval: find the code that answers a question, then follow the knowledge graph.

    question --hybrid search--> seed chunks --graph expansion--> related chunks + facts
                                                   --token budget--> context for the model

Plain search finds the function that matches the question. The graph adds what surrounds
it: who calls it, what it calls, its decorators and base classes, and the documentation that
mentions it. For "where is authentication handled?" that means the login view AND the
password check it calls AND the decorator that protects the endpoints, even when those
chunks do not mention the question's words.

Three things keep this from drowning the model in context:
- neighbours are scored (seed score x confidence of the edges x decay per hop) and rank
  below the direct hits;
- everything is cut to a token budget, direct hits first;
- relationships are also summarized as short "facts" lines, which cost few tokens and
  answer "what calls X?" directly. Edges that are only guesses are labelled as such.

`use_graph=False` gives plain search, which is what the evaluation compares against.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

from common.config import Config
from repo_qa.graph.model import KnowledgeGraph, Neighbor
from repo_qa.graph.schema import (
    EDGE_CALLS,
    EDGE_DECORATED_BY,
    EDGE_DOCUMENTS,
    EDGE_IMPORTS,
    EDGE_INHERITS,
    EDGE_USES,
    GUESS,
    NODE_CLASS,
    NODE_DOC,
    NODE_FUNCTION,
)
from repo_qa.index.index import RepoIndex
from repo_qa.index.search import HybridSearcher
from repo_qa.ingest.filters import CODE_EXTENSIONS, CONFIG_EXTENSIONS, DOC_EXTENSIONS
from repo_qa.ingest.models import Chunk

CHARS_PER_TOKEN = 3.5  # rough, deliberately a bit pessimistic for code
ITEM_OVERHEAD_TOKENS = 30  # the header printed above each chunk
EXPAND_EDGES = (EDGE_CALLS, EDGE_DECORATED_BY, EDGE_INHERITS, EDGE_DOCUMENTS)
GRAPH_SCORE_SCALE = 0.8  # a neighbour ranks below a seed of equal strength
GRAPH_DECAY = 0.6  # extra multiplier per hop beyond the first
MAX_FACT_NODES = 8
MAX_FACT_TOKENS = 600
TRUNCATION_NOTE = "\n... [truncated to fit the context budget]"

_RELATION_TEXT = {
    (EDGE_CALLS, True): "called by",
    (EDGE_CALLS, False): "calls",
    (EDGE_DECORATED_BY, True): "decorator used by",
    (EDGE_DECORATED_BY, False): "decorated by",
    (EDGE_INHERITS, True): "base class of",
    (EDGE_INHERITS, False): "subclass of",
    (EDGE_DOCUMENTS, True): "mentioned in docs section",
    (EDGE_DOCUMENTS, False): "documents",
}


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass(frozen=True)
class ContextItem:
    chunk: Chunk
    score: float
    origin: str  # "search" (a direct hit) or "graph" (reached through the graph)
    reason: str  # why it is here, e.g. "called by app/auth.py::login (certain)"


@dataclass
class Retrieval:
    query: str
    items: list[ContextItem] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    tokens: int = 0  # estimated size of items + facts
    budget: int = 0
    dropped: int = 0  # candidates that did not fit the budget
    used_graph: bool = True
    note: str | None = None
    effort: str = "medium"


class Retriever:
    def __init__(self, index: RepoIndex, searcher: HybridSearcher, cfg: Config) -> None:
        self.index = index
        self.searcher = searcher
        self.cfg = cfg

    def retrieve(
        self,
        query: str,
        *,
        use_graph: bool = True,
        k: int | None = None,
        mode: str = "hybrid",
        budget_tokens: int | None = None,
        effort: str = "medium",
    ) -> Retrieval:
        """Gather context for a question. Raises ValueError for an empty question."""
        profile = self.cfg.retrieval_profile
        base_budget = profile.context_tokens
        if effort == "high" and budget_tokens is None:
            base_budget = int(base_budget * 1.5)
        budget = budget_tokens if budget_tokens is not None else base_budget
        if budget < 100:
            raise ValueError("budget_tokens must be at least 100")
        search_k = k
        if search_k is None and effort == "high":
            search_k = min(self.cfg.search_top_k + 4, 16)
        hits = self.searcher.search(query, k=search_k, mode=mode)
        result = Retrieval(query=query, budget=budget, used_graph=use_graph, note=self.searcher.note, effort=effort)
        if not hits:
            return result

        top = hits[0].score or 1.0
        seeds = [ContextItem(h.chunk, h.score / top, "search", "matched the question") for h in hits]
        graph_items: dict[str, ContextItem] = {}
        facts: list[str] = []
        if use_graph:
            hops = profile.graph_hops + 1 if effort == "high" and profile.graph_hops < 3 else profile.graph_hops
            graph_items = self._expand(query, seeds, hops)
            facts = self._facts(query, seeds)
        fact_tokens = sum(estimate_tokens(f) for f in facts)
        ordered = seeds + sorted(graph_items.values(), key=lambda i: (-i.score, i.chunk.id))[: max(2, len(seeds))]

        remaining = budget - fact_tokens
        for item in ordered:
            cost = estimate_tokens(item.chunk.text) + ITEM_OVERHEAD_TOKENS
            if cost <= remaining:
                result.items.append(item)
                remaining -= cost
            elif not result.items:  # never return nothing: clip the best hit to fit
                room = max(int((remaining - ITEM_OVERHEAD_TOKENS) * CHARS_PER_TOKEN), 80)
                clipped = replace(item.chunk, text=item.chunk.text[:room] + TRUNCATION_NOTE)
                result.items.append(replace(item, chunk=clipped))
                remaining = 0
            else:
                result.dropped += 1
        result.facts = facts
        result.tokens = budget - remaining
        return result

    # ------------------------------------------------------------ graph expansion
    def _expand(self, query: str, seeds: list[ContextItem], hops: int) -> dict[str, ContextItem]:
        graph = self.index.graph
        asking_about_tests = "test" in query.lower()
        seen = {s.chunk.id for s in seeds}
        found: dict[str, ContextItem] = {}
        for seed in seeds:
            node_id = graph.node_id_for_chunk(seed.chunk.id)
            if node_id is None:
                continue
            neighbors = graph.neighbors(
                [node_id], hops=hops, edge_types=EXPAND_EDGES, min_confidence=GUESS, direction="both"
            )
            for nb in neighbors:
                data = graph.node(nb.node_id)
                if data["type"] not in (NODE_FUNCTION, NODE_CLASS, NODE_DOC) or not data.get("chunk_ids"):
                    continue
                chunk_id = data["chunk_ids"][0]  # first part of a long function is enough
                if chunk_id in seen:
                    continue
                chunk = self.index.chunk(chunk_id)
                score = seed.score * GRAPH_SCORE_SCALE * nb.score * GRAPH_DECAY ** (nb.distance - 1)
                if chunk.is_test and not asking_about_tests:
                    score *= self.cfg.test_file_penalty
                if chunk_id not in found or score > found[chunk_id].score:
                    found[chunk_id] = ContextItem(chunk, score, "graph", self._reason(graph, nb))
        return found

    @staticmethod
    def _reason(graph: KnowledgeGraph, nb: Neighbor) -> str:
        outgoing = graph.g.has_edge(nb.from_id, nb.node_id, key=nb.edge_type)  # from -> neighbour
        phrase = _RELATION_TEXT[(nb.edge_type, outgoing)]
        return f"{phrase} {graph.label(nb.from_id)} ({nb.confidence})"

    # ------------------------------------------------------------ facts
    def _facts(self, query: str, seeds: list[ContextItem]) -> list[str]:
        """Short relationship lines for the hits, for symbols named in the question, and named files."""
        graph, index = self.index.graph, self.index
        node_ids: list[str] = []
        for seed in seeds[:5]:
            node_id = graph.node_id_for_chunk(seed.chunk.id)
            if node_id and node_id not in node_ids:
                node_ids.append(node_id)
        for pos in sorted(HybridSearcher._named_chunks(query, index.symbols)):
            node_id = graph.node_id_for_chunk(index.chunks[pos].id)
            if node_id and node_id not in node_ids:
                node_ids.append(node_id)

        lines: list[str] = []
        for node_id in node_ids[:MAX_FACT_NODES]:
            line = self._node_fact(node_id)
            if line:
                lines.append(line)
        file_paths = {index.chunks[pos].path for pos in HybridSearcher._named_chunks(query, index.paths)}
        for path in sorted(file_paths)[:3]:
            line = self._file_fact(path)
            if line:
                lines.append(line)

        kept: list[str] = []
        used = 0
        for line in lines:
            cost = estimate_tokens(line)
            if used + cost > MAX_FACT_TOKENS:
                break
            kept.append(line)
            used += cost
        return kept

    def _node_fact(self, node_id: str) -> str | None:
        graph = self.index.graph
        data = graph.node(node_id)
        if data["type"] not in (NODE_FUNCTION, NODE_CLASS):
            return None
        parts: list[str] = []
        callers = graph.callers(node_id)[:5]
        if callers:
            parts.append("called by " + ", ".join(f"{graph.label(n)} ({e['confidence']})" for n, e in callers))
        callees = graph.callees(node_id)[:6]
        if callees:
            parts.append("calls " + ", ".join(f"{graph.label(n)} ({e['confidence']})" for n, e in callees))
        libs = graph.edges_of(node_id, incoming=False, edge_types=[EDGE_USES])[:5]
        if libs:
            parts.append("uses libraries " + ", ".join(
                f"{graph.node(n)['name']} ({', '.join(e['names'][:2])})" for n, e in libs))
        if data.get("decorators"):
            parts.append("decorators " + ", ".join("@" + d[:80] for d in data["decorators"][:4]))
        bases = graph.edges_of(node_id, incoming=False, edge_types=[EDGE_INHERITS])
        if bases:
            parts.append("inherits " + ", ".join(
                f"{graph.label(n)} ({e['confidence']})" for n, e in bases[:4]))
        if not parts:
            return None
        kind = "class" if data["type"] == NODE_CLASS else "function"
        return f"{graph.label(node_id)} [{kind}]: " + "; ".join(parts)

    def _file_fact(self, path: str) -> str | None:
        graph = self.index.graph
        matches = graph.find(path, "File")
        if not matches:
            return None
        node_id = matches[0]
        imports = graph.edges_of(node_id, incoming=False, edge_types=[EDGE_IMPORTS])[:8]
        importers = graph.edges_of(node_id, incoming=True, edge_types=[EDGE_IMPORTS])[:6]
        parts = []
        if imports:
            parts.append("imports " + ", ".join(graph.label(n).replace(" (library)", "") for n, _ in imports))
        if importers:
            parts.append("imported by " + ", ".join(graph.label(n) for n, _ in importers))
        return f"{path} [file]: " + "; ".join(parts) if parts else None


# Only real source/doc/config extensions count, so "example.com:8080" is not mistaken for a citation.
_EXTENSIONS = "|".join(sorted({e.lstrip(".") for e in (*CODE_EXTENSIONS, *DOC_EXTENSIONS, *CONFIG_EXTENSIONS)}, key=len, reverse=True))
_CITATION_RE = re.compile(rf"([A-Za-z0-9_./\-]+\.(?:{_EXTENSIONS})):(\d+)(?:\s*[-\u2013]\s*(\d+))?")


def verify_citations(text: str, items: list[ContextItem]) -> tuple[list[str], list[str]]:
    """Find path:line citations in an answer and check each against the context that was shown.

    A citation is valid if some context chunk is in the same file and its line range overlaps
    the cited range. Returns (all citations found, the invalid ones). This catches the model
    inventing files or line numbers that were never in its context.
    """
    cited: list[str] = []
    invalid: list[str] = []
    for match in _CITATION_RE.finditer(text):
        path, start_text, end_text = match.group(1), match.group(2), match.group(3)
        start = int(start_text)
        end = int(end_text) if end_text else start
        label = f"{path}:{start}-{end}" if end_text else f"{path}:{start}"
        if label in cited:
            continue
        cited.append(label)
        ok = any(
            (item.chunk.path == path or item.chunk.path.endswith("/" + path))
            and start <= item.chunk.end_line
            and end >= item.chunk.start_line
            for item in items
        )
        if not ok:
            invalid.append(label)
    return cited, invalid
