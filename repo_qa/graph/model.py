"""The KnowledgeGraph container: typed nodes, typed edges with confidence, and queries.

Wraps a networkx MultiDiGraph. The edge type is used as the edge key, so there is at most
one edge of each type between two nodes; adding it again merges the evidence (highest
confidence wins, names are combined). That keeps the graph small and deterministic.

The graph saves to plain, readable JSON (not pickle), so a saved index can be inspected
and is safe to load.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import networkx as nx

from common.errors import IngestError

from .schema import (
    CERTAIN,
    CONFIDENCE_RANK,
    CONFIDENCE_WEIGHT,
    EDGE_CALLS,
    EDGE_TYPES,
    GUESS,
    NODE_TYPES,
    at_least,
    best_confidence,
)

FORMAT_VERSION = 1


@dataclass(frozen=True)
class Neighbor:
    """A node reached while expanding from seed nodes."""

    node_id: str
    distance: int  # number of edges from the nearest seed
    from_id: str  # the node we came from
    edge_type: str
    confidence: str  # confidence of the LAST edge
    score: float  # product of edge weights along the path (1.0 = all certain)


class KnowledgeGraph:
    def __init__(self, meta: dict[str, Any] | None = None) -> None:
        self.g: nx.MultiDiGraph = nx.MultiDiGraph()
        self.meta: dict[str, Any] = dict(meta or {})
        self._chunk_index: dict[str, str] | None = None

    # ------------------------------------------------------------ building
    def add_node(self, node_id: str, node_type: str, **attrs: Any) -> None:
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unknown node type {node_type!r}")
        self.g.add_node(node_id, type=node_type, **attrs)
        self._chunk_index = None

    def add_edge(
        self,
        src: str,
        dst: str,
        edge_type: str,
        *,
        confidence: str = CERTAIN,
        via: str = "",
        names: Iterable[str] = (),
        **attrs: Any,
    ) -> None:
        """Add an edge, or merge into the existing one of the same type."""
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"Unknown edge type {edge_type!r}")
        if confidence not in CONFIDENCE_RANK:
            raise ValueError(f"Unknown confidence {confidence!r}")
        if src not in self.g or dst not in self.g:
            raise KeyError(f"Both ends of an edge must exist: {src!r} -> {dst!r}")
        if self.g.has_edge(src, dst, key=edge_type):
            data = self.g[src][dst][edge_type]
            if CONFIDENCE_RANK[confidence] > CONFIDENCE_RANK[data["confidence"]]:
                data["via"] = via
            data["confidence"] = best_confidence(data["confidence"], confidence)
            data["names"] = sorted(set(data["names"]) | set(names))
            data.update(attrs)
        else:
            self.g.add_edge(
                src, dst, key=edge_type, type=edge_type, confidence=confidence,
                via=via, names=sorted(set(names)), **attrs,
            )

    # ------------------------------------------------------------ reading
    def __len__(self) -> int:
        return self.g.number_of_nodes()

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.g

    def node(self, node_id: str) -> dict[str, Any]:
        return self.g.nodes[node_id]

    def nodes(self, node_type: str | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
        for node_id, data in self.g.nodes(data=True):
            if node_type is None or data["type"] == node_type:
                yield node_id, data

    def edges(self, edge_type: str | None = None) -> Iterator[tuple[str, str, dict[str, Any]]]:
        for src, dst, data in self.g.edges(data=True):
            if edge_type is None or data["type"] == edge_type:
                yield src, dst, data

    def label(self, node_id: str) -> str:
        """A short human-readable name for a node, used in printouts and prompts."""
        data = self.g.nodes[node_id]
        kind = data["type"]
        if kind in ("Function", "Class"):
            return f"{data['path']}::{data['qualname']}"
        if kind == "DocSection":
            return f"{data['path']} § {data['title']}"
        if kind == "External":
            return f"{data['name']} (library)"
        return data.get("path", node_id)

    def find(self, name: str, node_type: str | None = None) -> list[str]:
        """Node ids matching a name: exact qualname/name/path/title, or a dotted/path suffix.

        'verify_token' finds every function with that name; 'User.check' finds the method;
        'app/auth/routes.py' (or just 'routes.py') finds the file; and the labels this
        tool prints work too: 'app/auth/routes.py::login'.
        """
        needle = name.strip()
        if "::" in needle:  # a label as printed by label(): path::qualname
            path_part, _, qual = needle.partition("::")
            return sorted(
                node_id for node_id, data in self.nodes(node_type)
                if data.get("qualname") == qual
                and isinstance(data.get("path"), str)
                and (data["path"] == path_part or data["path"].endswith("/" + path_part))
            )
        found: list[str] = []
        for node_id, data in self.nodes(node_type):
            candidates = [data.get(k) for k in ("name", "qualname", "path", "title")]
            if any(
                isinstance(c, str)
                and (c == needle or c.endswith("." + needle) or c.endswith("/" + needle))
                for c in candidates
            ):
                found.append(node_id)
        return sorted(found)

    def callers(self, node_id: str, *, min_confidence: str = GUESS) -> list[tuple[str, dict[str, Any]]]:
        """Nodes that call this one, with the edge data."""
        return self._edges_at(node_id, incoming=True, edge_types=(EDGE_CALLS,), min_confidence=min_confidence)

    def callees(self, node_id: str, *, min_confidence: str = GUESS) -> list[tuple[str, dict[str, Any]]]:
        """Nodes this one calls, with the edge data."""
        return self._edges_at(node_id, incoming=False, edge_types=(EDGE_CALLS,), min_confidence=min_confidence)

    def edges_of(
        self,
        node_id: str,
        *,
        incoming: bool,
        edge_types: Iterable[str],
        min_confidence: str = GUESS,
    ) -> list[tuple[str, dict[str, Any]]]:
        return self._edges_at(node_id, incoming=incoming, edge_types=tuple(edge_types), min_confidence=min_confidence)

    def _edges_at(
        self, node_id: str, *, incoming: bool, edge_types: tuple[str, ...], min_confidence: str
    ) -> list[tuple[str, dict[str, Any]]]:
        if node_id not in self.g:
            raise KeyError(node_id)
        rows = self.g.in_edges(node_id, data=True) if incoming else self.g.out_edges(node_id, data=True)
        out = []
        for src, dst, data in rows:
            if data["type"] in edge_types and at_least(data["confidence"], min_confidence):
                out.append((src if incoming else dst, data))
        out.sort(key=lambda item: (-CONFIDENCE_RANK[item[1]["confidence"]], item[0]))
        return out

    def neighbors(
        self,
        seeds: Iterable[str],
        *,
        hops: int = 1,
        edge_types: Iterable[str] | None = None,
        min_confidence: str = GUESS,
        direction: str = "both",
    ) -> list[Neighbor]:
        """Breadth-first expansion from seed nodes, best-scoring first.

        direction is "out", "in" or "both". A neighbour's score is the product of the
        confidence weights along its best path, so guesses rank below certain edges.
        """
        if direction not in ("out", "in", "both"):
            raise ValueError("direction must be 'out', 'in' or 'both'")
        wanted = set(edge_types) if edge_types is not None else None
        seed_list = [s for s in seeds if s in self.g]
        seed_set = set(seed_list)
        best: dict[str, Neighbor] = {}
        queue: deque[tuple[str, int, float]] = deque((s, 0, 1.0) for s in seed_list)
        seen_depth = {s: 0 for s in seed_list}
        while queue:
            current, depth, score = queue.popleft()
            if depth >= hops:
                continue
            steps: list[tuple[str, dict[str, Any]]] = []
            if direction in ("out", "both"):
                steps += [(dst, d) for _, dst, d in self.g.out_edges(current, data=True)]
            if direction in ("in", "both"):
                steps += [(src, d) for src, _, d in self.g.in_edges(current, data=True)]
            for other, data in steps:
                if wanted is not None and data["type"] not in wanted:
                    continue
                if not at_least(data["confidence"], min_confidence):
                    continue
                new_score = score * CONFIDENCE_WEIGHT[data["confidence"]]
                existing = best.get(other)
                if other in seed_set:
                    continue
                if existing is None or new_score > existing.score:
                    best[other] = Neighbor(other, depth + 1, current, data["type"], data["confidence"], new_score)
                if other not in seen_depth or seen_depth[other] > depth + 1:
                    seen_depth[other] = depth + 1
                    queue.append((other, depth + 1, new_score))
        return sorted(best.values(), key=lambda n: (-n.score, n.distance, n.node_id))

    def shortest_path(
        self,
        src: str,
        dst: str,
        *,
        directed: bool = True,
        min_confidence: str = GUESS,
        edge_types: Iterable[str] | None = None,
    ) -> dict[str, Any] | None:
        """Find the shortest dependency or call path between two nodes.

        src and dst can be exact node ids, or names resolvable by find().
        """
        src_id = src if src in self.g else (self.find(src)[0] if self.find(src) else None)
        dst_id = dst if dst in self.g else (self.find(dst)[0] if self.find(dst) else None)
        if not src_id or not dst_id or src_id not in self.g or dst_id not in self.g:
            return None

        wanted = set(edge_types) if edge_types is not None else None
        sub_g = nx.MultiDiGraph()
        for u, v, k, d in self.g.edges(data=True, keys=True):
            if wanted is not None and d.get("type") not in wanted:
                continue
            if not at_least(d.get("confidence", GUESS), min_confidence):
                continue
            sub_g.add_edge(u, v, key=k, **d)

        graph_to_search = sub_g if directed else sub_g.to_undirected()
        try:
            path_nodes = nx.shortest_path(graph_to_search, src_id, dst_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        steps = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            candidates = []
            if sub_g.has_edge(u, v):
                for k, d in sub_g[u][v].items():
                    candidates.append((d, "outgoing"))
            if not directed and sub_g.has_edge(v, u):
                for k, d in sub_g[v][u].items():
                    candidates.append((d, "incoming"))
            if not candidates:
                continue
            candidates.sort(key=lambda item: -CONFIDENCE_RANK.get(item[0].get("confidence", GUESS), 0))
            best_edge, direction = candidates[0]
            steps.append({
                "from_id": u,
                "from_label": self.label(u),
                "to_id": v,
                "to_label": self.label(v),
                "direction": direction,
                "type": best_edge.get("type", ""),
                "confidence": best_edge.get("confidence", "certain"),
                "details": (", ".join(best_edge.get("names", [])) if best_edge.get("names") else "") or best_edge.get("via", ""),
            })

        return {
            "source_id": src_id,
            "source_label": self.label(src_id),
            "target_id": dst_id,
            "target_label": self.label(dst_id),
            "directed": directed,
            "hops": len(steps),
            "path_nodes": [self.label(n) for n in path_nodes],
            "steps": steps,
        }

    def hubs(
        self,
        node_types: Iterable[str] | None = None,
        limit: int = 10,
        edge_types: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most connected nodes (key architecture hubs), sorted by in-degree."""
        wanted_node_types = set(node_types) if node_types is not None else None
        wanted_edge_types = set(edge_types) if edge_types is not None else None

        in_degrees: dict[str, int] = {}
        out_degrees: dict[str, int] = {}
        for u, v, data in self.g.edges(data=True):
            if wanted_edge_types is not None and data.get("type") not in wanted_edge_types:
                continue
            out_degrees[u] = out_degrees.get(u, 0) + 1
            in_degrees[v] = in_degrees.get(v, 0) + 1

        results = []
        for node_id, data in self.g.nodes(data=True):
            ntype = data.get("type", "")
            if wanted_node_types is not None and ntype not in wanted_node_types:
                continue
            in_deg = in_degrees.get(node_id, 0)
            out_deg = out_degrees.get(node_id, 0)
            if in_deg == 0 and out_deg == 0:
                continue
            results.append({
                "id": node_id,
                "label": self.label(node_id),
                "type": ntype,
                "path": data.get("path", ""),
                "in_degree": in_deg,
                "out_degree": out_deg,
                "total_degree": in_deg + out_deg,
            })
        results.sort(key=lambda x: (-x["in_degree"], -x["total_degree"], x["label"]))
        return results[:limit]

    def subgraph_for_visualization(
        self,
        seeds: Iterable[str] | None = None,
        hops: int = 1,
        max_nodes: int = 60,
        edge_types: Iterable[str] | None = None,
        min_confidence: str = GUESS,
    ) -> dict[str, list[dict[str, Any]]]:
        """Extract a structured subgraph formatted for visual rendering."""
        wanted_edges = set(edge_types) if edge_types is not None else None

        seed_list = [s for s in (seeds or []) if s in self.g]
        if seed_list:
            neighbors = self.neighbors(seed_list, hops=hops, edge_types=wanted_edges, min_confidence=min_confidence)
            included = set(seed_list) | {n.node_id for n in neighbors[:max_nodes - len(seed_list)]}
        else:
            top_hubs = self.hubs(limit=max_nodes, edge_types=wanted_edges)
            included = {h["id"] for h in top_hubs}
            if not included:
                included = set(list(self.g.nodes)[:max_nodes])

        node_rows = []
        for nid in sorted(included):
            data = self.g.nodes[nid]
            lines = ""
            if "start_line" in data and "end_line" in data:
                lines = f"{data['start_line']}-{data['end_line']}"
            node_rows.append({
                "id": nid,
                "label": self.label(nid),
                "type": data.get("type", "Unknown"),
                "path": data.get("path", ""),
                "lines": lines,
                "is_seed": nid in seed_list,
            })

        edge_rows = []
        seen_edges = set()
        for u in included:
            for _, v, data in self.g.out_edges(u, data=True):
                if v not in included:
                    continue
                etype = data.get("type", "")
                if wanted_edges is not None and etype not in wanted_edges:
                    continue
                if not at_least(data.get("confidence", GUESS), min_confidence):
                    continue
                edge_key = (u, v, etype)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edge_rows.append({
                        "source": u,
                        "target": v,
                        "type": etype,
                        "confidence": data.get("confidence", "certain"),
                        "details": (", ".join(data.get("names", [])) if data.get("names") else "") or data.get("via", ""),
                    })

        return {"nodes": node_rows, "edges": edge_rows}

    # ------------------------------------------------------------ chunk links
    def node_id_for_chunk(self, chunk_id: str) -> str | None:
        """The graph node a retrieval chunk belongs to (a vector hit's way into the graph)."""
        if self._chunk_index is None:
            index: dict[str, str] = {}
            for node_id, data in self.g.nodes(data=True):
                for cid in data.get("chunk_ids", ()):
                    index[cid] = node_id
            self._chunk_index = index
        return self._chunk_index.get(chunk_id)

    # ------------------------------------------------------------ statistics
    def counts(self) -> dict[str, dict[str, int]]:
        nodes: dict[str, int] = {}
        for _, data in self.g.nodes(data=True):
            nodes[data["type"]] = nodes.get(data["type"], 0) + 1
        edges: dict[str, int] = {}
        for _, _, data in self.g.edges(data=True):
            edges[data["type"]] = edges.get(data["type"], 0) + 1
        return {"nodes": dict(sorted(nodes.items())), "edges": dict(sorted(edges.items()))}

    def edge_confidence_counts(self, edge_type: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, _, data in self.edges(edge_type):
            counts[data["confidence"]] = counts.get(data["confidence"], 0) + 1
        return counts

    # ------------------------------------------------------------ storage
    def to_dict(self) -> dict[str, Any]:
        nodes = [{"id": nid, **data} for nid, data in sorted(self.g.nodes(data=True), key=lambda x: x[0])]
        edges = [
            {"src": s, "dst": d, **data}
            for s, d, data in sorted(self.g.edges(data=True), key=lambda x: (x[0], x[1], x[2]["type"]))
        ]
        return {"format": FORMAT_VERSION, "meta": self.meta, "nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeGraph":
        if not isinstance(payload, dict) or payload.get("format") != FORMAT_VERSION:
            raise IngestError("Not a knowledge graph file, or saved by an incompatible version.")
        graph = cls(meta=payload.get("meta") or {})
        try:
            for node in payload["nodes"]:
                attrs = {k: v for k, v in node.items() if k not in ("id", "type")}
                graph.add_node(node["id"], node["type"], **attrs)
            for edge in payload["edges"]:
                attrs = {k: v for k, v in edge.items() if k not in ("src", "dst", "type", "confidence", "via", "names")}
                graph.add_edge(
                    edge["src"], edge["dst"], edge["type"],
                    confidence=edge["confidence"], via=edge.get("via", ""),
                    names=edge.get("names", ()), **attrs,
                )
        except (KeyError, ValueError, TypeError) as exc:
            raise IngestError(f"The knowledge graph file is damaged: {exc}") from exc
        return graph

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tmp.replace(target)  # atomic: a crash never leaves a half-written graph

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IngestError(f"Cannot read the knowledge graph at {path}: {type(exc).__name__}") from exc
        return cls.from_dict(payload)
