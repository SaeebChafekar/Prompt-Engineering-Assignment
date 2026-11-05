"""Tests for the KnowledgeGraph container: merging, queries, neighbour scoring and storage.

Run from the project root:

    python -m unittest tests.test_graph_model -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from common.errors import IngestError
from repo_qa.graph import KnowledgeGraph
from repo_qa.graph.schema import (
    CERTAIN, EDGE_CALLS, EDGE_CONTAINS, EDGE_IMPORTS, GUESS, LIKELY, at_least, best_confidence,
)


def make_graph() -> KnowledgeGraph:
    """   main --certain--> a --likely--> b --guess--> c        d (isolated)
          main --guess----> c
    """
    g = KnowledgeGraph(meta={"repo": "toy"})
    g.add_node("file:x.py", "File", path="x.py", name="x.py", is_test=False, chunk_ids=["x.py:1-9"])
    for name in ("main", "a", "b", "c", "d"):
        g.add_node(f"func:x.py::{name}", "Function", name=name, qualname=name, path="x.py",
                   chunk_ids=[f"x.py:{name}"])
    g.add_node("class:x.py::Svc", "Class", name="Svc", qualname="Svc", path="x.py")
    g.add_node("func:x.py::Svc.run", "Function", name="run", qualname="Svc.run", path="x.py")
    g.add_node("ext:flask", "External", name="flask", stdlib=False)
    f = lambda n: f"func:x.py::{n}"  # noqa: E731
    g.add_edge(f("main"), f("a"), EDGE_CALLS, confidence=CERTAIN, via="local", names=["a"])
    g.add_edge(f("a"), f("b"), EDGE_CALLS, confidence=LIKELY, via="inherited", names=["self.b"])
    g.add_edge(f("b"), f("c"), EDGE_CALLS, confidence=GUESS, via="method-name", names=["x.c"])
    g.add_edge(f("main"), f("c"), EDGE_CALLS, confidence=GUESS, via="method-name", names=["o.c"])
    g.add_edge("file:x.py", f("main"), EDGE_CONTAINS)
    g.add_edge("file:x.py", "ext:flask", EDGE_IMPORTS, names=["Flask"])
    return g


F = lambda n: f"func:x.py::{n}"  # noqa: E731


class BuildingTests(unittest.TestCase):
    def test_add_edge_merges_instead_of_duplicating(self) -> None:
        g = make_graph()
        g.add_edge(F("main"), F("c"), EDGE_CALLS, confidence=CERTAIN, via="local", names=["c", "o.c"])
        self.assertEqual(g.g.number_of_edges(F("main"), F("c")), 1)
        data = g.g[F("main")][F("c")][EDGE_CALLS]
        self.assertEqual(data["confidence"], CERTAIN)  # upgraded
        self.assertEqual(data["via"], "local")
        self.assertEqual(data["names"], ["c", "o.c"])  # combined, sorted, de-duplicated

    def test_merging_a_weaker_edge_does_not_downgrade(self) -> None:
        g = make_graph()
        g.add_edge(F("main"), F("a"), EDGE_CALLS, confidence=GUESS, via="method-name", names=["x.a"])
        data = g.g[F("main")][F("a")][EDGE_CALLS]
        self.assertEqual((data["confidence"], data["via"]), (CERTAIN, "local"))
        self.assertEqual(data["names"], ["a", "x.a"])

    def test_different_edge_types_between_the_same_nodes_coexist(self) -> None:
        g = make_graph()
        g.add_edge(F("main"), F("a"), EDGE_CONTAINS)
        self.assertEqual(len(list(g.g.out_edges(F("main"), data=True))), 3)  # CALLS a, CALLS c, CONTAINS a

    def test_validation(self) -> None:
        g = make_graph()
        with self.assertRaises(ValueError):
            g.add_node("x", "Planet")
        with self.assertRaises(ValueError):
            g.add_edge(F("a"), F("b"), "LOVES")
        with self.assertRaises(ValueError):
            g.add_edge(F("a"), F("b"), EDGE_CALLS, confidence="sure")
        with self.assertRaises(KeyError):
            g.add_edge(F("a"), "func:missing", EDGE_CALLS)

    def test_confidence_helpers(self) -> None:
        self.assertEqual(best_confidence(GUESS, LIKELY), LIKELY)
        self.assertEqual(best_confidence(CERTAIN, LIKELY), CERTAIN)
        self.assertTrue(at_least(CERTAIN, LIKELY))
        self.assertTrue(at_least(LIKELY, LIKELY))
        self.assertFalse(at_least(GUESS, LIKELY))


class QueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.g = make_graph()

    def test_find(self) -> None:
        self.assertEqual(self.g.find("main"), [F("main")])
        self.assertEqual(self.g.find("run"), [F("Svc.run")])
        self.assertEqual(self.g.find("Svc.run"), [F("Svc.run")])
        self.assertEqual(self.g.find("x.py", "File"), ["file:x.py"])
        self.assertEqual(self.g.find("flask"), ["ext:flask"])
        self.assertEqual(self.g.find("nothing-like-this"), [])
        self.assertEqual(self.g.find("main", "Class"), [])  # type filter

    def test_find_accepts_printed_labels(self) -> None:
        self.assertEqual(self.g.find("x.py::Svc.run"), [F("Svc.run")])
        self.assertEqual(self.g.find("x.py::main", "Function"), [F("main")])
        self.assertEqual(self.g.find("other.py::main"), [])
        self.assertEqual(self.g.find("x.py::nothing"), [])
        for node_id in self.g.find("x.py"):  # every label round-trips
            if self.g.node(node_id)["type"] in ("Function", "Class"):
                self.assertEqual(self.g.find(self.g.label(node_id)), [node_id])

    def test_find_does_not_match_partial_words(self) -> None:
        self.assertEqual(self.g.find("ain"), [])  # not a suffix at a name boundary
        self.assertEqual(self.g.find("un"), [])

    def test_callers_and_callees_sorted_by_confidence(self) -> None:
        callees = self.g.callees(F("main"))
        self.assertEqual([n for n, _ in callees], [F("a"), F("c")])  # certain before guess
        callers = self.g.callers(F("c"))
        self.assertEqual({n for n, _ in callers}, {F("b"), F("main")})

    def test_min_confidence_filter(self) -> None:
        self.assertEqual([n for n, _ in self.g.callees(F("main"), min_confidence=LIKELY)], [F("a")])
        self.assertEqual(self.g.callers(F("c"), min_confidence=LIKELY), [])
        self.assertEqual([n for n, _ in self.g.callees(F("a"), min_confidence=LIKELY)], [F("b")])

    def test_unknown_node_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.g.callers("func:nope")

    def test_edges_of_by_type(self) -> None:
        rows = self.g.edges_of("file:x.py", incoming=False, edge_types=[EDGE_IMPORTS])
        self.assertEqual([n for n, _ in rows], ["ext:flask"])

    def test_labels(self) -> None:
        self.assertEqual(self.g.label(F("Svc.run")), "x.py::Svc.run")
        self.assertEqual(self.g.label("ext:flask"), "flask (library)")
        self.assertEqual(self.g.label("file:x.py"), "x.py")

    def test_counts(self) -> None:
        counts = self.g.counts()
        self.assertEqual(counts["nodes"], {"Class": 1, "External": 1, "File": 1, "Function": 6})
        self.assertEqual(counts["edges"], {"CALLS": 4, "CONTAINS": 1, "IMPORTS": 1})
        self.assertEqual(self.g.edge_confidence_counts(EDGE_CALLS), {CERTAIN: 1, LIKELY: 1, GUESS: 2})

    def test_chunk_index(self) -> None:
        self.assertEqual(self.g.node_id_for_chunk("x.py:main"), F("main"))
        self.assertEqual(self.g.node_id_for_chunk("x.py:1-9"), "file:x.py")
        self.assertIsNone(self.g.node_id_for_chunk("nope"))
        self.g.add_node(F("late"), "Function", name="late", qualname="late", path="x.py", chunk_ids=["x.py:late"])
        self.assertEqual(self.g.node_id_for_chunk("x.py:late"), F("late"))  # index is refreshed after changes


class NeighborTests(unittest.TestCase):
    def setUp(self) -> None:
        self.g = make_graph()

    def test_one_hop_out_scores_by_confidence(self) -> None:
        found = {n.node_id: n for n in self.g.neighbors([F("main")], hops=1, direction="out", edge_types=[EDGE_CALLS])}
        self.assertEqual(set(found), {F("a"), F("c")})
        self.assertEqual(found[F("a")].score, 1.0)
        self.assertEqual(found[F("c")].score, 0.3)
        self.assertEqual(found[F("a")].distance, 1)

    def test_results_are_best_first(self) -> None:
        order = [n.node_id for n in self.g.neighbors([F("main")], hops=1, direction="out")]
        self.assertEqual(order[0], F("a"))

    def test_two_hops_multiply_weights_along_the_path(self) -> None:
        found = {n.node_id: n for n in self.g.neighbors([F("main")], hops=2, direction="out", edge_types=[EDGE_CALLS])}
        self.assertEqual(set(found), {F("a"), F("b"), F("c")})
        self.assertAlmostEqual(found[F("b")].score, 1.0 * 0.7)  # certain then likely
        self.assertEqual(found[F("b")].distance, 2)
        self.assertEqual(found[F("c")].score, 0.3)  # the direct guess still beats the longer path (0.7 * 0.3)

    def test_min_confidence_prunes_paths(self) -> None:
        found = {n.node_id for n in self.g.neighbors([F("main")], hops=3, direction="out", min_confidence=LIKELY)}
        self.assertEqual(found, {F("a"), F("b")})

    def test_incoming_and_both_directions(self) -> None:
        incoming = {n.node_id for n in self.g.neighbors([F("c")], hops=1, direction="in")}
        self.assertEqual(incoming, {F("b"), F("main")})
        both = {n.node_id for n in self.g.neighbors([F("a")], hops=1, direction="both")}
        self.assertEqual(both, {F("main"), F("b")})

    def test_seeds_are_not_returned_and_unknown_seeds_are_ignored(self) -> None:
        found = {n.node_id for n in self.g.neighbors([F("a"), F("main"), "func:missing"], hops=2, direction="out")}
        self.assertNotIn(F("a"), found)
        self.assertNotIn(F("main"), found)

    def test_isolated_node_and_zero_hops(self) -> None:
        self.assertEqual(self.g.neighbors([F("d")], hops=3), [])
        self.assertEqual(self.g.neighbors([F("main")], hops=0), [])

    def test_cycles_terminate(self) -> None:
        self.g.add_edge(F("c"), F("main"), EDGE_CALLS, confidence=CERTAIN)
        found = {n.node_id for n in self.g.neighbors([F("main")], hops=10, direction="out")}
        self.assertEqual(found, {F("a"), F("b"), F("c")})

    def test_bad_direction(self) -> None:
        with self.assertRaises(ValueError):
            self.g.neighbors([F("a")], direction="sideways")

    def test_shortest_path(self) -> None:
        # main -> a -> b -> c
        path = self.g.shortest_path(F("main"), F("c"))
        self.assertIsNotNone(path)
        self.assertEqual(path["hops"], 1)  # main -> c direct guess edge exists
        self.assertEqual(path["source_id"], F("main"))
        self.assertEqual(path["target_id"], F("c"))

        # From c to main directed should fail
        rev = self.g.shortest_path(F("c"), F("main"), directed=True)
        self.assertIsNone(rev)

        # From c to main undirected should succeed
        undir = self.g.shortest_path(F("c"), F("main"), directed=False)
        self.assertIsNotNone(undir)
        self.assertEqual(undir["hops"], 1)

        # Path between disconnected node d and main
        no_path = self.g.shortest_path(F("d"), F("main"), directed=False)
        self.assertIsNone(no_path)

    def test_hubs(self) -> None:
        hubs = self.g.hubs(node_types=["Function"], limit=5)
        self.assertTrue(len(hubs) > 0)
        # Function c has in-degree of 2 (called by b and main)
        c_hub = next((h for h in hubs if h["id"] == F("c")), None)
        self.assertIsNotNone(c_hub)
        self.assertEqual(c_hub["in_degree"], 2)

    def test_subgraph_for_visualization(self) -> None:
        sub = self.g.subgraph_for_visualization(seeds=[F("main")], hops=1)
        self.assertIn("nodes", sub)
        self.assertIn("edges", sub)
        node_ids = [n["id"] for n in sub["nodes"]]
        self.assertIn(F("main"), node_ids)
        self.assertIn(F("a"), node_ids)
        self.assertIn(F("c"), node_ids)
        # Verify edge structure
        self.assertTrue(any(e["source"] == F("main") and e["target"] == F("a") for e in sub["edges"]))


class StorageTests(unittest.TestCase):
    def test_round_trip_preserves_everything(self) -> None:
        g = make_graph()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep" / "dir" / "graph.json"
            g.save(path)
            loaded = KnowledgeGraph.load(path)
            self.assertEqual(loaded.to_dict(), g.to_dict())
            self.assertEqual(loaded.meta, {"repo": "toy"})
            data = loaded.g[F("a")][F("b")][EDGE_CALLS]
            self.assertEqual((data["confidence"], data["via"], data["names"]), (LIKELY, "inherited", ["self.b"]))
            self.assertEqual(loaded.node_id_for_chunk("x.py:main"), F("main"))

    def test_output_is_deterministic(self) -> None:
        self.assertEqual(json.dumps(make_graph().to_dict()), json.dumps(make_graph().to_dict()))

    def test_save_replaces_existing_file_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            path.write_text("old content")
            make_graph().save(path)
            self.assertTrue(path.read_text().startswith("{"))
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["graph.json"])  # no leftover .tmp

    def test_loading_bad_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaises(IngestError):
                KnowledgeGraph.load(tmp_path / "missing.json")
            (tmp_path / "garbage.json").write_text("not json {{{")
            with self.assertRaises(IngestError):
                KnowledgeGraph.load(tmp_path / "garbage.json")
            (tmp_path / "wrong.json").write_text(json.dumps({"hello": "world"}))
            with self.assertRaises(IngestError):
                KnowledgeGraph.load(tmp_path / "wrong.json")
            future = make_graph().to_dict()
            future["format"] = 99
            (tmp_path / "future.json").write_text(json.dumps(future))
            with self.assertRaises(IngestError) as ctx:
                KnowledgeGraph.load(tmp_path / "future.json")
            self.assertIn("incompatible", str(ctx.exception))
            damaged = make_graph().to_dict()
            damaged["edges"][0]["dst"] = "func:does-not-exist"
            (tmp_path / "damaged.json").write_text(json.dumps(damaged))
            with self.assertRaises(IngestError) as ctx2:
                KnowledgeGraph.load(tmp_path / "damaged.json")
            self.assertIn("damaged", str(ctx2.exception))


if __name__ == "__main__":
    unittest.main()
