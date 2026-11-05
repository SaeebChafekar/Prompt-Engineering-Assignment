"""Tests for the index package: tokenizer, BM25, vector store, building, saving and hybrid search.

A fake embedder (hashed bag of words) stands in for Ollama, so no model or GPU is needed.

    python -m unittest tests.test_index -v
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from common.config import Config
from common.errors import ConfigError, IngestError, InvalidOutputError
from repo_qa.graph import build_graph
from repo_qa.index import BM25Index, HybridSearcher, RepoIndex, VectorStore, build_index, query_tokens, tokenize
from repo_qa.index.__main__ import main as cli_main
from repo_qa.index.bm25 import split_identifier
from repo_qa.ingest.pipeline import ingest_directory

FILES = {
    "app/__init__.py": "VERSION = 1\n",
    "app/auth.py": '''"""Login and password checks."""


def login(request):
    """Log a user in with a password."""
    user = find_user(request)
    return verify_password(user, request)


def verify_password(user, password):
    """Check the password hash for the user."""
    return user.hash == password


def find_user(request):
    return request.user
''',
    "app/billing.py": '''def create_invoice(order):
    """Create an invoice for an order."""
    total = sum(item.price for item in order.items)
    return {"total": total}


def sendReceipt(invoice):
    return invoice
''',
    "tests/test_auth.py": '''from app.auth import login


def test_login_checks_password():
    assert login(None) is None  # exercises verify_password and the password hash
''',
    "README.md": "# Demo\n\nThis service handles login and invoices for customers.\n",
}


class FakeEmbedder:
    """Hashed bag-of-words vectors: texts sharing words get similar vectors. Counts calls."""

    model_name = "fake-embed"

    def __init__(self, dim: int = 64, fail_after: int | None = None) -> None:
        self.dim = dim
        self.calls = 0
        self.texts_embedded = 0
        self.fail_after = fail_after

    def embed(self, texts):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise InvalidOutputError("simulated embedding failure")
        self.texts_embedded += len(texts)
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for token in tokenize(text):
                v[zlib.crc32(token.encode()) % self.dim] += 1.0
            out.append(v)
        return out

    def health_check(self) -> None:
        pass

    def close(self) -> None:
        pass


class IndexCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls._tmp.name)
        root = cls.base / "demo"
        for rel, text in FILES.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        cls.cfg = Config(data_dir=str(cls.base / "data"), embed_batch_size=4, search_top_k=5)
        cls.result = ingest_directory(root, cls.cfg)
        cls.graph, _ = build_graph(cls.result)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def build(self, embedder=None, **kwargs) -> RepoIndex:
        return build_index(self.result, self.graph, embedder, self.cfg, **kwargs)


class TokenizerTests(unittest.TestCase):
    def test_split_identifier(self) -> None:
        self.assertEqual(split_identifier("verifyToken"), ["verify", "token"])
        self.assertEqual(split_identifier("verify_token"), ["verify", "token"])
        self.assertEqual(split_identifier("HTTPServer"), ["http", "server"])
        self.assertEqual(split_identifier("user2Name"), ["user", "2", "name"])
        self.assertEqual(split_identifier("login"), ["login"])

    def test_tokenize_keeps_whole_identifier_and_parts(self) -> None:
        tokens = tokenize("def verify_token(sendReceipt): x.check_password()")
        for expected in ("verify_token", "verify", "token", "sendreceipt", "send", "receipt", "check_password", "check", "password"):
            self.assertIn(expected, tokens)

    def test_query_tokens_drop_filler_words(self) -> None:
        self.assertEqual(query_tokens("Where is the login handled?"), ["login", "handled"])
        self.assertEqual(query_tokens("what does verify_password do"), ["verify_password", "verify", "password"])
        self.assertEqual(query_tokens(""), [])


class BM25Tests(unittest.TestCase):
    DOCS = ["def verify_password(user): check hash", "def create_invoice(order): total price", "invoice total for order order order", "login page"]

    def test_ranks_by_relevance(self) -> None:
        bm = BM25Index(self.DOCS)
        ranked = bm.search(query_tokens("invoice total"), 4)
        self.assertEqual([i for i, _ in ranked][:2], [2, 1])  # repeated terms in a short doc win
        self.assertTrue(all(score > 0 for _, score in ranked))

    def test_identifier_parts_match(self) -> None:
        bm = BM25Index(self.DOCS)
        self.assertEqual(bm.search(query_tokens("verifyPassword"), 3)[0][0], 0)

    def test_rare_terms_outweigh_common_ones(self) -> None:
        bm = BM25Index(["alpha beta", "alpha gamma", "alpha delta", "alpha beta"])
        self.assertEqual(bm.search(["alpha", "gamma"], 4)[0][0], 1)

    def test_edge_cases(self) -> None:
        self.assertEqual(BM25Index([]).search(["x"], 3), [])
        bm = BM25Index(self.DOCS)
        self.assertEqual(bm.search([], 3), [])
        self.assertEqual(bm.search(["nonexistentterm"], 3), [])
        self.assertEqual(bm.search(["invoice"], 0), [])
        self.assertEqual(len(bm.search(["invoice"], 1)), 1)


class VectorStoreTests(unittest.TestCase):
    def test_search_returns_best_first(self) -> None:
        store = VectorStore.from_lists([[1, 0], [0, 1], [1, 1]], ["a", "b", "c"], model="m")
        hits = store.search([1, 0.1], 3)
        self.assertEqual([i for i, _ in hits], [0, 2, 1])
        self.assertAlmostEqual(hits[0][1], 0.995, places=2)

    def test_scale_does_not_matter(self) -> None:
        store = VectorStore.from_lists([[10, 0], [0, 0.001]], ["a", "b"], model="m")
        self.assertEqual(store.search([0.001, 0], 1)[0][0], 0)

    def test_limit_larger_than_store_and_empty_store(self) -> None:
        store = VectorStore.from_lists([[1, 0], [0, 1]], ["a", "b"], model="m")
        self.assertEqual(len(store.search([1, 0], 10)), 2)
        self.assertEqual(VectorStore.from_lists([], [], model="m").search([1, 0], 3), [])

    def test_bad_input_is_rejected(self) -> None:
        with self.assertRaises(InvalidOutputError):
            VectorStore.from_lists([[1, 0]], ["a", "b"], model="m")  # count mismatch
        with self.assertRaises(InvalidOutputError):
            VectorStore.from_lists([[1, float("nan")]], ["a"], model="m")
        with self.assertRaises(InvalidOutputError):
            VectorStore.from_lists([[]], ["a"], model="m")
        store = VectorStore.from_lists([[1, 0]], ["a"], model="m")
        with self.assertRaises(InvalidOutputError) as ctx:
            store.search([1, 0, 0], 1)  # wrong dimension
        self.assertIn("different embedding model", str(ctx.exception))

    def test_zero_vector_matches_nothing_and_does_not_crash(self) -> None:
        store = VectorStore.from_lists([[0, 0], [1, 0]], ["zero", "one"], model="m")
        scores = dict(store.search([1, 0], 2))
        self.assertEqual(scores[0], 0.0)

    def test_save_load_round_trip_and_model_check(self) -> None:
        store = VectorStore.from_lists([[1, 0], [0, 1]], ["a", "b"], model="nomic")
        with tempfile.TemporaryDirectory() as tmp:
            store.save(Path(tmp))
            loaded = VectorStore.load(Path(tmp), expect_model="nomic")
            self.assertEqual(loaded.ids, ["a", "b"])
            np.testing.assert_allclose(loaded.vectors, store.vectors)
            with self.assertRaises(IngestError) as ctx:
                VectorStore.load(Path(tmp), expect_model="other-model")
            self.assertIn("Re-index", str(ctx.exception))
            (Path(tmp) / "vectors.npy").unlink()
            with self.assertRaises(IngestError):
                VectorStore.load(Path(tmp))


class BuildTests(IndexCase):
    def test_keyword_only_index(self) -> None:
        index = self.build(None)
        self.assertFalse(index.has_vectors)
        self.assertEqual(len(index.chunks), len(self.result.chunks))
        self.assertEqual(index.meta["vectors_embedded"], 0)

    def test_embeds_every_chunk_in_batches_with_progress(self) -> None:
        embedder, seen = FakeEmbedder(), []
        index = self.build(embedder, progress=lambda done, total: seen.append((done, total)))
        total = len(self.result.chunks)
        self.assertEqual(embedder.texts_embedded, total)
        self.assertEqual(seen[-1], (total, total))
        self.assertEqual(embedder.calls, -(-total // 4))  # batches of 4
        self.assertEqual(len(index.store), total)
        self.assertEqual(index.store.model, "fake-embed")

    def test_document_prefix_is_applied(self) -> None:
        captured = []

        class Spy(FakeEmbedder):
            def embed(self, texts):
                captured.extend(texts)
                return super().embed(texts)

        self.build(Spy())
        self.assertTrue(all(t.startswith("search_document: ") for t in captured))

    def test_embedding_failure_is_raised_not_swallowed(self) -> None:
        with self.assertRaises(InvalidOutputError):
            self.build(FakeEmbedder(fail_after=1))

    def test_time_limit(self) -> None:
        ticks = iter(range(0, 100_000, 5000))
        cfg = Config(embed_batch_size=2, index_timeout_s=10)
        with self.assertRaises(IngestError) as ctx:
            build_index(self.result, self.graph, FakeEmbedder(), cfg, clock=lambda: float(next(ticks)))
        self.assertIn("CHATBOT_INDEX_TIMEOUT_S", str(ctx.exception))

    def test_wrong_vector_count_from_embedder(self) -> None:
        class Broken(FakeEmbedder):
            def embed(self, texts):
                return super().embed(texts)[:-1]

        with self.assertRaises(IngestError):
            self.build(Broken())

    def test_unchanged_chunks_are_not_re_embedded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = self.build(FakeEmbedder())
            first.save(Path(tmp))
            again = FakeEmbedder()
            second = self.build(again, previous_dir=Path(tmp))
            self.assertEqual(again.calls, 0)
            self.assertEqual(second.meta["vectors_reused"], len(self.result.chunks))
            np.testing.assert_allclose(second.store.vectors, first.store.vectors, atol=1e-6)

    def test_changed_chunks_only_are_re_embedded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.build(FakeEmbedder()).save(Path(tmp))
            edited = ingest_directory(Path(self.result.root), self.cfg)
            victim = next(i for i, c in enumerate(edited.chunks) if c.symbol == "create_invoice")
            edited.chunks[victim] = type(edited.chunks[victim])(**{**edited.chunks[victim].to_dict(), "text": "def create_invoice(order):\n    return 0"})
            again = FakeEmbedder()
            build_index(edited, self.graph, again, self.cfg, previous_dir=Path(tmp))
            self.assertEqual(again.texts_embedded, 1)

    def test_a_different_model_does_not_reuse_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.build(FakeEmbedder()).save(Path(tmp))

            class Other(FakeEmbedder):
                model_name = "another-model"

            other = Other()
            self.build(other, previous_dir=Path(tmp))
            self.assertEqual(other.texts_embedded, len(self.result.chunks))

    def test_corrupt_previous_index_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.build(FakeEmbedder()).save(Path(tmp))
            (Path(tmp) / "chunks.jsonl").write_text("not json\n")
            embedder = FakeEmbedder()
            self.build(embedder, previous_dir=Path(tmp))
            self.assertEqual(embedder.texts_embedded, len(self.result.chunks))


class SaveLoadTests(IndexCase):
    def test_round_trip(self) -> None:
        index = self.build(FakeEmbedder())
        with tempfile.TemporaryDirectory() as tmp:
            index.save(Path(tmp))
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["chunks.jsonl", "graph.json", "index.json", "store.json", "vectors.npy"])
            loaded = RepoIndex.load(Path(tmp), expect_model="fake-embed")
            self.assertEqual([c.id for c in loaded.chunks], [c.id for c in index.chunks])
            self.assertEqual(loaded.chunks[0], index.chunks[0])
            self.assertEqual(loaded.graph.to_dict(), index.graph.to_dict())
            self.assertEqual(loaded.repo_name, "demo")
            self.assertTrue(loaded.has_vectors)
            self.assertEqual(json.loads((Path(tmp) / "index.json").read_text())["embed_model"], "fake-embed")

    def test_keyword_only_save_removes_stale_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.build(FakeEmbedder()).save(Path(tmp))
            self.build(None).save(Path(tmp))
            self.assertFalse((Path(tmp) / "vectors.npy").exists())
            self.assertFalse(RepoIndex.load(Path(tmp)).has_vectors)

    def test_load_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(IngestError):
                RepoIndex.load(Path(tmp))  # no index.json
            self.build(FakeEmbedder()).save(Path(tmp))
            with self.assertRaises(IngestError) as ctx:
                RepoIndex.load(Path(tmp), expect_model="a-different-model")
            self.assertIn("Re-index", str(ctx.exception))
            (Path(tmp) / "chunks.jsonl").write_text("{broken")
            with self.assertRaises(IngestError):
                RepoIndex.load(Path(tmp))

    def test_vectors_that_do_not_match_chunks_are_rejected(self) -> None:
        index = self.build(FakeEmbedder())
        with tempfile.TemporaryDirectory() as tmp:
            index.save(Path(tmp))
            lines = (Path(tmp) / "chunks.jsonl").read_text().splitlines()
            (Path(tmp) / "chunks.jsonl").write_text("\n".join(lines[:-1]) + "\n")  # one chunk lost
            with self.assertRaises(IngestError):
                RepoIndex.load(Path(tmp))


class SearchTests(IndexCase):
    def setUp(self) -> None:
        self.embedder = FakeEmbedder()
        self.index = self.build(self.embedder)
        self.searcher = HybridSearcher(self.index, self.embedder, self.cfg)

    def top(self, query: str, **kw):
        return self.searcher.search(query, **kw)

    def test_exact_identifier_finds_the_definition(self) -> None:
        hits = self.top("verify_password")
        self.assertEqual(hits[0].chunk.symbol, "verify_password")
        self.assertIn("name match", hits[0].signals)

    def test_camel_case_identifier_matches_by_name(self) -> None:
        self.assertEqual(self.top("sendReceipt", mode="keyword")[0].chunk.symbol, "sendReceipt")

    def test_natural_language_question(self) -> None:
        symbols = [h.chunk.symbol for h in self.top("where is the password checked for a login?")[:3]]
        self.assertIn("verify_password", symbols)
        self.assertIn("login", symbols)

    def test_hybrid_reports_both_ranks(self) -> None:
        hit = self.top("create an invoice for an order")[0]
        self.assertEqual(hit.chunk.symbol, "create_invoice")
        self.assertIsNotNone(hit.vector_rank)
        self.assertIsNotNone(hit.keyword_rank)

    def test_modes(self) -> None:
        vector_only = self.top("invoice", mode="vector")
        self.assertTrue(all(h.keyword_rank is None and h.vector_rank for h in vector_only))
        keyword_only = self.top("invoice", mode="keyword")
        self.assertTrue(all(h.vector_rank is None and h.keyword_rank for h in keyword_only))
        with self.assertRaises(ValueError):
            self.top("x", mode="telepathy")

    def test_file_name_in_question_boosts_that_file(self) -> None:
        hits = self.top("what is in billing.py")
        self.assertEqual(hits[0].chunk.path, "app/billing.py")
        self.assertIn("file match", hits[0].signals)

    def test_test_files_are_down_weighted_unless_asked_for(self) -> None:
        plain = self.top("login checks password", k=10)
        test_hits = [h for h in plain if h.chunk.is_test]
        self.assertTrue(test_hits)
        self.assertTrue(all("test penalty" in h.signals for h in test_hits))
        ranked_paths = [h.chunk.path for h in plain]
        self.assertNotEqual(ranked_paths[0], "tests/test_auth.py")
        asked = self.top("tests for login", k=10)
        self.assertTrue(all("test penalty" not in h.signals for h in asked))

    def test_penalty_can_be_disabled(self) -> None:
        cfg = Config(test_file_penalty=1.0, embed_batch_size=4)
        hits = HybridSearcher(self.index, self.embedder, cfg).search("login checks password", k=10)
        self.assertTrue(all("test penalty" not in h.signals for h in hits))

    def test_k_and_candidates(self) -> None:
        self.assertEqual(len(self.top("login", k=2)), 2)
        self.assertLessEqual(len(self.top("login", k=500)), len(self.index.chunks))

    def test_input_validation(self) -> None:
        for bad in ("", "   "):
            with self.assertRaises(ValueError):
                self.top(bad)
        with self.assertRaises(ValueError):
            self.top("login", k=0)

    def test_query_with_no_known_words_returns_vector_results_only(self) -> None:
        hits = self.top("zzzqqq")
        self.assertTrue(all(h.keyword_rank is None for h in hits))

    def test_keyword_only_index_degrades_with_a_note(self) -> None:
        searcher = HybridSearcher(self.build(None), None, self.cfg)
        hits = searcher.search("verify_password")
        self.assertEqual(hits[0].chunk.symbol, "verify_password")
        self.assertIn("keyword search only", searcher.note)
        with self.assertRaises(ValueError):
            searcher.search("verify_password", mode="vector")

    def test_query_prefix_is_applied(self) -> None:
        captured = []

        class Spy(FakeEmbedder):
            def embed(self, texts):
                captured.extend(texts)
                return super().embed(texts)

        HybridSearcher(self.index, Spy(), self.cfg).search("login")
        self.assertEqual(captured, ["search_query: login"])

    def test_embedder_errors_propagate(self) -> None:
        searcher = HybridSearcher(self.index, FakeEmbedder(fail_after=0), self.cfg)
        with self.assertRaises(InvalidOutputError):
            searcher.search("login")

    def test_query_from_a_different_model_is_detected(self) -> None:
        searcher = HybridSearcher(self.index, FakeEmbedder(dim=8), self.cfg)
        with self.assertRaises(InvalidOutputError) as ctx:
            searcher.search("login")
        self.assertIn("different embedding model", str(ctx.exception))


class ConfigTests(unittest.TestCase):
    def test_validation(self) -> None:
        for bad in ({"index_timeout_s": 0}, {"search_top_k": 0}, {"search_candidates": 0},
                    {"test_file_penalty": 0}, {"test_file_penalty": 1.5}):
            with self.assertRaises(ConfigError, msg=str(bad)):
                Config(**bad)

    def test_defaults_suit_nomic_embed_text(self) -> None:
        cfg = Config()
        self.assertEqual((cfg.embed_doc_prefix, cfg.embed_query_prefix), ("search_document: ", "search_query: "))


class CliTests(IndexCase):
    def run_cli(self, *args: str):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_build_then_search_keyword_only(self) -> None:
        env_dir = str(self.base / "cli_data")
        import os

        os.environ["CHATBOT_DATA_DIR"] = env_dir
        try:
            code, out, err = self.run_cli("build", self.result.root, "--keyword-only")
            self.assertEqual(code, 0, err)
            self.assertIn("keyword-only index", out)
            index_dir = Path(env_dir) / "index" / "demo"
            self.assertTrue((index_dir / "index.json").exists())
            code, out, err = self.run_cli("search", str(index_dir), "verify_password", "--mode", "keyword", "--show-text")
            self.assertEqual(code, 0, err)
            self.assertIn("verify_password", out)
            self.assertIn("app/auth.py:", out)
            self.assertIn("| def verify_password", out)
        finally:
            os.environ.pop("CHATBOT_DATA_DIR", None)

    def test_errors(self) -> None:
        code, _, err = self.run_cli("search", str(self.base / "nothing-here"), "x")
        self.assertEqual(code, 1)
        self.assertIn("ERROR", err)
        code, _, err = self.run_cli("build", "definitely-not-a-folder-or-url", "--keyword-only")
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
