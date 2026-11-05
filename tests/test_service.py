"""Tests for repo_qa.service: listing indexes, indexing a repository, and opening a chat session.

    python -m unittest tests.test_service -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from common.config import Config
from common.errors import CloneError, IngestError, LLMError
from repo_qa.service import MAX_HISTORY, IndexInfo, RepoChat, index_repository, list_indexes
from tests.fakes import FakeChat
from tests.test_index import FakeEmbedder
from tests.test_rag import FILES


class TrackedEmbedder(FakeEmbedder):
    def __init__(self, *args, fail_health: bool = False, **kw) -> None:
        super().__init__(*args, **kw)
        self.closed = False
        self.fail_health = fail_health

    def health_check(self) -> None:
        if self.fail_health:
            raise LLMError("embedding model missing")

    def close(self) -> None:
        self.closed = True


class ServiceCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.root = self.base / "demo"
        for rel, text in FILES.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.cfg = Config(data_dir=str(self.base / "data"), embed_batch_size=8)

    def index(self, **kw):
        embedders: list[TrackedEmbedder] = []

        def factory(cfg):
            embedder = TrackedEmbedder()
            embedders.append(embedder)
            return embedder

        report = index_repository(str(self.root), self.cfg, embedder_factory=factory, **kw)
        return report, embedders


class ListIndexesTests(ServiceCase):
    def test_empty_and_missing_folder(self) -> None:
        self.assertEqual(list_indexes(self.cfg), [])

    def test_lists_valid_indexes_and_skips_broken_ones(self) -> None:
        report, _ = self.index()
        (Path(self.cfg.data_dir) / "index" / "junk").mkdir()
        broken = Path(self.cfg.data_dir) / "index" / "broken"
        broken.mkdir()
        (broken / "index.json").write_text("{not json")
        (Path(self.cfg.data_dir) / "index" / "a_file.txt").write_text("ignore me")
        infos = list_indexes(self.cfg)
        self.assertEqual([i.name for i in infos], ["demo"])
        info = infos[0]
        self.assertIsInstance(info, IndexInfo)
        self.assertEqual((info.repo, info.chunks, info.has_vectors, info.embed_model),
                         ("demo", report.chunks, True, "fake-embed"))
        self.assertEqual(info.path, report.path)


class IndexRepositoryTests(ServiceCase):
    def test_full_index_with_progress_and_stages(self) -> None:
        stages, progress = [], []
        report, embedders = self.index(stage=stages.append, progress=lambda d, t: progress.append((d, t)))
        self.assertEqual(report.repo_name, "demo")
        self.assertTrue(report.has_vectors)
        self.assertEqual(report.vectors_embedded, report.chunks)
        self.assertEqual(report.path, Path(self.cfg.data_dir) / "index" / "demo")
        self.assertTrue((report.path / "index.json").exists())
        self.assertIn("Repository : demo", report.summary)
        self.assertGreater(sum(report.graph_counts["nodes"].values()), 5)
        self.assertIn("certain", report.call_stats)
        self.assertEqual(progress[-1], (report.chunks, report.chunks))
        self.assertEqual(stages[0], "Reading the repository ...")
        self.assertTrue(any(s.startswith("Embedding") for s in stages))
        self.assertEqual(stages[-1], "Saving the index ...")
        self.assertTrue(embedders[0].closed)

    def test_keyword_only_creates_no_embedder(self) -> None:
        report, embedders = self.index(keyword_only=True)
        self.assertFalse(report.has_vectors)
        self.assertEqual(embedders, [])

    def test_second_run_reuses_vectors(self) -> None:
        self.index()
        report, _ = self.index()
        self.assertEqual((report.vectors_embedded, report.vectors_reused), (0, report.chunks))

    def test_unhealthy_embedder_fails_early_and_is_closed(self) -> None:
        embedder = TrackedEmbedder(fail_health=True)
        with self.assertRaises(LLMError):
            index_repository(str(self.root), self.cfg, embedder_factory=lambda cfg: embedder)
        self.assertTrue(embedder.closed)
        self.assertEqual(list_indexes(self.cfg), [])  # nothing half-written

    def test_embedding_failure_closes_the_embedder(self) -> None:
        embedder = TrackedEmbedder(fail_after=0)
        with self.assertRaises(Exception):
            index_repository(str(self.root), self.cfg, embedder_factory=lambda cfg: embedder)
        self.assertTrue(embedder.closed)

    def test_bad_sources_raise_clear_errors(self) -> None:
        with self.assertRaises(IngestError):
            index_repository("definitely-not-a-folder-or-url", self.cfg)
        with self.assertRaises(CloneError):
            index_repository("https://github.com/only-owner", self.cfg)

    def test_custom_output_folder_and_warnings(self) -> None:
        (self.root / "app" / "bad.py").write_text("def broken(:\n    pass\n")
        out = self.base / "custom"
        report, _ = self.index(keyword_only=True, out_dir=out)
        self.assertEqual(report.path, out)
        self.assertTrue(any("could not be parsed" in w for w in report.warnings))


class RepoChatTests(ServiceCase):
    def open(self, chat: FakeChat | None = None, embedder: TrackedEmbedder | None = None) -> tuple[RepoChat, FakeChat, TrackedEmbedder]:
        report, _ = self.index()
        chat = chat or FakeChat(["It is handled in [app/auth.py:7-10]."])
        embedder = embedder or TrackedEmbedder()
        session = RepoChat.open(report.path, self.cfg, chat_factory=lambda cfg: chat, embedder_factory=lambda cfg: embedder)
        return session, chat, embedder

    def test_ask_records_history_and_clear_resets_it(self) -> None:
        session, chat, _ = self.open()
        answer = session.ask("Where is login handled?")
        self.assertIn("app/auth.py", answer.text)
        self.assertEqual(session.history, [("Where is login handled?", answer.text)])
        session.clear()
        self.assertEqual(session.history, [])

    def test_history_is_capped_and_used_for_follow_ups(self) -> None:
        chat = FakeChat(["Standalone question?", "Answer [app/auth.py:7-10]."])
        session, chat, _ = self.open(chat)
        session.history.append(("first question", "first answer"))
        answer = session.ask("and then?")
        self.assertEqual(answer.standalone_question, "Standalone question?")
        for i in range(MAX_HISTORY + 4):
            session.ask(f"question {i}")
        self.assertEqual(len(session.history), MAX_HISTORY)

    def test_options_reach_the_answerer(self) -> None:
        session, _, _ = self.open()
        answer = session.ask("login", use_graph=False, mode="keyword")
        self.assertFalse(answer.retrieval.used_graph)
        self.assertTrue(all(h.chunk for h in answer.retrieval.items))

    def test_close_releases_both_backends(self) -> None:
        session, chat, embedder = self.open()
        session.close()
        self.assertTrue(chat.closed and embedder.closed)

    def test_keyword_only_index_needs_no_embedder(self) -> None:
        report, _ = self.index(keyword_only=True)
        called = []
        session = RepoChat.open(report.path, self.cfg, chat_factory=lambda cfg: FakeChat(),
                                embedder_factory=lambda cfg: called.append(1))
        self.assertEqual(called, [])
        self.assertFalse(session.index.has_vectors)
        session.ask("verify_password")

    def test_missing_index(self) -> None:
        with self.assertRaises(IngestError):
            RepoChat.open(self.base / "nope", self.cfg)

    def test_embedding_model_mismatch_closes_resources(self) -> None:
        report, _ = self.index()

        class Other(TrackedEmbedder):
            model_name = "another-model"

        other = Other()
        with self.assertRaises(IngestError) as ctx:
            RepoChat.open(report.path, self.cfg, chat_factory=lambda cfg: FakeChat(), embedder_factory=lambda cfg: other)
        self.assertIn("Re-index", str(ctx.exception))
        self.assertTrue(other.closed)

    def test_chat_backend_failure_closes_the_embedder(self) -> None:
        report, _ = self.index()

        class Broken(FakeChat):
            def health_check(self) -> None:
                raise LLMError("model server down")

        embedder = TrackedEmbedder()
        with self.assertRaises(LLMError):
            RepoChat.open(report.path, self.cfg, chat_factory=lambda cfg: Broken(), embedder_factory=lambda cfg: embedder)
        self.assertTrue(embedder.closed)

    def test_index_json_is_untouched_by_chatting(self) -> None:
        session, _, _ = self.open()
        before = (session.index.meta.get("chunks"), json.loads((Path(self.cfg.data_dir) / "index" / "demo" / "index.json").read_text())["chunks"])
        session.ask("login")
        self.assertEqual(before[1], len(session.index.chunks))


if __name__ == "__main__":
    unittest.main()
