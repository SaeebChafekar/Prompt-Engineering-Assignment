"""Tests for retrieval, answering and the chat CLI, using fake embedding and chat backends.

    python -m unittest tests.test_rag -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from common.config import Config
from common.errors import IngestError, InvalidOutputError, LLMError
from repo_qa.graph import build_graph
from repo_qa.index import HybridSearcher, RepoIndex, build_index
from repo_qa.ingest.pipeline import ingest_directory
from repo_qa.rag import Answerer, ContextItem, Retriever, estimate_tokens, verify_citations
from repo_qa.rag.__main__ import main as chat_main
from repo_qa.rag.prompts import NOT_FOUND_TEXT, REWRITE_SYSTEM_PROMPT, SYSTEM_PROMPT
from repo_qa.rag.retrieve import TRUNCATION_NOTE
from tests.fakes import FakeChat
from tests.test_index import FakeEmbedder

BIG_BODY = "\n".join(f"    step_{i} = compute_value({i}) + offset_{i}" for i in range(120))
FILES = {
    "app/__init__.py": "VERSION = 1\n",
    "app/auth.py": '''"""Authentication."""
from flask_login import login_required

from app.models import User


def login(request):
    """Log a user in with a password."""
    user = find_user(request)
    return verify_password(user, request)


def verify_password(user, password):
    """Check the password hash for the user."""
    return user.check(password)


def find_user(request):
    return User(request)


@login_required
def profile(request):
    """Show the profile page."""
    return find_user(request)
''',
    "app/models.py": '''class Base:
    def save(self):
        return 1


class User(Base):
    """A registered user."""

    def __init__(self, name):
        self.name = name

    def check(self, password):
        return password == "x"
''',
    "app/billing.py": f'''def create_invoice(order):
    """Create an invoice for an order."""
    return {{"total": 1}}


def big_report(rows):
    """A very long function."""
{BIG_BODY}
    return rows
''',
    "tests/test_auth.py": "from app.auth import login\n\n\ndef test_login():\n    assert login(None) is None\n",
    "README.md": "# Demo\n\nAuthentication lives in `app/auth.py`: the `login` function checks the password.\n",
}


class RagCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls._tmp.name)
        root = cls.base / "demo"
        for rel, text in FILES.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        cls.cfg = Config(data_dir=str(cls.base / "data"), embed_batch_size=8, search_top_k=3)
        cls.result = ingest_directory(root, cls.cfg)
        cls.graph, _ = build_graph(cls.result)
        cls.embedder = FakeEmbedder()
        cls.index = build_index(cls.result, cls.graph, cls.embedder, cls.cfg)
        cls.keyword_index = build_index(cls.result, cls.graph, None, cls.cfg)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def retriever(self, index: RepoIndex | None = None, cfg: Config | None = None) -> Retriever:
        index = index or self.index
        cfg = cfg or self.cfg
        embedder = self.embedder if index.has_vectors else None
        return Retriever(index, HybridSearcher(index, embedder, cfg), cfg)


class RetrievalTests(RagCase):
    def test_seeds_first_then_graph_neighbours_with_reasons(self) -> None:
        r = self.retriever().retrieve("login", k=1)
        self.assertEqual(r.items[0].chunk.symbol, "login")
        self.assertEqual(r.items[0].origin, "search")
        by_symbol = {i.chunk.symbol: i for i in r.items if i.origin == "graph"}
        self.assertIn("verify_password", by_symbol)
        self.assertIn("find_user", by_symbol)
        self.assertEqual(by_symbol["verify_password"].reason, "called by app/auth.py::login (certain)")
        self.assertTrue(all(i.score < r.items[0].score for i in by_symbol.values()))

    def test_callers_are_found_too_and_tests_rank_lower(self) -> None:
        r = self.retriever().retrieve("verify_password", k=1)
        symbols = {i.chunk.symbol: i for i in r.items}
        self.assertIn("login", symbols)  # login calls verify_password
        self.assertEqual(symbols["login"].reason, "calls app/auth.py::verify_password (certain)")
        retriever = self.retriever()
        seed = retriever.retrieve("login", k=1, use_graph=False).items[0]
        found = retriever._expand("login", [seed], hops=1)  # all neighbours, before the item cap
        test_item = next(i for i in found.values() if i.chunk.is_test)
        normal_item = next(i for i in found.values() if i.chunk.symbol == "verify_password")
        self.assertLess(test_item.score, normal_item.score)  # same relation strength, but a test file
        self.assertAlmostEqual(test_item.score, normal_item.score * self.cfg.test_file_penalty)
        asking = retriever._expand("tests for login", [seed], hops=1)
        asked_item = next(i for i in asking.values() if i.chunk.is_test)
        self.assertAlmostEqual(asked_item.score, normal_item.score)  # no penalty when the question is about tests

    def test_decorator_and_inheritance_relations(self) -> None:
        r = self.retriever().retrieve("User", k=1)
        reasons = {i.chunk.symbol: i.reason for i in r.items if i.origin == "graph"}
        self.assertEqual(reasons.get("Base"), "base class of app/models.py::User (certain)")

    def test_graph_can_be_turned_off(self) -> None:
        r = self.retriever().retrieve("login", k=1, use_graph=False)
        self.assertEqual([i.chunk.symbol for i in r.items], ["login"])
        self.assertEqual(r.facts, [])
        self.assertFalse(r.used_graph)

    def test_no_duplicate_chunks(self) -> None:
        r = self.retriever().retrieve("login password user", k=5)
        ids = [i.chunk.id for i in r.items]
        self.assertEqual(len(ids), len(set(ids)))

    def test_facts_describe_relationships(self) -> None:
        r = self.retriever().retrieve("login", k=1)
        login_fact = next(f for f in r.facts if f.startswith("app/auth.py::login [function]"))
        self.assertIn("calls app/auth.py::find_user (certain), app/auth.py::verify_password (certain)", login_fact)
        self.assertIn("called by tests/test_auth.py::test_login (certain)", login_fact)

    def test_facts_show_decorators_libraries_and_inheritance(self) -> None:
        facts = self.retriever()._facts("what does the view need", [])
        self.assertEqual(facts, [])  # nothing named in the question, no seeds
        facts = self.retriever()._facts("profile", [])
        self.assertTrue(any("decorators @login_required" in f for f in facts))
        user_facts = self.retriever()._facts("User", [])
        self.assertTrue(any("inherits app/models.py::Base (certain)" in f for f in user_facts))

    def test_facts_for_symbols_named_in_the_question_even_when_not_seeds(self) -> None:
        facts = self.retriever()._facts("how does verify_password work", [])
        self.assertTrue(any(f.startswith("app/auth.py::verify_password") for f in facts))

    def test_file_facts_list_imports(self) -> None:
        facts = self.retriever()._facts("what does app/auth.py import", [])
        file_fact = next(f for f in facts if f.startswith("app/auth.py [file]"))
        self.assertIn("flask_login", file_fact)
        self.assertIn("app/models.py", file_fact)

    def test_budget_is_respected_and_overflow_is_counted(self) -> None:
        r = self.retriever().retrieve("login", k=1, budget_tokens=130)
        self.assertLessEqual(r.tokens, 130)
        self.assertGreater(r.dropped, 0)
        self.assertEqual(r.items[0].chunk.symbol, "login")  # the direct hit always survives first

    def test_oversized_best_hit_is_clipped_not_dropped(self) -> None:
        r = self.retriever().retrieve("big_report", k=1, budget_tokens=150, use_graph=False)
        self.assertEqual(len(r.items), 1)
        self.assertIn(TRUNCATION_NOTE.strip(), r.items[0].chunk.text)
        self.assertLessEqual(len(r.items[0].chunk.text), 150 * 4)

    def test_default_budget_comes_from_the_backend_profile(self) -> None:
        self.assertEqual(self.retriever().retrieve("login").budget, self.cfg.retrieval_profile.context_tokens)

    def test_graph_neighbours_are_capped_by_hops_in_config(self) -> None:
        r = self.retriever().retrieve("login", k=1)
        self.assertTrue(all(i.origin == "search" or "(" in i.reason for i in r.items))

    def test_keyword_only_index_carries_the_note(self) -> None:
        r = self.retriever(self.keyword_index).retrieve("verify_password", k=2)
        self.assertIn("keyword search only", r.note or "")
        self.assertEqual(r.items[0].chunk.symbol, "verify_password")

    def test_no_match_gives_empty_retrieval(self) -> None:
        r = self.retriever(self.keyword_index).retrieve("zzzqqqxxx")
        self.assertEqual((r.items, r.facts, r.tokens), ([], [], 0))

    def test_validation(self) -> None:
        retriever = self.retriever()
        with self.assertRaises(ValueError):
            retriever.retrieve("   ")
        with self.assertRaises(ValueError):
            retriever.retrieve("login", budget_tokens=10)

    def test_estimate_tokens(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertLess(estimate_tokens("abc"), estimate_tokens("abc" * 100))


class CitationTests(RagCase):
    def items(self) -> list[ContextItem]:
        return self.retriever().retrieve("login", k=2).items

    def test_valid_and_invalid_citations(self) -> None:
        items = self.items()
        login = next(i.chunk for i in items if i.chunk.symbol == "login")
        text = f"See [{login.path}:{login.start_line}-{login.end_line}] and [app/ghost.py:1-5]."
        cited, invalid = verify_citations(text, items)
        self.assertEqual(len(cited), 2)
        self.assertEqual(invalid, ["app/ghost.py:1-5"])

    def test_overlapping_range_and_single_line_are_accepted(self) -> None:
        items = self.items()
        login = next(i.chunk for i in items if i.chunk.symbol == "login")
        for text in (f"[{login.path}:{login.start_line + 1}]", f"[{login.path}:1-{login.start_line}]",
                     f"[{login.path}:{login.end_line}\u2013{login.end_line + 30}]"):
            self.assertEqual(verify_citations(text, items)[1], [], text)

    def test_right_file_wrong_lines_is_invalid(self) -> None:
        self.assertEqual(verify_citations("[app/auth.py:900-950]", self.items())[1], ["app/auth.py:900-950"])

    def test_path_suffix_matches(self) -> None:
        items = self.items()
        login = next(i.chunk for i in items if i.chunk.symbol == "login")
        self.assertEqual(verify_citations(f"[auth.py:{login.start_line}]", items)[1], [])

    def test_duplicates_and_non_citations(self) -> None:
        items = self.items()
        login = next(i.chunk for i in items if i.chunk.symbol == "login")
        label = f"{login.path}:{login.start_line}-{login.end_line}"
        cited, _ = verify_citations(f"[{label}] and again [{label}], plus example.com:8080 and v1.2:3", items)
        self.assertEqual(cited, [label])  # deduplicated, and URLs/versions are not citations
        self.assertEqual(verify_citations("no citations here", items), ([], []))

    def test_multiple_citations_in_one_bracket(self) -> None:
        cited, invalid = verify_citations("[app/auth.py:7-12, app/zzz.py:3]", self.items())
        self.assertEqual(len(cited), 2)
        self.assertEqual(invalid, ["app/zzz.py:3"])


class AnswererTests(RagCase):
    def answerer(self, chat: FakeChat, index: RepoIndex | None = None) -> Answerer:
        return Answerer(self.retriever(index), chat, self.cfg)

    def test_prompt_contains_rules_context_facts_and_question(self) -> None:
        chat = FakeChat()
        self.answerer(chat).ask("Where is login handled?")
        call = chat.calls[0]
        self.assertEqual(call["system"], SYSTEM_PROMPT)
        user = call["messages"][0]["content"]
        self.assertEqual(call["messages"][0]["role"], "user")
        self.assertIn("Repository: demo", user)
        self.assertIn("Relationship facts (from the code graph):", user)
        self.assertIn("### [app/auth.py:", user)
        self.assertIn("matched the question", user)
        self.assertTrue(user.rstrip().endswith("Question: Where is login handled?"))
        self.assertIn(NOT_FOUND_TEXT, SYSTEM_PROMPT)

    def test_answer_object(self) -> None:
        chat = FakeChat(["Login is in [app/auth.py:7-10] and [app/ghost.py:1-2]."])
        answer = self.answerer(chat).ask("Where is login handled?")
        self.assertEqual(answer.text, "Login is in [app/auth.py:7-10] and [app/ghost.py:1-2].")
        self.assertEqual(answer.question, answer.standalone_question)
        self.assertEqual(answer.invalid_citations, ["app/ghost.py:1-2"])
        self.assertFalse(answer.citations_ok)
        self.assertFalse(answer.not_found)
        self.assertTrue(answer.retrieval.items)

    def test_effort_level_controls_prompt_and_tokens(self) -> None:
        chat = FakeChat(["Detailed explanation."])
        answer = self.answerer(chat).ask("Where is login handled?", effort="high")
        self.assertEqual(answer.effort, "high")
        self.assertIn("Deep code walkthrough", chat.calls[0]["system"])
        self.assertEqual(chat.calls[0]["kw"]["max_tokens"], 4096)

    def test_valid_citations_pass(self) -> None:
        answer = self.answerer(FakeChat(["Handled in [app/auth.py:7-10]."])).ask("Where is login handled?")
        self.assertEqual(answer.invalid_citations, [])
        self.assertTrue(answer.citations_ok)

    def test_not_found_reply_is_detected(self) -> None:
        answer = self.answerer(FakeChat([NOT_FOUND_TEXT])).ask("What colour is the sky?")
        self.assertTrue(answer.not_found)
        self.assertEqual(answer.cited, [])

    def test_streaming_callback_receives_pieces(self) -> None:
        pieces: list[str] = []
        answer = self.answerer(FakeChat(["one two three"])).ask("login?", on_token=pieces.append)
        self.assertEqual("".join(pieces), "one two three")
        self.assertGreater(len(pieces), 1)
        self.assertEqual(answer.text, "one two three")

    def test_graph_flag_reaches_retrieval(self) -> None:
        self.assertFalse(self.answerer(FakeChat()).ask("login", use_graph=False).retrieval.used_graph)
        self.assertTrue(self.answerer(FakeChat()).ask("login").retrieval.used_graph)

    def test_no_history_means_no_rewrite_call(self) -> None:
        chat = FakeChat()
        self.answerer(chat).ask("login")
        self.assertEqual([c["kind"] for c in chat.calls], ["stream"])

    def test_follow_up_is_rewritten_and_retrieval_uses_the_rewrite(self) -> None:
        chat = FakeChat(["What does verify_password do?", "It checks the hash [app/auth.py:13-16]."])
        history = [("Where is login handled?", "In the login function [app/auth.py:7-10].")]
        answer = self.answerer(chat).ask("and what does it call?", history)
        self.assertEqual(chat.calls[0]["system"], REWRITE_SYSTEM_PROMPT)
        self.assertIn("Where is login handled?", chat.calls[0]["messages"][0]["content"])
        self.assertIn("Follow-up question: and what does it call?", chat.calls[0]["messages"][0]["content"])
        self.assertEqual(answer.question, "and what does it call?")
        self.assertEqual(answer.standalone_question, "What does verify_password do?")
        self.assertEqual(answer.retrieval.query, "What does verify_password do?")
        self.assertTrue(chat.calls[1]["messages"][0]["content"].rstrip().endswith("Question: What does verify_password do?"))

    def test_rewrite_failure_falls_back_to_the_original(self) -> None:
        chat = FakeChat(["unused", "Answer [app/auth.py:7-10]."], fail_on={1})
        answer = self.answerer(chat).ask("and that one?", [("q", "a")])
        self.assertEqual(answer.standalone_question, "and that one?")
        self.assertEqual(answer.text, "Answer [app/auth.py:7-10].")

    def test_unusable_rewrites_are_ignored(self) -> None:
        for bad in ("", "line one\nline two", "x" * 500):
            chat = FakeChat([bad, "Answer."])
            answer = self.answerer(chat).ask("and that one?", [("q", "a")])
            self.assertEqual(answer.standalone_question, "and that one?", repr(bad[:20]))

    def test_rewrite_strips_quotes(self) -> None:
        chat = FakeChat(['"What does login do?"', "Answer."])
        self.assertEqual(self.answerer(chat).ask("what does it do?", [("q", "a")]).standalone_question, "What does login do?")

    def test_input_validation(self) -> None:
        answerer = self.answerer(FakeChat())
        with self.assertRaises(ValueError):
            answerer.ask("   ")
        with self.assertRaises(ValueError):
            answerer.ask("x" * 2001)

    def test_model_failures_propagate(self) -> None:
        with self.assertRaises(LLMError):
            self.answerer(FakeChat(fail_on={1})).ask("login")

    def test_empty_model_output_is_an_error(self) -> None:
        with self.assertRaises(InvalidOutputError):
            self.answerer(FakeChat(["   "])).ask("login")

    def test_works_on_a_keyword_only_index(self) -> None:
        answer = self.answerer(FakeChat(), self.keyword_index).ask("verify_password")
        self.assertEqual(answer.retrieval.items[0].chunk.symbol, "verify_password")


class ChatCliTests(RagCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.index_dir = Path(self._dir.name) / "index"
        self.index.save(self.index_dir)
        self.out: list[str] = []
        self.chat = FakeChat()
        self.embedder_instance = FakeEmbedder()

    def run_cli(self, *args: str, inputs: list[str] | None = None, chat: FakeChat | None = None, embedder=None) -> int:
        feed = iter(inputs or [])

        def input_fn(_prompt: str) -> str:
            try:
                return next(feed)
            except StopIteration:
                raise EOFError

        return chat_main(
            [str(self.index_dir), *args],
            chat_factory=lambda cfg: chat or self.chat,
            embedder_factory=lambda cfg: embedder or self.embedder_instance,
            input_fn=input_fn,
            write=self.out.append,
        )

    @property
    def output(self) -> str:
        return "".join(self.out)

    def test_single_question(self) -> None:
        code = self.run_cli("--ask", "Where is login handled?")
        self.assertEqual(code, 0)
        self.assertIn("It is handled in [app/auth.py:7-10].", self.output)
        self.assertIn("Sources used:", self.output)
        self.assertIn("[search] app/auth.py:", self.output)
        self.assertIn("[graph]", self.output)
        self.assertTrue(self.chat.closed)

    def test_no_graph_flag(self) -> None:
        self.run_cli("--ask", "login", "--no-graph")
        self.assertNotIn("[graph]", self.output)

    def test_invalid_citations_are_called_out(self) -> None:
        self.run_cli("--ask", "login", chat=FakeChat(["See [app/ghost.py:1-2]."]))
        self.assertIn("WARNING", self.output)
        self.assertIn("app/ghost.py:1-2", self.output)

    def test_show_context(self) -> None:
        self.run_cli("--ask", "login", "--show-context")
        self.assertIn("--- context given to the model ---", self.output)
        self.assertIn("Relationship facts", self.output)

    def test_interactive_session_with_commands_and_history(self) -> None:
        chat = FakeChat(["First answer [app/auth.py:7-10].", "What does login call?", "Second answer."])
        code = self.run_cli(inputs=[
            "", "/help", "Where is login handled?", "/sources", "and what does it call?",
            "/graph off", "/mode keyword", "/bogus", "/context", "/clear", "/quit",
        ], chat=chat)
        out = self.output
        self.assertEqual(code, 0)
        self.assertIn("Commands:", out)
        self.assertIn("First answer", out)
        self.assertIn("(understood as: What does login call?)", out)
        self.assertIn("Second answer.", out)
        self.assertIn("Graph expansion off.", out)
        self.assertIn("Search mode: keyword.", out)
        self.assertIn("Unknown or incomplete command", out)
        self.assertIn("Conversation cleared.", out)
        self.assertEqual([c["kind"] for c in chat.calls], ["stream", "chat", "stream"])  # rewrite only for the follow-up

    def test_errors_in_a_question_do_not_end_the_session(self) -> None:
        chat = FakeChat(["unused", "Recovered [app/auth.py:7-10]."], fail_on={1})
        code = self.run_cli(inputs=["login?", "login again?"], chat=chat)
        self.assertEqual(code, 0)
        self.assertIn("Error (LLMError)", self.output)
        self.assertIn("Recovered", self.output)

    def test_sources_command_before_any_question(self) -> None:
        self.run_cli(inputs=["/sources"])
        self.assertIn("Nothing yet", self.output)

    def test_end_of_input_exits_cleanly(self) -> None:
        self.assertEqual(self.run_cli(inputs=[]), 0)

    def test_missing_index_is_a_clean_error(self) -> None:
        code = chat_main([str(Path(self._dir.name) / "nope"), "--ask", "x"], chat_factory=lambda c: self.chat,
                         embedder_factory=lambda c: self.embedder_instance, write=self.out.append)
        self.assertEqual(code, 1)

    def test_embedding_model_mismatch_is_detected(self) -> None:
        class Other(FakeEmbedder):
            model_name = "some-other-model"

        code = self.run_cli("--ask", "login", embedder=Other())
        self.assertEqual(code, 1)

    def test_keyword_only_index_prints_a_note(self) -> None:
        self.keyword_index.save(self.index_dir)
        code = self.run_cli("--ask", "verify_password")
        self.assertEqual(code, 0)
        self.assertIn("only keyword search is used", self.output)
        self.assertIn("It is handled in", self.output)

    def test_empty_question_via_ask(self) -> None:
        self.assertEqual(self.run_cli("--ask", "   "), 1)

    def test_load_error_type(self) -> None:
        with self.assertRaises(IngestError):
            RepoIndex.load(Path(self._dir.name) / "nope")


if __name__ == "__main__":
    unittest.main()
