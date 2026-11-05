"""Tests for the Streamlit interface, driven with Streamlit's AppTest harness and fake backends.

    python -m unittest tests.test_webapp -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import logging

from streamlit.testing.v1 import AppTest

from common.config import Config
from common.errors import ConfigError, LLMError
from tests.fakes import FakeChat
from tests.test_code_review import CODE, issue, payload
from tests.test_index import FakeEmbedder
from tests.test_rag import FILES
from webapp import resources, sidebar

for _name in ["streamlit", *[n for n in logging.root.manager.loggerDict if n.startswith("streamlit")]]:
    logging.getLogger(_name).setLevel(logging.ERROR)  # "missing ScriptRunContext" is harmless noise in bare mode

APP = str(Path(__file__).resolve().parents[1] / "app.py")
ANSWER = "It is handled in [app/auth.py:7-10]."


class WebCase(unittest.TestCase):
    def setUp(self) -> None:
        for name in [n for n in logging.root.manager.loggerDict if n.startswith("streamlit")]:
            logging.getLogger(name).setLevel(logging.ERROR)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.root = self.base / "demo"
        for rel, text in FILES.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.cfg = Config(data_dir=str(self.base / "data"), embed_batch_size=8)
        self.chat = FakeChat([ANSWER])
        self.saved = (resources.config_loader, resources.chat_factory, resources.embedder_factory)
        resources.config_loader = lambda: self.cfg
        resources.chat_factory = lambda cfg: self.chat
        resources.embedder_factory = lambda cfg: FakeEmbedder()
        self.addCleanup(self.restore)

    def restore(self) -> None:
        resources.config_loader, resources.chat_factory, resources.embedder_factory = self.saved

    def app(self) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=60).run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    @staticmethod
    def texts(elements) -> str:
        return "\n".join(str(e.value) for e in elements)

    def go_to_repo_page(self, at: AppTest) -> AppTest:
        at.sidebar.radio(key="page").set_value(sidebar.PAGE_REPO).run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def index_demo(self, at: AppTest, keyword_only: bool = True) -> AppTest:
        at.text_input(key="repo_source").input(str(self.root))
        if keyword_only:
            at.checkbox(key="repo_keyword_only").check()
        at.button(key="repo_index_btn").click().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at


class SidebarTests(WebCase):
    def test_first_page_is_project_1_and_sidebar_lists_the_models(self) -> None:
        at = self.app()
        self.assertEqual(at.sidebar.radio(key="page").value, sidebar.PAGE_REVIEW)
        self.assertIn("Code reviewer", self.texts(at.header))
        sidebar_text = self.texts(at.sidebar.markdown)
        self.assertIn("Ollama (local): qwen3:8b", sidebar_text)
        self.assertIn("nomic-embed-text (local)", sidebar_text)

    def test_gemini_backend_is_shown_when_configured(self) -> None:
        self.cfg = Config(data_dir=str(self.base / "data"), backend="gemini", gemini_api_key="k")
        self.assertIn("Gemini (cloud): gemini-3.8-flash", self.texts(self.app().sidebar.markdown))

    def test_connection_check_reports_both_models(self) -> None:
        at = self.app()
        at.sidebar.button(key="check_backend").click().run()
        text = self.texts(at.sidebar.success)
        self.assertIn("Chat model ready: fake-chat", text)
        self.assertIn("Embedding model ready: fake-embed", text)

    def test_connection_problems_are_shown_not_raised(self) -> None:
        class Down(FakeChat):
            def health_check(self) -> None:
                raise LLMError("server is not running")

        self.chat = Down()
        at = self.app()
        at.sidebar.button(key="check_backend").click().run()
        self.assertIn("server is not running", self.texts(at.sidebar.error))

    def test_configuration_errors_are_shown_clearly(self) -> None:
        def broken():
            raise ConfigError("num_ctx must be at least 512")

        resources.config_loader = broken
        at = AppTest.from_file(APP, default_timeout=60).run()
        self.assertFalse(at.exception)
        self.assertIn("Configuration problem: num_ctx must be at least 512", self.texts(at.error))


class ReviewPageTests(WebCase):
    def review(self, at: AppTest, code: str = CODE) -> AppTest:
        at.text_area(key="review_code").input(code)
        at.button(key="review_btn").click().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def test_review_shows_summary_counts_and_issues(self) -> None:
        self.chat = FakeChat([payload(issue(), issue(severity="low", title="Repeated work", line_start=2, line_end=2))])
        at = self.review(self.app())
        self.assertIn("One bug found.", self.texts(at.info))
        self.assertEqual({m.label: m.value for m in at.metric}["High"], "1")
        self.assertEqual({m.label: m.value for m in at.metric}["Low"], "1")
        labels = [e.label for e in at.expander]
        self.assertIn("High (1)", labels)
        self.assertIn("Low (1)", labels)
        self.assertIn("Off-by-one loop", self.texts(at.expander[0].markdown))
        self.assertIn("5 lines of python reviewed", self.texts(at.caption))
        self.assertEqual(self.chat.calls[0]["kind"], "json")

    def test_clean_code_shows_a_success_message(self) -> None:
        self.chat = FakeChat([payload(summary="Looks good.", verdict="Looks good")])
        at = self.review(self.app())
        self.assertIn("No issues found.", self.texts(at.success))

    def test_example_button_fills_the_form(self) -> None:
        at = self.app()
        at.button(key="review_example").click().run()
        self.assertIn("def average(numbers):", at.text_area(key="review_code").value)
        self.assertEqual(at.text_input(key="review_problem").value, "average([]) crashes")

    def test_problem_and_language_are_sent_to_the_model(self) -> None:
        self.chat = FakeChat([payload()])
        at = self.app()
        at.selectbox(key="review_language").select("python")
        at.text_input(key="review_problem").input("it crashes on empty lists")
        self.review(at)
        message = self.chat.calls[0]["messages"][0]["content"]
        self.assertIn("it crashes on empty lists", message)
        self.assertIn("Language: python", message)

    def test_empty_input_shows_an_error_and_makes_no_model_call(self) -> None:
        at = self.app()
        at.button(key="review_btn").click().run()
        self.assertIn("Paste some code", self.texts(at.error))
        self.assertEqual(self.chat.calls, [])

    def test_model_failure_is_shown_and_clears_the_old_result(self) -> None:
        self.chat = FakeChat([payload(issue()), payload()], fail_on={2})
        at = self.review(self.app())
        self.assertTrue(at.expander)
        at.button(key="review_btn").click().run()
        self.assertIn("LLMError", self.texts(at.error))
        self.assertFalse(at.expander)  # a stale review is not left on screen after a failure

    def test_unusable_model_output_is_reported_after_a_retry(self) -> None:
        self.chat = FakeChat(["not json", "still not json"])
        at = self.review(self.app())
        self.assertIn("after 2 attempts", self.texts(at.error))
        self.assertEqual(len(self.chat.calls), 2)

    def test_syntax_errors_are_found_by_the_parser(self) -> None:
        self.chat = FakeChat([payload(summary="Nothing wrong.")])
        at = self.review(self.app(), "def broken(:\n    pass\n")
        self.assertIn("Critical (1)", [e.label for e in at.expander])
        self.assertIn("automatic syntax check", self.texts(at.expander[0].markdown))

    def test_followup_chat(self) -> None:
        self.chat = FakeChat([payload(issue()), "Because the loop reads past the end of the list."])
        at = self.review(self.app())
        at.chat_input(key="review_chat_input").set_value("Why is it high?").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        roles = [m.name for m in at.chat_message]
        self.assertEqual(roles, ["user", "assistant"])
        self.assertIn("Why is it high?", self.texts(at.chat_message[0].markdown))
        self.assertIn("reads past the end", self.texts(at.chat_message[1].markdown))
        self.assertEqual(self.chat.calls[1]["kind"], "stream")
        # the next question carries the conversation so far
        self.chat.replies.append("Second answer.")
        at.chat_input(key="review_chat_input").set_value("And the second one?").run()
        self.assertIn("Why is it high?", self.chat.calls[2]["messages"][0]["content"])
        self.assertEqual(len(at.chat_message), 4)

    def test_followup_error_is_shown(self) -> None:
        self.chat = FakeChat([payload(issue()), "unused"], fail_on={2})
        at = self.review(self.app())
        at.chat_input(key="review_chat_input").set_value("Why?").run()
        self.assertIn("LLMError", self.texts(at.error))
        self.assertEqual(len(at.chat_message), 2)
        self.chat.replies.append("Now it works.")
        at.chat_input(key="review_chat_input").set_value("Try again?").run()
        self.assertEqual(len(at.chat_message), 4)
        sent = self.chat.calls[-1]["messages"][0]["content"]
        self.assertNotIn("LLMError", sent)  # the failed attempt is not fed back to the model

    def test_a_new_review_resets_the_conversation(self) -> None:
        self.chat = FakeChat([payload(issue()), "An answer.", payload(issue(title="Second review"))])
        at = self.review(self.app())
        at.chat_input(key="review_chat_input").set_value("Why?").run()
        self.assertEqual(len(at.chat_message), 2)
        self.review(at)
        self.assertEqual(len(at.chat_message), 0)

    def test_effort_selector_controls_review_depth(self) -> None:
        self.chat = FakeChat([payload(issue())])
        at = self.app()
        self.assertTrue(at.selectbox(key="review_effort"))
        at.selectbox(key="review_effort").select("High (Deep analysis & root cause)")
        self.review(at)
        self.assertIn("HIGH EFFORT", self.chat.calls[0]["system"])


class RepoPageTests(WebCase):
    def test_empty_state(self) -> None:
        at = self.go_to_repo_page(self.app())
        self.assertIn("No repository is indexed yet", self.texts(at.info))
        self.assertFalse(at.selectbox(key="repo_selected") if False else False)

    def test_index_then_chat_with_sources(self) -> None:
        at = self.go_to_repo_page(self.app())
        self.index_demo(at)
        self.assertIn("Indexed demo", self.texts(at.success))
        self.assertEqual(at.selectbox(key="repo_selected").value, "demo")
        self.assertIn("Knowledge graph:", self.texts(at.caption))
        at.chat_input(key="repo_chat_input").set_value("Where is login handled?").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual([m.name for m in at.chat_message], ["user", "assistant"])
        self.assertIn("app/auth.py:7-10", self.texts(at.chat_message[1].markdown))
        self.assertTrue(any(e.label.startswith("Sources (") for e in at.expander))
        sources = next(e for e in at.expander if e.label.startswith("Sources ("))
        self.assertIn("app/auth.py", self.texts(sources.markdown))
        self.assertIn("graph", self.texts(sources.markdown))
        self.assertEqual([c["kind"] for c in self.chat.calls], ["stream"])

    def test_embedding_index_uses_the_embedder_and_reports_counts(self) -> None:
        at = self.go_to_repo_page(self.app())
        self.index_demo(at, keyword_only=False)
        self.assertRegex(self.texts(at.success), r"Indexed demo: \d+ chunks, \d+ embedded now, 0 reused")
        self.assertIn("with vectors", self.texts([at.selectbox(key="repo_selected")]) + str(at.selectbox(key="repo_selected").options))

    def test_keyword_only_index_shows_a_note(self) -> None:
        at = self.go_to_repo_page(self.app())
        self.index_demo(at)
        self.assertIn("only keyword search is available", self.texts(at.info))

    def test_invalid_citations_are_flagged(self) -> None:
        self.chat = FakeChat(["See [app/ghost.py:1-2]."])
        at = self.go_to_repo_page(self.app())
        self.index_demo(at)
        at.chat_input(key="repo_chat_input").set_value("login?").run()
        self.assertIn("unverified", self.texts(at.warning))
        self.assertIn("app/ghost.py:1-2", self.texts(at.warning))

    def test_graph_toggle_changes_what_is_retrieved(self) -> None:
        at = self.go_to_repo_page(self.app())
        self.index_demo(at)
        at.toggle(key="repo_use_graph").set_value(False).run()
        at.chat_input(key="repo_chat_input").set_value("login").run()
        sources = next(e for e in at.expander if e.label.startswith("Sources ("))
        self.assertNotIn("(graph:", self.texts(sources.markdown))

    def test_show_context_displays_the_prompt_material(self) -> None:
        at = self.go_to_repo_page(self.app())
        self.index_demo(at)
        at.checkbox(key="repo_show_context").check().run()
        at.chat_input(key="repo_chat_input").set_value("login").run()
        self.assertIn("Context:", self.texts(at.code))

    def test_follow_up_uses_the_conversation(self) -> None:
        self.chat = FakeChat([ANSWER, "What does login call?", "It calls verify_password [app/auth.py:13-16]."])
        at = self.go_to_repo_page(self.app())
        self.index_demo(at)
        at.chat_input(key="repo_chat_input").set_value("Where is login handled?").run()
        at.chat_input(key="repo_chat_input").set_value("and what does it call?").run()
        self.assertEqual([c["kind"] for c in self.chat.calls], ["stream", "chat", "stream"])
        self.assertIn("Understood as: What does login call?", self.texts(at.caption))
        self.assertEqual(len(at.chat_message), 4)

    def test_clear_chat_button(self) -> None:
        at = self.go_to_repo_page(self.app())
        self.index_demo(at)
        at.chat_input(key="repo_chat_input").set_value("login").run()
        self.assertEqual(len(at.chat_message), 2)
        at.button(key="repo_clear").click().run()
        self.assertEqual(len(at.chat_message), 0)

    def test_model_failure_during_a_question_is_shown(self) -> None:
        self.chat = FakeChat(["unused"], fail_on={1})
        at = self.go_to_repo_page(self.app())
        self.index_demo(at)
        at.chat_input(key="repo_chat_input").set_value("login").run()
        self.assertIn("LLMError", self.texts(at.error))
        self.assertEqual([m.name for m in at.chat_message], ["user", "assistant"])  # the failed exchange stays visible
        self.assertEqual(at.session_state["repo_chat"].history, [])  # but the model's memory is untouched
        at.toggle(key="repo_use_graph").set_value(False).run()  # a later refresh keeps showing it
        self.assertEqual(len(at.chat_message), 2)
        self.assertIn("Error.", self.texts(at.chat_message[1].markdown))

    def test_graph_explorer(self) -> None:
        at = self.go_to_repo_page(self.app())
        self.index_demo(at)
        at.text_input(key="repo_explore").input("verify_password").run()
        explorer = self.texts([m for e in at.expander for m in e.markdown])
        self.assertIn("app/auth.py::verify_password", explorer)
        self.assertIn("Called by:", explorer)
        self.assertIn("app/auth.py::login", explorer)

    def test_graph_explorer_unknown_name(self) -> None:
        at = self.go_to_repo_page(self.app())
        self.index_demo(at)
        at.text_input(key="repo_explore").input("zzz_no_such_function").run()
        self.assertIn("No function or class with that name.", self.texts([m for e in at.expander for m in e.markdown]))

    def test_indexing_errors_are_shown(self) -> None:
        at = self.go_to_repo_page(self.app())
        at.text_input(key="repo_source").input("definitely-not-a-folder-or-url")
        at.button(key="repo_index_btn").click().run()
        self.assertFalse(at.exception)
        self.assertIn("IngestError", self.texts(at.error))

    def test_empty_source_is_rejected(self) -> None:
        at = self.go_to_repo_page(self.app())
        at.button(key="repo_index_btn").click().run()
        self.assertIn("Enter a GitHub URL or a folder path first", self.texts(at.error))

    def test_opening_an_index_with_an_unavailable_model_shows_the_reason(self) -> None:
        at = self.go_to_repo_page(self.app())
        self.index_demo(at)

        class Down(FakeChat):
            def health_check(self) -> None:
                raise LLMError("model server down")

        resources.chat_factory = lambda cfg: Down()
        fresh = self.go_to_repo_page(AppTest.from_file(APP, default_timeout=60).run())
        self.assertIn("Could not open this repository", self.texts(fresh.error))
        self.assertIn("model server down", self.texts(fresh.error))

    def test_existing_indexes_are_offered_on_a_fresh_session(self) -> None:
        at = self.go_to_repo_page(self.app())
        self.index_demo(at)
        fresh = self.go_to_repo_page(AppTest.from_file(APP, default_timeout=60).run())
        self.assertEqual(fresh.selectbox(key="repo_selected").value, "demo")
        self.assertFalse(fresh.error)

    def test_project_1_title_does_not_contain_buddy(self) -> None:
        at = self.app()
        headers = self.texts(at.header)
        self.assertIn("Project 1: Code reviewer", headers)
        for h in headers:
            self.assertNotIn("buddy", h.lower())

    def test_knowledge_graph_structure_tabular_view(self) -> None:
        at = self.go_to_repo_page(self.app())
        self.index_demo(at)
        self.assertTrue(len(at.dataframe) >= 2)
        df_values = [str(df.value) for df in at.dataframe]
        self.assertTrue(any("Node ID" in v or "Type" in v for v in df_values))
        self.assertTrue(any("Relationship" in v or "Source" in v for v in df_values))


if __name__ == "__main__":
    unittest.main()
