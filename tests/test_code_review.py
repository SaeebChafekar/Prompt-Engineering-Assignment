"""Tests for the code reviewer (Project 1): schema validation, the reviewer flow, rendering and CLI.

    python -m unittest tests.test_code_review -v
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from code_review import CodeReviewer, REVIEW_SCHEMA, Review, SEVERITIES
from code_review.__main__ import main as cli_main
from code_review.prompts import SYSTEM_PROMPT
from code_review.render import review_markdown
from code_review.reviewer import (
    MAX_CODE_CHARS, MAX_CODE_LINES, detect_language, normalize_code, number_lines, python_syntax_issue,
)
from code_review.schema import MAX_ISSUES, Issue
from common.config import Config
from common.errors import InvalidOutputError, LLMError
from tests.fakes import FakeChat

CODE = "def average(numbers):\n    total = 0\n    for i in range(len(numbers) + 1):\n        total += numbers[i]\n    return total / len(numbers)\n"


def issue(**overrides) -> dict:
    base = {"severity": "high", "category": "bug", "line_start": 3, "line_end": 4, "title": "Off-by-one loop",
            "problem": "range goes one past the end", "why_it_matters": "IndexError", "minimal_fix": "range(len(numbers))"}
    base.update(overrides)
    return base


def payload(*issues: dict, summary: str = "One bug found.", verdict: str = "Fix it first.") -> dict:
    return {"summary": summary, "issues": list(issues), "verdict": verdict}


class SchemaTests(unittest.TestCase):
    def test_schema_describes_the_contract(self) -> None:
        item = REVIEW_SCHEMA["properties"]["issues"]["items"]
        self.assertEqual(item["properties"]["severity"]["enum"], list(SEVERITIES))
        self.assertEqual(set(item["required"]), set(item["properties"]))
        self.assertEqual(REVIEW_SCHEMA["required"], ["summary", "issues", "verdict"])

    def test_valid_payload(self) -> None:
        review = Review.from_payload(payload(issue()), 5)
        self.assertEqual(len(review.issues), 1)
        first = review.issues[0]
        self.assertEqual((first.severity, first.category, first.lines, first.source), ("high", "bug", "L3-4", "model"))
        self.assertEqual(review.counts()["high"], 1)

    def test_no_issues_is_valid(self) -> None:
        review = Review.from_payload(payload(summary="Looks fine."), 5)
        self.assertEqual(review.issues, ())
        self.assertEqual(sum(review.counts().values()), 0)

    def test_severity_aliases_and_unknown_values(self) -> None:
        review = Review.from_payload(payload(issue(severity="MAJOR", title="a"), issue(severity="warning", title="b"),
                                             issue(severity="banana", title="c"), issue(severity="Nit", title="d")), 10)
        self.assertEqual({i.title: i.severity for i in review.issues}, {"a": "high", "b": "medium", "c": "medium", "d": "style"})

    def test_category_normalization(self) -> None:
        review = Review.from_payload(payload(issue(category="Error Handling", title="a"), issue(category="weird", title="b")), 10)
        self.assertEqual({i.title: i.category for i in review.issues}, {"a": "error-handling", "b": "other"})

    def test_line_numbers_are_clamped_swapped_and_repaired(self) -> None:
        review = Review.from_payload(payload(
            issue(line_start=900, line_end=950, title="a"), issue(line_start=8, line_end=2, title="b"),
            issue(line_start="4", line_end=None, title="c"), issue(line_start=-5, line_end=0, title="d"),
            issue(line_start=True, line_end="x", title="e"),
        ), 10)
        got = {i.title: (i.line_start, i.line_end) for i in review.issues}
        self.assertEqual(got, {"a": (10, 10), "b": (2, 8), "c": (4, 4), "d": (1, 1), "e": (1, 1)})

    def test_issues_are_sorted_by_severity_then_line(self) -> None:
        review = Review.from_payload(payload(
            issue(severity="low", line_start=1, title="z"), issue(severity="critical", line_start=9, title="y"),
            issue(severity="critical", line_start=2, title="x"), issue(severity="style", line_start=1, title="w"),
        ), 10)
        self.assertEqual([i.title for i in review.issues], ["x", "y", "z", "w"])

    def test_duplicates_empty_shells_and_cap(self) -> None:
        review = Review.from_payload(payload(issue(), issue(), issue(title="", problem="", minimal_fix="x")), 10)
        self.assertEqual(len(review.issues), 1)
        many = [issue(title=f"t{i}", line_start=i % 10 + 1) for i in range(MAX_ISSUES + 10)]
        self.assertEqual(len(Review.from_payload(payload(*many), 20).issues), MAX_ISSUES)

    def test_title_and_problem_fall_back_to_each_other(self) -> None:
        review = Review.from_payload(payload(issue(title="", problem="Something is wrong here")), 10)
        self.assertEqual(review.issues[0].title, "Something is wrong here")

    def test_long_text_is_truncated(self) -> None:
        review = Review.from_payload(payload(issue(problem="p" * 5000)), 10)
        self.assertLessEqual(len(review.issues[0].problem), 700)
        self.assertTrue(review.issues[0].problem.endswith("..."))

    def test_unusable_payloads_raise(self) -> None:
        for bad in ("text", None, [], {"summary": "x"}, {"summary": "x", "issues": "no"},
                    {"summary": "", "issues": []}, {"summary": 5, "issues": []},
                    payload(issue(), "not an object")):
            with self.assertRaises(InvalidOutputError, msg=repr(bad)):
                Review.from_payload(bad, 10)

    def test_serialization_and_with_issue(self) -> None:
        review = Review.from_payload(payload(issue(severity="low")), 10)
        json.dumps(review.to_dict())
        extra = Issue("critical", "bug", 1, 1, "Syntax error", "p", "w", "f", source="parser")
        updated = review.with_issue(extra)
        self.assertEqual([i.severity for i in updated.issues], ["critical", "low"])
        self.assertEqual(len(review.issues), 1)  # the original is unchanged (frozen)


class HelperTests(unittest.TestCase):
    def test_detect_language(self) -> None:
        self.assertEqual(detect_language("x", hint="Rust"), "rust")
        self.assertEqual(detect_language("x", filename="app.JS"), "javascript")
        self.assertEqual(detect_language("x", filename="notes.zzz"), "code")
        self.assertEqual(detect_language(CODE), "python")
        self.assertEqual(detect_language("const a = () => console.log(1)\nlet b = 2"), "javascript")
        self.assertEqual(detect_language("#include <stdio.h>\nint main() { printf(\"x\"); }"), "c")
        self.assertEqual(detect_language("hello there", hint="auto"), "code")
        self.assertEqual(detect_language("def broken(:\n    pass\n"), "python")  # broken code must still be recognised
        self.assertEqual(detect_language("SELECT name from users where id = 1"), "sql")
        self.assertEqual(detect_language("just some prose with a def in it"), "code")

    def test_number_lines_aligns_columns(self) -> None:
        numbered = number_lines("\n".join(f"l{i}" for i in range(12)))
        self.assertTrue(numbered.startswith(" 1 | l0\n 2 | l1"))
        self.assertTrue(numbered.endswith("12 | l11"))

    def test_normalize_code(self) -> None:
        self.assertEqual(normalize_code("\ufeffa = 1\r\nb = 2\r\n\n"), "a = 1\nb = 2")
        for bad in ("", "   \n  ", None):
            with self.assertRaises(ValueError):
                normalize_code(bad)
        with self.assertRaises(ValueError) as ctx:
            normalize_code("x" * (MAX_CODE_CHARS + 1))
        self.assertIn("too long", str(ctx.exception))
        with self.assertRaises(ValueError):
            normalize_code("\n".join("x" for _ in range(MAX_CODE_LINES + 1)))
        with self.assertRaises(ValueError):
            normalize_code("a\x00b")

    def test_python_syntax_check(self) -> None:
        note, found = python_syntax_issue("def broken(:\n    pass\n")
        self.assertEqual((found.severity, found.line_start, found.source), ("critical", 1, "parser"))
        self.assertIn("line 1", note)
        note, found = python_syntax_issue(CODE)
        self.assertIsNone(found)
        self.assertIn("accepted", note)
        python_syntax_issue("a = 1\x00")  # NUL bytes never reach here (normalize_code rejects them); must not raise


class ReviewerTests(unittest.TestCase):
    def reviewer(self, chat: FakeChat) -> CodeReviewer:
        return CodeReviewer(chat, Config())

    def test_review_flow_and_prompt(self) -> None:
        chat = FakeChat([payload(issue())])
        result = self.reviewer(chat).review(CODE, problem="average([]) crashes", filename="stats.py")
        call = chat.calls[0]
        self.assertEqual((call["kind"], call["system"]), ("json", SYSTEM_PROMPT))
        self.assertEqual(call["kw"]["schema"], REVIEW_SCHEMA)
        message = call["messages"][0]["content"]
        self.assertIn("Language: python", message)
        self.assertIn("Reported problem (check this first): average([]) crashes", message)
        self.assertIn("Automatic check result: Python's parser accepted", message)
        self.assertIn("3 |     for i in range(len(numbers) + 1):", message)
        self.assertEqual((result.language, result.line_count, result.attempts), ("python", 5, 1))
        self.assertEqual(result.review.issues[0].title, "Off-by-one loop")

    def test_token_limit_is_set_only_for_local_models(self) -> None:
        local = FakeChat([payload()])
        local.backend_name = "ollama"
        self.reviewer(local).review(CODE)
        self.assertEqual(local.calls[0]["kw"]["max_tokens"], 2048)
        cloud = FakeChat([payload()])
        cloud.backend_name = "gemini"
        self.reviewer(cloud).review(CODE)
        self.assertIsNone(cloud.calls[0]["kw"]["max_tokens"])

    def test_line_numbers_from_the_model_are_clamped_to_the_code(self) -> None:
        result = self.reviewer(FakeChat([payload(issue(line_start=99, line_end=120))])).review(CODE)
        self.assertEqual((result.review.issues[0].line_start, result.review.issues[0].line_end), (5, 5))

    def test_syntax_error_is_added_by_the_program_even_if_the_model_misses_it(self) -> None:
        result = self.reviewer(FakeChat([payload(summary="No problems.")])).review("def broken(:\n    pass\n")
        self.assertEqual(len(result.review.issues), 1)
        self.assertEqual(result.review.issues[0].source, "parser")
        self.assertIn("syntax error at line 1", result.parser_note)

    def test_syntax_error_is_not_duplicated_when_the_model_found_it(self) -> None:
        model_issue = issue(severity="critical", line_start=1, line_end=1, title="Syntax error in def")
        result = self.reviewer(FakeChat([payload(model_issue)])).review("def broken(:\n    pass\n")
        self.assertEqual([i.source for i in result.review.issues], ["model"])

    def test_non_python_code_has_no_parser_note(self) -> None:
        result = self.reviewer(FakeChat([payload()])).review("const a = () => console.log(1)\nlet b = 2", language="javascript")
        self.assertIsNone(result.parser_note)

    def test_bad_json_is_retried_once_with_the_error_in_the_prompt(self) -> None:
        chat = FakeChat(["this is not json", payload(issue())])
        result = self.reviewer(chat).review(CODE)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(chat.calls), 2)
        correction = chat.calls[1]["messages"][-1]["content"]
        self.assertIn("could not be used", correction)
        self.assertIn("not valid JSON", correction)
        self.assertEqual(len(chat.calls[1]["messages"]), 2)  # original request + correction

    def test_valid_json_with_the_wrong_shape_is_retried(self) -> None:
        chat = FakeChat([{"summary": "x"}, payload()])
        self.assertEqual(self.reviewer(chat).review(CODE).attempts, 2)

    def test_two_bad_replies_give_a_clear_error(self) -> None:
        chat = FakeChat(["nope", "still nope"])
        with self.assertRaises(InvalidOutputError) as ctx:
            self.reviewer(chat).review(CODE)
        self.assertIn("after 2 attempts", str(ctx.exception))
        self.assertEqual(len(chat.calls), 2)

    def test_model_failures_are_not_retried(self) -> None:
        chat = FakeChat([payload()], fail_on={1})
        with self.assertRaises(LLMError):
            self.reviewer(chat).review(CODE)
        self.assertEqual(len(chat.calls), 1)

    def test_input_validation_happens_before_any_model_call(self) -> None:
        chat = FakeChat()
        reviewer = self.reviewer(chat)
        for kwargs in ({"code": ""}, {"code": CODE, "problem": "x" * 1001}):
            with self.assertRaises(ValueError):
                reviewer.review(**kwargs)
        self.assertEqual(chat.calls, [])

    def test_whitespace_only_problem_is_ignored(self) -> None:
        chat = FakeChat([payload()])
        self.reviewer(chat).review(CODE, problem="   ")
        self.assertNotIn("Reported problem", chat.calls[0]["messages"][0]["content"])

    def test_effort_level_controls_prompt_and_tokens(self) -> None:
        chat = FakeChat([payload()])
        reviewer = self.reviewer(chat)
        result = reviewer.review(CODE, effort="high")
        self.assertEqual(result.effort, "high")
        self.assertIn("HIGH EFFORT", chat.calls[0]["system"])
        self.assertEqual(chat.calls[0]["kw"]["max_tokens"], 4096)

        chat_low = FakeChat([payload()])
        reviewer_low = self.reviewer(chat_low)
        result_low = reviewer_low.review(CODE, effort="low")
        self.assertEqual(result_low.effort, "low")
        self.assertIn("FAST / CONCISE", chat_low.calls[0]["system"])

    def test_high_effort_permits_longer_text(self) -> None:
        long_prob = "p" * 2000
        review = Review.from_payload(payload(issue(problem=long_prob)), 10, effort="high")
        self.assertGreater(len(review.issues[0].problem), 1000)


class FollowupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = CodeReviewer(FakeChat([payload(issue())]), Config()).review(CODE)

    def test_followup_includes_code_review_history_and_question(self) -> None:
        chat = FakeChat(["Because the loop reads past the end."])
        reviewer = CodeReviewer(chat, Config())
        pieces: list[str] = []
        answer = reviewer.followup(self.result, "Why is it high?", [("earlier q", "earlier a")], on_token=pieces.append)
        self.assertEqual(answer, "Because the loop reads past the end.")
        self.assertEqual("".join(pieces), answer)
        message = chat.calls[0]["messages"][0]["content"]
        for expected in ("Off-by-one loop", "3 |     for i in range", "earlier q", "earlier a", "Why is it high?"):
            self.assertIn(expected, message)

    def test_validation_and_empty_output(self) -> None:
        reviewer = CodeReviewer(FakeChat(["   "]), Config())
        with self.assertRaises(ValueError):
            reviewer.followup(self.result, "  ")
        with self.assertRaises(ValueError):
            reviewer.followup(self.result, "x" * 1001)
        with self.assertRaises(InvalidOutputError):
            reviewer.followup(self.result, "why?")


class RenderTests(unittest.TestCase):
    def test_markdown_groups_by_severity_and_marks_parser_findings(self) -> None:
        chat = FakeChat([payload(issue(severity="low", title="Repeated work"), issue(severity="high"))])
        result = CodeReviewer(chat, Config()).review("def broken(:\n  pass")
        text = review_markdown(result, title="My review")
        self.assertTrue(text.startswith("# My review"))
        self.assertLess(text.index("## Critical (1)"), text.index("## High (1)"))
        self.assertLess(text.index("## High (1)"), text.index("## Low (1)"))
        self.assertIn("```python", text)
        self.assertIn("Found by the automatic syntax check", text)
        self.assertIn("Automatic check:", text)

    def test_clean_code(self) -> None:
        result = CodeReviewer(FakeChat([payload(summary="Looks good.", verdict="Looks good")]), Config()).review(CODE)
        text = review_markdown(result)
        self.assertIn("no issues", text)
        self.assertNotIn("## Critical", text)


class CliTests(unittest.TestCase):
    def run_cli(self, *args: str, chat=None, stdin: str | None = None) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main(list(args), chat_factory=lambda cfg: chat or FakeChat([payload(issue())]),
                            stdin=io.StringIO(stdin) if stdin is not None else None)
        return code, out.getvalue(), err.getvalue()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.file = Path(self._tmp.name) / "stats.py"
        self.file.write_text(CODE)

    def test_markdown_output(self) -> None:
        code, out, err = self.run_cli(str(self.file), "--problem", "crashes")
        self.assertEqual(code, 0, err)
        self.assertIn("# Review of stats.py", out)
        self.assertIn("Off-by-one loop", out)

    def test_json_output(self) -> None:
        code, out, _ = self.run_cli(str(self.file), "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["issues"][0]["severity"], "high")

    def test_stdin(self) -> None:
        code, out, _ = self.run_cli("-", stdin=CODE)
        self.assertEqual(code, 0)
        self.assertIn("standard input", out)

    def test_errors(self) -> None:
        self.assertEqual(self.run_cli(str(Path(self._tmp.name) / "missing.py"))[0], 1)
        code, _, err = self.run_cli("-", stdin="   ")
        self.assertEqual(code, 1)
        self.assertIn("Paste some code", err)
        big = Path(self._tmp.name) / "big.py"
        big.write_text("x = 1\n" * (MAX_CODE_CHARS))
        self.assertEqual(self.run_cli(str(big))[0], 1)
        code, _, err = self.run_cli(str(self.file), chat=FakeChat(fail_on={1}))
        self.assertEqual(code, 1)
        self.assertIn("LLMError", err)


if __name__ == "__main__":
    unittest.main()
