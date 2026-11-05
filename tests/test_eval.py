"""Tests for the evaluation harness: dataset, metrics, runners, reports, call accuracy and the CLI.

    python -m unittest tests.test_eval -v
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from common.config import Config
from common.errors import LLMError
from repo_qa.eval import calls, report
from repo_qa.eval.__main__ import DEFAULT_QUESTIONS, main as eval_main
from repo_qa.eval.dataset import CATEGORIES, Question, load_questions, parse_gold, validate_against_index
from repo_qa.eval.metrics import answer_scores, chunk_matches, retrieval_scores
from repo_qa.eval.runner import (
    AnswerRow, RetrievalRow, append_jsonl, default_conditions, parse_condition, read_jsonl, run_answers, run_retrieval,
)
from repo_qa.graph import build_graph, write_call_labeling_csv
from repo_qa.index import HybridSearcher, build_index
from repo_qa.ingest.pipeline import ingest_directory
from repo_qa.rag import Answerer, Retriever
from repo_qa.rag.answer import Answer
from repo_qa.rag.retrieve import ContextItem, Retrieval
from tests.fakes import FakeChat
from tests.test_index import FakeEmbedder
from tests.test_rag import FILES

QUESTIONS = {
    "questions": [
        {"id": "a1", "category": "locate", "question": "Where is login handled?", "gold": ["app/auth.py::login"], "must_include": ["login"]},
        {"id": "a2", "category": "relationship", "question": "What does login call?",
         "gold": ["app/auth.py::find_user", "app/auth.py::verify_password"], "must_include": ["find_user", "verify_password"]},
        {"id": "a3", "category": "explain", "question": "How are invoices created?", "gold": ["app/billing.py"], "must_include": ["invoice"]},
        {"id": "u1", "category": "unanswerable", "question": "Which payment provider is used?", "answerable": False, "gold": [], "must_include": []},
    ]
}


def make_question(**overrides) -> Question:
    base = dict(id="q", category="locate", question="?", gold=("app/auth.py::login",), must_include=("login",), answerable=True)
    base.update(overrides)
    return Question(**base)


class EvalCase(unittest.TestCase):
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
        cls.graph, cls.report = build_graph(cls.result)
        cls.embedder = FakeEmbedder()
        cls.index = build_index(cls.result, cls.graph, cls.embedder, cls.cfg)
        cls.keyword_index = build_index(cls.result, cls.graph, None, cls.cfg)
        cls.questions_file = cls.base / "questions.json"
        cls.questions_file.write_text(json.dumps(QUESTIONS))
        cls.questions = load_questions(cls.questions_file)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def retriever(self, index=None) -> Retriever:
        index = index or self.index
        return Retriever(index, HybridSearcher(index, self.embedder if index.has_vectors else None, self.cfg), self.cfg)

    def answerer(self, chat: FakeChat, index=None) -> Answerer:
        return Answerer(self.retriever(index), chat, self.cfg)


class DatasetTests(unittest.TestCase):
    def write(self, data) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "q.json"
        path.write_text(data if isinstance(data, str) else json.dumps(data))
        return path

    def test_shipped_question_file_is_well_formed(self) -> None:
        questions = load_questions(DEFAULT_QUESTIONS)
        self.assertGreaterEqual(len(questions), 20)
        self.assertEqual(len({q.id for q in questions}), len(questions))
        self.assertTrue({q.category for q in questions} <= set(CATEGORIES))
        self.assertGreaterEqual(sum(1 for q in questions if not q.answerable), 3)
        for q in questions:
            self.assertTrue(q.gold or not q.answerable, q.id)
            for g in q.gold:
                self.assertTrue(parse_gold(g)[0].endswith((".py", ".md")), g)

    def test_parse_gold(self) -> None:
        self.assertEqual(parse_gold("app/models.py::User.check"), ("app/models.py", "User.check"))
        self.assertEqual(parse_gold("app/x.py"), ("app/x.py", None))
        self.assertEqual(parse_gold("app/x.py::"), ("app/x.py", None))

    def test_invalid_files_are_rejected_with_the_question_id(self) -> None:
        bad = {
            "not json": "{nope",
            "no list": {"questions": []},
            "not an object": {"questions": ["x"]},
            "no id": {"questions": [{"question": "?", "category": "locate", "gold": ["a.py"]}]},
            "empty text": {"questions": [{"id": "x", "question": " ", "category": "locate", "gold": ["a.py"]}]},
            "bad category": {"questions": [{"id": "x", "question": "?", "category": "weird", "gold": ["a.py"]}]},
            "answerable needs gold": {"questions": [{"id": "x", "question": "?", "category": "locate", "gold": []}]},
            "unanswerable has gold": {"questions": [{"id": "x", "question": "?", "category": "unanswerable", "answerable": False, "gold": ["a.py"]}]},
            "gold type": {"questions": [{"id": "x", "question": "?", "category": "locate", "gold": "a.py"}]},
            "terms type": {"questions": [{"id": "x", "question": "?", "category": "locate", "gold": ["a.py"], "must_include": "t"}]},
            "answerable type": {"questions": [{"id": "x", "question": "?", "category": "locate", "gold": ["a.py"], "answerable": "yes"}]},
            "duplicate": {"questions": [{"id": "x", "question": "?", "category": "locate", "gold": ["a.py"]}] * 2},
        }
        for name, data in bad.items():
            with self.assertRaises(ValueError, msg=name):
                load_questions(self.write(data))
        with self.assertRaises(ValueError):
            load_questions(Path("/nonexistent/q.json"))
        with self.assertRaises(ValueError) as ctx:
            load_questions(self.write(bad["empty text"]))
        self.assertIn("x", str(ctx.exception))


class ValidateTests(EvalCase):
    def test_gold_locations_are_checked_against_the_index(self) -> None:
        self.assertEqual(validate_against_index(self.questions, self.index), [])
        bad = [make_question(id="b1", gold=("app/missing.py",)), make_question(id="b2", gold=("app/auth.py::nope",)),
               make_question(id="b3", gold=("app/auth.py::login", "app/auth.py"))]
        problems = validate_against_index(bad, self.index)
        self.assertEqual(len(problems), 2)
        self.assertIn("b1: file 'app/missing.py' is not in the index", problems)
        self.assertIn("b2: no function or class 'nope' in app/auth.py", problems)


class MetricsTests(EvalCase):
    def items(self, *symbols: str) -> list[ContextItem]:
        chunks = {c.symbol: c for c in self.index.chunks if c.symbol}
        return [ContextItem(chunks[s], 1.0, "search", "x") for s in symbols]

    def test_chunk_matches(self) -> None:
        chunk = next(c for c in self.index.chunks if c.symbol == "login")
        self.assertTrue(chunk_matches(chunk, "app/auth.py::login"))
        self.assertTrue(chunk_matches(chunk, "app/auth.py"))
        self.assertFalse(chunk_matches(chunk, "app/auth.py::find_user"))
        self.assertFalse(chunk_matches(chunk, "app/billing.py"))
        self.assertFalse(chunk_matches(chunk, "auth.py::login"))  # exact paths only

    def test_retrieval_scores(self) -> None:
        items = self.items("create_invoice", "login", "find_user")
        s = retrieval_scores(items, ["app/auth.py::login", "app/auth.py::verify_password", "app/models.py::User"])
        self.assertAlmostEqual(s["recall"], 1 / 3)
        self.assertEqual(s["found"], ["app/auth.py::login"])
        self.assertEqual(s["missing"], ["app/auth.py::verify_password", "app/models.py::User"])
        self.assertEqual(s["first_rank"], 2)
        self.assertAlmostEqual(s["file_recall"], 0.5)  # auth.py found, models.py not
        none = retrieval_scores(items, ["app/models.py::User"])
        self.assertEqual((none["recall"], none["first_rank"]), (0.0, None))
        self.assertEqual(retrieval_scores([], ["app/auth.py"])["recall"], 0.0)
        self.assertEqual(retrieval_scores(items, [])["recall"], 0.0)

    def answer(self, text: str, cited: list[str], invalid: list[str] | None = None) -> Answer:
        return Answer("q", "q", text, Retrieval("q"), cited, invalid or [])

    def test_answer_scores(self) -> None:
        q = make_question(gold=("app/auth.py::login", "app/models.py::User"), must_include=("login", "password"))
        s = answer_scores(q, self.answer("The Login view checks the PASSWORD [app/auth.py:7-10]", ["app/auth.py:7-10"]))
        self.assertEqual((s["gold_cited"], s["terms"], s["n_cited"], s["refused"]), (0.5, 1.0, 1, False))
        s = answer_scores(q, self.answer("I couldn't find this in the indexed code.", [], ["x.py:1"]))
        self.assertEqual((s["refused"], s["gold_cited"], s["terms"], s["n_invalid"]), (True, 0.0, 0.0, 1))
        s = answer_scores(make_question(must_include=()), self.answer("anything", []))
        self.assertEqual(s["terms"], 1.0)
        suffix = answer_scores(make_question(gold=("app/auth.py",)), self.answer("x", ["auth.py:3-4"]))
        self.assertEqual(suffix["gold_cited"], 1.0)


class RunnerTests(EvalCase):
    def test_conditions(self) -> None:
        self.assertEqual(parse_condition("hybrid+graph"), ("hybrid", True))
        self.assertEqual(parse_condition(" keyword "), ("keyword", False))
        for bad in ("telepathy", "hybrid+magic", "hybrid+graph+x", ""):
            with self.assertRaises(ValueError, msg=bad):
                parse_condition(bad)
        self.assertEqual(default_conditions(self.index), ["keyword", "vector", "hybrid", "hybrid+graph"])
        self.assertEqual(default_conditions(self.index, for_answers=True), ["hybrid", "hybrid+graph"])
        self.assertEqual(default_conditions(self.keyword_index), ["keyword", "keyword+graph"])

    def test_retrieval_run(self) -> None:
        seen: list[RetrievalRow] = []
        rows = run_retrieval(self.retriever(), self.questions, ["hybrid", "hybrid+graph"], k=1, on_row=seen.append)
        self.assertEqual(len(rows), 3 * 2)  # the unanswerable question is skipped
        self.assertEqual(rows, seen)
        by = {(r.question_id, r.condition): r for r in rows}
        plain, graph = by[("a2", "hybrid")], by[("a2", "hybrid+graph")]
        self.assertGreaterEqual(graph.recall, plain.recall)  # the graph never loses evidence
        self.assertGreater(graph.n_items, plain.n_items)
        self.assertGreater(graph.tokens, plain.tokens)
        self.assertEqual(by[("a1", "hybrid")].recall, 1.0)

    def test_retrieval_run_rejects_a_bad_condition_before_doing_work(self) -> None:
        with self.assertRaises(ValueError):
            run_retrieval(self.retriever(), self.questions, ["nonsense"])

    def test_answer_run_and_scoring(self) -> None:
        ticks = iter(range(0, 1000, 3))
        rows = run_answers(self.answerer(FakeChat(["The login view is in [app/auth.py:7-10]."])), self.questions,
                           ["hybrid", "hybrid+graph"], clock=lambda: float(next(ticks)))
        self.assertEqual(len(rows), 4 * 2)
        row = next(r for r in rows if r.question_id == "a1" and r.condition == "hybrid")
        self.assertEqual((row.cited, row.invalid_citations, row.gold_cited, row.terms), (["app/auth.py:7-10"], [], 1.0, 1.0))
        self.assertEqual((row.seconds, row.error, row.question, row.gold), (3.0, None, "Where is login handled?", ["app/auth.py::login"]))
        self.assertFalse(next(r for r in rows if r.question_id == "u1").answerable)

    def test_errors_are_recorded_and_the_run_continues(self) -> None:
        rows = run_answers(self.answerer(FakeChat(["Answer [app/auth.py:7-10]."], fail_on={2})), self.questions, ["hybrid"])
        failed = [r for r in rows if r.error]
        self.assertEqual(len(failed), 1)
        self.assertIn("LLMError", failed[0].error)
        self.assertEqual(len(rows), 4)

    def test_skip_and_limit(self) -> None:
        chat = FakeChat(["Answer."])
        rows = run_answers(self.answerer(chat), self.questions, ["hybrid"], skip={("a1", "hybrid")}, limit=2)
        self.assertEqual([r.question_id for r in rows], ["a2"])  # first two questions, minus the skipped one
        self.assertEqual(len(chat.calls), 1)

    def test_refusals_are_detected(self) -> None:
        rows = run_answers(self.answerer(FakeChat(["I couldn't find this in the indexed code."])), self.questions, ["hybrid"])
        self.assertTrue(all(r.refused for r in rows))

    def test_jsonl_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "rows.jsonl"
            rows = run_answers(self.answerer(FakeChat(["Answer [app/auth.py:7-10]."])), self.questions[:2], ["hybrid"])
            for r in rows:
                append_jsonl(path, r)
            self.assertEqual(read_jsonl(path, AnswerRow), rows)
            self.assertEqual(read_jsonl(Path(tmp) / "none.jsonl", AnswerRow), [])
            path.write_text("{broken\n")
            with self.assertRaises(ValueError):
                read_jsonl(path, AnswerRow)


class ReportTests(unittest.TestCase):
    def rrow(self, qid, cond, recall, rank=1, cat="locate", tokens=100, items=5) -> RetrievalRow:
        return RetrievalRow(qid, cat, cond, recall, 1.0 if recall else 0.0, rank, [], [], items, tokens, 0)

    def test_retrieval_table_numbers_and_graph_effect(self) -> None:
        rows = [self.rrow("q1", "hybrid", 0.5, rank=2), self.rrow("q2", "hybrid", 1.0), self.rrow("q3", "hybrid", 0.0, rank=None),
                self.rrow("q1", "hybrid+graph", 1.0, tokens=200, items=9), self.rrow("q2", "hybrid+graph", 1.0, tokens=200, items=9),
                self.rrow("q3", "hybrid+graph", 0.0, rank=None, tokens=200, items=9)]
        text = report.retrieval_report(rows)
        self.assertIn("| hybrid | 3 | 50% | 67% | 33% | 0.50 | 5.0 | 100 |", text)  # recall (0.5+1+0)/3, MRR (0.5+1+0)/3
        self.assertIn("| hybrid+graph | 3 | 67% | 67% | 67% | 0.67 | 9.0 | 200 |", text)
        self.assertIn("improved 1, unchanged 2, worse 0", text)
        self.assertIn("q1 (50% to 100%)", text)

    def test_graph_can_be_worse(self) -> None:
        text = report.retrieval_report([self.rrow("q1", "keyword", 1.0), self.rrow("q1", "keyword+graph", 0.5)])
        self.assertIn("worse 1", text)
        self.assertIn("q1 (100% to 50%)", text)

    def test_by_type_table(self) -> None:
        rows = [self.rrow("q1", "hybrid", 1.0, cat="locate"), self.rrow("q2", "hybrid", 0.0, cat="relationship")]
        text = report.retrieval_report(rows)
        self.assertIn("| locate | 100% |", text)
        self.assertIn("| relationship | 0% |", text)

    def arow(self, qid, cond, answerable=True, cited=("a.py:1",), invalid=(), refused=False, gold=1.0, terms=1.0, secs=10.0, err=None):
        return AnswerRow(qid, "locate" if answerable else "unanswerable", cond, answerable, "" if err else "text", list(cited), list(invalid),
                         refused, gold, terms, 5, 100, secs, err, [], "question?")

    def test_answer_table(self) -> None:
        rows = [self.arow("q1", "hybrid"), self.arow("q2", "hybrid", invalid=("x.py:1",), gold=0.0, terms=0.5),
                self.arow("q3", "hybrid", refused=True, cited=(), gold=0.0, terms=0.0),
                self.arow("u1", "hybrid", answerable=False, refused=True, cited=()),
                self.arow("u2", "hybrid", answerable=False, refused=False),
                self.arow("q4", "hybrid", err="LLMError: boom", secs=1.0)]
        text = report.answer_report(rows)
        # 3 answerable without errors: citations valid for 1 of 3; gold 1/3; terms 0.5; wrongly refused 1/3; correct refusals 1/2
        self.assertIn("| hybrid | 6 | 1 | 33% | 33% | 50% | 33% | 50% | 10.0s |", text)

    def test_empty_inputs(self) -> None:
        self.assertIn("No retrieval results", report.retrieval_report([]))
        self.assertIn("No answer results", report.answer_report([]))

    def test_grading_csv_and_human_scores(self) -> None:
        rows = [self.arow("q1", "hybrid"), self.arow("q1", "hybrid+graph"), self.arow("q2", "hybrid", err="LLMError: x")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grading.csv"
            report.write_grading_csv(rows, path)
            with open(path, newline="", encoding="utf-8") as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual(len(table), 3)
            self.assertEqual(table[0][report.GRADE_COLUMN], "")
            self.assertIn("[ERROR] LLMError: x", table[2]["answer"])
            self.assertEqual(report.human_scores(path), {})  # nothing graded yet
            self.assertIn("No hand grades", report.human_report(path))
            table[0][report.GRADE_COLUMN], table[1][report.GRADE_COLUMN] = "2", "1"
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
            scores = report.human_scores(path)
            self.assertEqual(scores["hybrid"], {"graded": 1, "mean": 2.0, "correct": 1.0})
            self.assertEqual(scores["hybrid+graph"]["mean"], 1.0)
            self.assertIn("2.00 / 2", report.human_report(path))
            table[2][report.GRADE_COLUMN] = "7"
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
            with self.assertRaises(ValueError):
                report.human_scores(path)

    def test_build_report_from_a_results_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertIn("No retrieval results", report.build_report(out))
            append_jsonl(out / "retrieval.jsonl", self.rrow("q1", "hybrid", 1.0))
            append_jsonl(out / "answers.jsonl", self.arow("q1", "hybrid"))
            text = report.build_report(out)
            self.assertIn("# Evaluation report", text)
            self.assertIn("| hybrid | 1 | 100%", text)
            self.assertIn("How to read these numbers", text)


class CallAccuracyTests(EvalCase):
    def calls_rows(self) -> list[dict]:
        path = self.base / "calls.csv"
        write_call_labeling_csv(self.report, path)
        return calls.read_calls_csv(path)

    def test_sample_is_stratified_repeatable_and_has_call_sites(self) -> None:
        rows = self.calls_rows()
        sample = calls.sample_calls(self.index, rows, 6, seed=3)
        self.assertEqual(sample, calls.sample_calls(self.index, rows, 6, seed=3))
        self.assertTrue(all(r["confidence"] in calls.LEVELS for r in sample))
        self.assertTrue(all(r["resolved_to"] for r in sample))
        keys = [(r["caller"], r["call_as_written"], r["resolved_to"]) for r in sample]
        self.assertEqual(len(keys), len(set(keys)))
        site = next(r["call_site"] for r in sample if r["call_as_written"] == "verify_password")
        self.assertRegex(site, r"^app/auth\.py:\d+: return verify_password\(user, request\)$")
        self.assertEqual([r["id"] for r in sample], list(range(1, len(sample) + 1)))

    def test_sample_larger_than_available_returns_everything(self) -> None:
        rows = self.calls_rows()
        resolved = {(r["caller"], r["call_as_written"], r["resolved_to"]) for r in rows if r["resolved_to"]}
        self.assertEqual(len(calls.sample_calls(self.index, rows, 1000)), len(resolved))
        with self.assertRaises(ValueError):
            calls.sample_calls(self.index, rows, 0)

    def test_calls_into_libraries_are_excluded_unless_asked_for(self) -> None:
        rows = self.calls_rows() + [
            {"caller": "func:app/auth.py::login", "call_as_written": "jwt.decode", "resolved_to": "ext:jwt",
             "confidence_or_outcome": "certain", "via": "import"},
            {"caller": "func:app/auth.py::login", "call_as_written": "db.add", "resolved_to": "ext:flask_sqlalchemy",
             "confidence_or_outcome": "likely", "via": "instance"},
        ]
        default = calls.sample_calls(self.index, rows, 1000)
        self.assertFalse(any(r["resolved_to"].startswith("ext:") for r in default))
        with_libs = calls.sample_calls(self.index, rows, 1000, include_external=True)
        self.assertEqual(len(with_libs), len(default) + 2)

    def test_call_site_prefers_the_exact_call_over_the_method_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mini"
            root.mkdir()
            (root / "m.py").write_text("def f(a, b):\n    a.run()\n    b.run()\n    return 1\n")
            result = ingest_directory(root, Config(data_dir=tmp))
            graph, _ = build_graph(result)
            index = build_index(result, graph, None, self.cfg)
            self.assertEqual(calls.call_site(index, "func:m.py::f", "b.run"), "m.py:3: b.run()")
            self.assertEqual(calls.call_site(index, "func:m.py::f", "a.run"), "m.py:2: a.run()")
            self.assertEqual(calls.call_site(index, "func:m.py::f", "?.run"), "m.py:2: a.run()")  # unknown receiver: first match

    def test_call_site_edge_cases(self) -> None:
        self.assertEqual(calls.call_site(self.index, "func:nope::x", "x"), "")
        login = "func:app/auth.py::login"
        self.assertIn("def login(request):", calls.call_site(self.index, login, "not_called_here"))  # falls back to the first line

    def test_precision_from_labels(self) -> None:
        sample = calls.sample_calls(self.index, self.calls_rows(), 8, seed=1)
        self.assertGreaterEqual(len(sample), 6)  # the toy repo has 6 distinct resolved calls
        sample = sample[:6]
        path = self.base / "labelled.csv"
        for i, row in enumerate(sample):
            row[calls.LABEL_COLUMN] = ["y", "Y", "yes", "n", "no", ""][i]
        calls.write_sample(sample, path)
        result = calls.calls_precision(path)
        labelled = sum(r["labelled"] for r in result.values())
        self.assertEqual(labelled, 5)
        self.assertEqual(sum(r["unlabelled"] for r in result.values()), 1)
        self.assertEqual(sum(r["correct"] for r in result.values()), 3)  # y, Y, yes
        text = calls.precision_text(result)
        self.assertIn("| Confidence | Labelled | Correct | Precision | 95% range |", text)
        self.assertIn("still unlabelled", text)

    def test_bad_labels_and_files(self) -> None:
        sample = calls.sample_calls(self.index, self.calls_rows(), 2)
        sample[0][calls.LABEL_COLUMN] = "maybe"
        path = self.base / "bad.csv"
        calls.write_sample(sample, path)
        with self.assertRaises(ValueError):
            calls.calls_precision(path)
        with self.assertRaises(ValueError):
            calls.calls_precision(self.base / "missing.csv")
        wrong = self.base / "wrong.csv"
        wrong.write_text("a,b\n1,2\n")
        with self.assertRaises(ValueError):
            calls.read_calls_csv(wrong)

    def test_wilson_interval(self) -> None:
        self.assertEqual(calls.wilson_interval(0, 0), (0.0, 0.0))
        low, high = calls.wilson_interval(10, 10)
        self.assertGreater(low, 0.69)
        self.assertEqual(high, 1.0)
        low, high = calls.wilson_interval(5, 10)
        self.assertTrue(0.2 < low < 0.5 < high < 0.8)
        self.assertLess(calls.wilson_interval(50, 100)[1] - calls.wilson_interval(50, 100)[0],
                        calls.wilson_interval(5, 10)[1] - calls.wilson_interval(5, 10)[0])  # more data, narrower range


class CliTests(EvalCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = Path(self._dir.name)
        self.keyword_dir = self.root / "kw_index"
        self.keyword_index.save(self.keyword_dir)
        self.vector_dir = self.root / "vec_index"
        self.index.save(self.vector_dir)
        self.out: list[str] = []
        self.chat = FakeChat(["It is handled in [app/auth.py:7-10]."])

    def run_cli(self, *args: str, chat=None) -> int:
        return eval_main(list(args), chat_factory=lambda cfg: chat or self.chat,
                         embedder_factory=lambda cfg: FakeEmbedder(), write=self.out.append)

    @property
    def output(self) -> str:
        return "\n".join(self.out)

    def test_validate(self) -> None:
        self.assertEqual(self.run_cli("validate", str(self.keyword_dir), "--questions", str(self.questions_file)), 0)
        self.assertIn("4 questions checked; 0 problem(s).", self.output)
        broken = self.root / "broken.json"
        broken.write_text(json.dumps({"questions": [{"id": "x", "category": "locate", "question": "?", "gold": ["app/nope.py"]}]}))
        self.assertEqual(self.run_cli("validate", str(self.keyword_dir), "--questions", str(broken)), 1)
        self.assertIn("PROBLEM: x: file 'app/nope.py' is not in the index", self.output)

    def test_retrieval_on_a_keyword_index(self) -> None:
        out = self.root / "res"
        code = self.run_cli("retrieval", str(self.keyword_dir), "--questions", str(self.questions_file), "--out", str(out))
        self.assertEqual(code, 0)
        self.assertIn("| keyword+graph |", self.output)
        self.assertEqual(len(read_jsonl(out / "retrieval.jsonl", RetrievalRow)), 3 * 2)
        self.run_cli("retrieval", str(self.keyword_dir), "--questions", str(self.questions_file), "--out", str(out))
        self.assertEqual(len(read_jsonl(out / "retrieval.jsonl", RetrievalRow)), 3 * 2)  # re-running replaces, never duplicates

    def test_retrieval_with_vectors(self) -> None:
        code = self.run_cli("retrieval", str(self.vector_dir), "--questions", str(self.questions_file), "--conditions", "hybrid,hybrid+graph",
                            "--k", "2", "--out", str(self.root / "res2"))
        self.assertEqual(code, 0)
        self.assertIn("| hybrid+graph |", self.output)

    def test_conditions_that_need_vectors_fail_on_a_keyword_index(self) -> None:
        code = self.run_cli("retrieval", str(self.keyword_dir), "--questions", str(self.questions_file), "--conditions", "hybrid")
        self.assertEqual(code, 1)
        self.assertEqual(self.run_cli("retrieval", str(self.keyword_dir), "--questions", str(self.questions_file),
                                      "--conditions", "bogus"), 1)

    def test_answers_resume_and_report(self) -> None:
        out = self.root / "ans"
        args = ["answers", str(self.keyword_dir), "--questions", str(self.questions_file), "--out", str(out)]
        self.assertEqual(self.run_cli(*args, "--limit", "2"), 0)
        rows = read_jsonl(out / "answers.jsonl", AnswerRow)
        self.assertEqual(len(rows), 2 * 2)  # two questions x (keyword, keyword+graph)
        self.assertTrue((out / "answers_for_grading.csv").exists())
        calls_before = len(self.chat.calls)
        self.assertEqual(self.run_cli(*args, "--resume"), 0)
        rows = read_jsonl(out / "answers.jsonl", AnswerRow)
        self.assertEqual(len(rows), 4 * 2)
        self.assertEqual(len(self.chat.calls) - calls_before, 4)  # only the 2 unfinished questions x 2 conditions
        self.assertEqual(self.run_cli("report", str(out)), 0)
        self.assertIn("# Evaluation report", (out / "report.md").read_text())

    def test_failed_answers_are_retried_on_resume(self) -> None:
        out = self.root / "retry"
        args = ["answers", str(self.keyword_dir), "--questions", str(self.questions_file), "--out", str(out), "--conditions", "keyword"]
        self.run_cli(*args, chat=FakeChat(["Answer [app/auth.py:7-10]."], fail_on={2}))
        self.assertEqual(sum(1 for r in read_jsonl(out / "answers.jsonl", AnswerRow) if r.error), 1)
        self.run_cli(*args, "--resume", chat=FakeChat(["Answer [app/auth.py:7-10]."]))
        rows = read_jsonl(out / "answers.jsonl", AnswerRow)
        self.assertEqual((len(rows), sum(1 for r in rows if r.error)), (4, 0))

    def test_answers_without_a_model_is_a_clean_error(self) -> None:
        class Down(FakeChat):
            def health_check(self) -> None:
                raise LLMError("model server down")

        code = self.run_cli("answers", str(self.keyword_dir), "--questions", str(self.questions_file), chat=Down())
        self.assertEqual(code, 1)

    def test_call_accuracy_commands(self) -> None:
        calls_csv = self.root / "calls.csv"
        write_call_labeling_csv(self.report, calls_csv)
        sample = self.root / "sample.csv"
        self.assertEqual(self.run_cli("sample-calls", str(self.keyword_dir), str(calls_csv), "--n", "5", "--out", str(sample)), 0)
        rows = list(csv.DictReader(open(sample, newline="", encoding="utf-8")))
        self.assertEqual(len(rows), 5)
        for row in rows:
            row[calls.LABEL_COLUMN] = "y"
        calls.write_sample(rows, sample)
        self.assertEqual(self.run_cli("calls-precision", str(sample)), 0)
        self.assertIn("100%", self.output)
        self.assertEqual(self.run_cli("calls-precision", str(self.root / "missing.csv")), 1)

    def test_missing_index(self) -> None:
        self.assertEqual(self.run_cli("retrieval", str(self.root / "nope")), 1)


if __name__ == "__main__":
    unittest.main()
