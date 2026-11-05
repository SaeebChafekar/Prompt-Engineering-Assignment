"""Tests for the knowledge graph builder, on a synthetic repository with known correct edges.

Run from the project root:

    python -m unittest tests.test_graph_build -v
"""

from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from common.config import Config
from repo_qa.graph import KnowledgeGraph, build_graph, write_call_labeling_csv
from repo_qa.graph.__main__ import main as cli_main
from repo_qa.graph.resolve import decorator_reference
from repo_qa.graph.schema import (
    CERTAIN, EDGE_CALLS, EDGE_CONTAINS, EDGE_DECORATED_BY, EDGE_DOCUMENTS, EDGE_IMPORTS,
    EDGE_INHERITS, EDGE_USES, GUESS, LIKELY, class_id, external_id, file_id, function_id,
)
from repo_qa.ingest.pipeline import ingest_directory

FILES = {
    "README.md": """# Demo

The `login` view in `pkg/views.py` uses `User.check` to verify passwords.

## Models

See `pkg/models.py` for the `User` class and `verify`. Start with `python main.py`.
""",
    "pkg/__init__.py": "from .models import User\nfrom .utils import helper\n",
    "pkg/models.py": '''from flask_login import UserMixin
import jwt


class Base:
    def __init__(self):
        self.ready = True

    def save(self):
        return persist(self)


class User(UserMixin, Base):
    """A user."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def check(self, password):
        return verify(password) and self.save()


def verify(password):
    return jwt.decode(password, "key")


def persist(obj):
    return obj
''',
    "pkg/utils.py": '''from .models import User, verify


def helper(x):
    user = User(x)
    return user.check("secret")
''',
    "pkg/views.py": '''from . import utils
from pkg.models import User as U
from pkg import helper
from flask_login import login_required
from flask import Blueprint

bp = Blueprint("views", __name__)


def tracked(fn):
    return fn


@bp.route("/login")
@login_required
@tracked
def login(request):
    utils.helper(request)
    U.check(None, "x")
    helper(request)
    print("done")
    unknown_function()
    request.json.get("a")

    def inner():
        return deep()

    return inner()


def logout():
    return login(None)
''',
    "pkg/star.py": "from .utils import *\n\n\ndef uses_star():\n    return helper(1)\n",
    "pkg/process.py": '''class A:
    def process(self):
        return 1


class B:
    def process(self):
        return 2


class C:
    def rare(self):
        return 3


class D:
    def many(self):
        return 4


class E:
    def many(self):
        return 5


class F:
    def many(self):
        return 6


class G:
    def many(self):
        return 7


def run(obj):
    obj.process()
    obj.rare()
    obj.nothing()
    obj.many()
''',
    "app/__init__.py": "VERSION = 1\n",
    "app/types.py": "def make():\n    return 1\n",
    "app/use.py": "import types\n\n\ndef f():\n    return types.SimpleNamespace()\n",
    "src/lib/__init__.py": "",
    "src/lib/core.py": "def core():\n    return 1\n",
    "scripts/tools.py": "def go():\n    return 1\n",
    "scripts/run.py": "import tools\n\n\ndef main():\n    return tools.go()\n",
    "main.py": "from lib.core import core\nimport tools\n\ncore()\ntools.go()\n",
}


def fid(path: str) -> str:
    return file_id(path)


def fn(path: str, qualname: str) -> str:
    return function_id(path, qualname)


def cls(path: str, qualname: str) -> str:
    return class_id(path, qualname)


class GraphTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name) / "demo"
        for rel, text in FILES.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        cls.root = root
        cls.result = ingest_directory(root, Config(data_dir=str(Path(cls._tmp.name) / "data")))
        cls.graph, cls.report = build_graph(cls.result)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def edge(self, src: str, dst: str, edge_type: str) -> dict | None:
        if self.graph.g.has_edge(src, dst, key=edge_type):
            return self.graph.g[src][dst][edge_type]
        return None

    def assertEdge(self, src: str, dst: str, edge_type: str, confidence: str | None = None, via: str | None = None) -> dict:
        data = self.edge(src, dst, edge_type)
        self.assertIsNotNone(data, f"missing {edge_type}: {src} -> {dst}")
        if confidence:
            self.assertEqual(data["confidence"], confidence, f"{edge_type} {src} -> {dst}")
        if via:
            self.assertEqual(data["via"], via, f"{edge_type} {src} -> {dst}")
        return data

    def assertNoEdge(self, src: str, dst: str, edge_type: str) -> None:
        self.assertIsNone(self.edge(src, dst, edge_type), f"unexpected {edge_type}: {src} -> {dst}")


class NodeTests(GraphTestCase):
    def test_node_types_and_counts(self) -> None:
        counts = self.graph.counts()["nodes"]
        self.assertEqual(counts["Class"], 9)  # Base User A B C D E F G
        self.assertEqual(counts["DocSection"], 2)
        self.assertEqual(counts["File"], len(FILES))
        self.assertIn("External", counts)

    def test_nested_functions_are_folded(self) -> None:
        self.assertIn(fn("pkg/views.py", "login"), self.graph)
        self.assertNotIn(fn("pkg/views.py", "login.inner"), self.graph)
        names = {d["qualname"] for _, d in self.graph.nodes("Function")}
        self.assertNotIn("login.inner", names)

    def test_contains_edges(self) -> None:
        self.assertEdge(fid("pkg/models.py"), cls("pkg/models.py", "User"), EDGE_CONTAINS)
        self.assertEdge(cls("pkg/models.py", "User"), fn("pkg/models.py", "User.check"), EDGE_CONTAINS)
        self.assertEdge(fid("pkg/views.py"), fn("pkg/views.py", "login"), EDGE_CONTAINS)
        self.assertNoEdge(fid("pkg/models.py"), fn("pkg/models.py", "User.check"), EDGE_CONTAINS)

    def test_function_attributes(self) -> None:
        login = self.graph.node(fn("pkg/views.py", "login"))
        self.assertEqual(login["signature"], "login(request)")
        self.assertEqual(login["decorators"], ["bp.route('/login')", "login_required", "tracked"])
        self.assertFalse(login["is_method"])
        self.assertTrue(self.graph.node(fn("pkg/models.py", "User.check"))["is_method"])

    def test_chunks_are_linked_to_nodes(self) -> None:
        node_id = fn("pkg/views.py", "login")
        chunk_ids = self.graph.node(node_id)["chunk_ids"]
        self.assertTrue(chunk_ids)
        for chunk_id in chunk_ids:
            self.assertEqual(self.graph.node_id_for_chunk(chunk_id), node_id)
        self.assertTrue(self.graph.node(fid("pkg/models.py"))["chunk_ids"])  # module-level chunk
        self.assertIsNone(self.graph.node_id_for_chunk("does/not/exist:1-2"))

    def test_external_nodes_flag_stdlib(self) -> None:
        self.assertFalse(self.graph.node(external_id("flask_login"))["stdlib"])
        self.assertTrue(self.graph.node(external_id("types"))["stdlib"])


class ImportTests(GraphTestCase):
    def test_relative_and_absolute_imports(self) -> None:
        self.assertEdge(fid("pkg/views.py"), fid("pkg/utils.py"), EDGE_IMPORTS, CERTAIN)  # from . import utils
        self.assertEdge(fid("pkg/views.py"), fid("pkg/models.py"), EDGE_IMPORTS)  # from pkg.models import ...
        self.assertEdge(fid("pkg/utils.py"), fid("pkg/models.py"), EDGE_IMPORTS)  # from .models import ...
        self.assertEdge(fid("pkg/__init__.py"), fid("pkg/models.py"), EDGE_IMPORTS)
        self.assertEdge(fid("pkg/__init__.py"), fid("pkg/utils.py"), EDGE_IMPORTS)

    def test_import_through_package_reexport_points_at_the_package(self) -> None:
        data = self.assertEdge(fid("pkg/views.py"), fid("pkg/__init__.py"), EDGE_IMPORTS)  # from pkg import helper
        self.assertIn("helper", data["names"])

    def test_external_imports(self) -> None:
        self.assertEdge(fid("pkg/models.py"), external_id("jwt"), EDGE_IMPORTS)
        self.assertEdge(fid("pkg/models.py"), external_id("flask_login"), EDGE_IMPORTS)
        self.assertEdge(fid("pkg/views.py"), external_id("flask"), EDGE_IMPORTS)

    def test_source_root_and_script_imports(self) -> None:
        self.assertEdge(fid("main.py"), fid("src/lib/core.py"), EDGE_IMPORTS)  # `from lib.core import core`, src/ is a root
        self.assertEdge(fid("main.py"), fid("scripts/tools.py"), EDGE_IMPORTS)  # `import tools`, scripts/ is a root
        self.assertEdge(fid("scripts/run.py"), fid("scripts/tools.py"), EDGE_IMPORTS)  # sibling import

    def test_stdlib_name_is_never_matched_to_a_repo_file(self) -> None:
        self.assertEdge(fid("app/use.py"), external_id("types"), EDGE_IMPORTS)
        self.assertNoEdge(fid("app/use.py"), fid("app/types.py"), EDGE_IMPORTS)


class InheritanceAndDecoratorTests(GraphTestCase):
    def test_inheritance(self) -> None:
        user = cls("pkg/models.py", "User")
        self.assertEdge(user, cls("pkg/models.py", "Base"), EDGE_INHERITS, CERTAIN)
        self.assertEdge(user, external_id("flask_login"), EDGE_INHERITS, CERTAIN)

    def test_decorators(self) -> None:
        login = fn("pkg/views.py", "login")
        self.assertEdge(login, external_id("flask_login"), EDGE_DECORATED_BY, CERTAIN)  # imported decorator
        self.assertEdge(login, fn("pkg/views.py", "tracked"), EDGE_DECORATED_BY, CERTAIN)  # local decorator
        route = self.assertEdge(login, external_id("flask"), EDGE_DECORATED_BY, LIKELY)  # bp = Blueprint(...)
        self.assertEqual(route["names"], ["bp.route('/login')"])

    def test_decorator_reference_parsing(self) -> None:
        self.assertEqual(decorator_reference("login_required"), "login_required")
        self.assertEqual(decorator_reference("@bp.route('/x', methods=['GET'])"), "bp.route")
        self.assertEqual(decorator_reference("a.b.c"), "a.b.c")
        self.assertIsNone(decorator_reference("funcs[0]"))
        self.assertIsNone(decorator_reference("not valid python ("))


class CallTests(GraphTestCase):
    def test_local_and_module_attribute_calls(self) -> None:
        self.assertEdge(fn("pkg/views.py", "logout"), fn("pkg/views.py", "login"), EDGE_CALLS, CERTAIN, "local")
        data = self.assertEdge(fn("pkg/views.py", "login"), fn("pkg/utils.py", "helper"), EDGE_CALLS, CERTAIN)
        self.assertEqual(data["names"], ["helper", "utils.helper"])  # both spellings merged: module attr + re-export

    def test_class_attribute_and_instantiation(self) -> None:
        self.assertEdge(fn("pkg/views.py", "login"), fn("pkg/models.py", "User.check"), EDGE_CALLS, CERTAIN, "class-attr")
        self.assertEdge(fn("pkg/utils.py", "helper"), cls("pkg/models.py", "User"), EDGE_CALLS, CERTAIN, "instantiation")

    def test_unknown_receiver_falls_back_to_method_name_as_a_guess(self) -> None:
        self.assertEdge(fn("pkg/utils.py", "helper"), fn("pkg/models.py", "User.check"), EDGE_CALLS, GUESS, "method-name")

    def test_self_calls_and_inheritance(self) -> None:
        check = fn("pkg/models.py", "User.check")
        self.assertEdge(check, fn("pkg/models.py", "verify"), EDGE_CALLS, CERTAIN)
        self.assertEdge(check, fn("pkg/models.py", "Base.save"), EDGE_CALLS, LIKELY, "inherited")  # self.save() in a base class
        self.assertEdge(fn("pkg/models.py", "Base.save"), fn("pkg/models.py", "persist"), EDGE_CALLS, CERTAIN)

    def test_super_init(self) -> None:
        self.assertEdge(fn("pkg/models.py", "User.__init__"), fn("pkg/models.py", "Base.__init__"), EDGE_CALLS, LIKELY, "super")

    def test_library_calls_are_uses_edges(self) -> None:
        data = self.assertEdge(fn("pkg/models.py", "verify"), external_id("jwt"), EDGE_USES)
        self.assertEqual(data["names"], ["jwt.decode"])
        self.assertEdge(fn("app/use.py", "f"), external_id("types"), EDGE_USES)

    def test_star_import(self) -> None:
        self.assertEdge(fn("pkg/star.py", "uses_star"), fn("pkg/utils.py", "helper"), EDGE_CALLS, LIKELY, "star-import")

    def test_module_level_calls_come_from_the_file(self) -> None:
        self.assertEdge(fid("main.py"), fn("src/lib/core.py", "core"), EDGE_CALLS, CERTAIN)
        self.assertEdge(fid("main.py"), fn("scripts/tools.py", "go"), EDGE_CALLS, CERTAIN)
        self.assertEdge(fn("scripts/run.py", "main"), fn("scripts/tools.py", "go"), EDGE_CALLS, CERTAIN)

    def test_method_name_matching_few_classes_gives_guesses_for_each(self) -> None:
        run = fn("pkg/process.py", "run")
        self.assertEdge(run, fn("pkg/process.py", "A.process"), EDGE_CALLS, GUESS)
        self.assertEdge(run, fn("pkg/process.py", "B.process"), EDGE_CALLS, GUESS)
        self.assertEdge(run, fn("pkg/process.py", "C.rare"), EDGE_CALLS, GUESS)

    def test_ambiguous_and_foreign_calls_create_no_edges(self) -> None:
        run = fn("pkg/process.py", "run")
        for cls_name in "DEFG":
            self.assertNoEdge(run, fn("pkg/process.py", f"{cls_name}.many"), EDGE_CALLS)
        self.assertGreaterEqual(self.report.calls.ambiguous, 1)  # obj.many(): 4 candidates
        self.assertGreaterEqual(self.report.calls.foreign, 1)  # obj.nothing(): no such method

    def test_builtins_and_generic_methods_are_not_edges(self) -> None:
        login = fn("pkg/views.py", "login")
        targets = {dst for _, dst, d in self.graph.g.out_edges(login, data=True) if d["type"] in (EDGE_CALLS, EDGE_USES)}
        self.assertEqual(targets, {fn("pkg/utils.py", "helper"), fn("pkg/models.py", "User.check")})
        self.assertGreaterEqual(self.report.calls.builtin, 1)  # print
        self.assertGreaterEqual(self.report.calls.skipped, 2)  # request.json.get (generic) and inner() (nested)
        self.assertGreaterEqual(self.report.calls.unresolved, 2)  # unknown_function() and deep() inside inner()

    def test_every_call_is_counted_exactly_once(self) -> None:
        c = self.report.calls
        accounted = c.resolved + c.external + c.builtin + c.skipped + c.unresolved + c.ambiguous + c.foreign
        self.assertEqual(accounted, c.total)
        self.assertGreaterEqual(len(self.report.call_rows), c.total)  # guesses can add several rows per call

    def test_calls_inside_nested_functions_belong_to_the_parent(self) -> None:
        # `deep()` is called inside login.inner, so it is attributed to login and counted as unresolved
        rows = [r for r in self.report.call_rows if r[1] == "deep"]
        self.assertEqual([(r[0], r[3]) for r in rows], [(fn("pkg/views.py", "login"), "unresolved")])


class DocTests(GraphTestCase):
    def test_sections_link_to_what_they_mention(self) -> None:
        intro = next(n for n, d in self.graph.nodes("DocSection") if d["title"] == "Demo")
        models = next(n for n, d in self.graph.nodes("DocSection") if d["title"] == "Demo > Models")
        self.assertEdge(intro, fid("pkg/views.py"), EDGE_DOCUMENTS, CERTAIN)
        self.assertEdge(intro, fn("pkg/views.py", "login"), EDGE_DOCUMENTS, LIKELY)
        self.assertEdge(intro, fn("pkg/models.py", "User.check"), EDGE_DOCUMENTS, LIKELY)
        self.assertEdge(models, fid("pkg/models.py"), EDGE_DOCUMENTS, CERTAIN)
        self.assertEdge(models, cls("pkg/models.py", "User"), EDGE_DOCUMENTS)
        self.assertEdge(models, fn("pkg/models.py", "verify"), EDGE_DOCUMENTS)
        self.assertEdge(models, fid("main.py"), EDGE_DOCUMENTS)  # bare path token `main.py`
        self.assertEqual(self.report.doc_links, len(list(self.graph.edges(EDGE_DOCUMENTS))))

    def test_doc_links_can_be_turned_off(self) -> None:
        graph, report = build_graph(self.result, with_doc_links=False)
        self.assertEqual(list(graph.edges(EDGE_DOCUMENTS)), [])
        self.assertEqual(report.doc_links, 0)


class OutputTests(GraphTestCase):
    def test_labeling_csv(self) -> None:
        out = Path(self._tmp.name) / "calls.csv"
        write_call_labeling_csv(self.report, out)
        with out.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0][-1], "correct (y/n)")
        self.assertEqual(len(rows) - 1, len(self.report.call_rows))
        self.assertTrue(all(row[-1] == "" for row in rows[1:]))

    def test_report_to_dict_is_json_friendly(self) -> None:
        import json

        json.dumps(self.report.to_dict())
        self.assertNotIn("call_rows", self.report.to_dict())

    def test_graph_meta_has_counts_and_report(self) -> None:
        self.assertEqual(self.graph.meta["repo"], "demo")
        self.assertIn("nodes", self.graph.meta["counts"])
        self.assertIn("calls", self.graph.meta["report"])

    def test_save_and_load_round_trip(self) -> None:
        path = Path(self._tmp.name) / "out" / "graph.json"
        self.graph.save(path)
        self.assertFalse(path.with_suffix(".json.tmp").exists())
        loaded = KnowledgeGraph.load(path)
        self.assertEqual(loaded.to_dict(), self.graph.to_dict())
        self.assertEdge(fn("pkg/views.py", "logout"), fn("pkg/views.py", "login"), EDGE_CALLS)
        self.assertEqual(loaded.counts(), self.graph.counts())

    def test_cli(self) -> None:
        out_dir = Path(self._tmp.name) / "cli_out"
        csv_path = Path(self._tmp.name) / "cli_calls.csv"
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main([
                str(self.root), "--callers", "check", "--callees", "login", "--imports", "views.py",
                "--find", "verify", "--out", str(out_dir), "--labeling-csv", str(csv_path),
            ])
        text = out.getvalue()
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("Callers of pkg/models.py::User.check", text)
        self.assertIn("pkg/views.py::login", text)
        self.assertIn("pkg/views.py::login calls", text)
        self.assertIn("pkg/views.py is imported by", text)
        self.assertIn("[Function] pkg/models.py::verify", text)
        self.assertTrue((out_dir / "graph.json").exists() and (out_dir / "stats.txt").exists())
        self.assertTrue(csv_path.exists())

    def test_cli_errors(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main(["definitely-not-a-folder-or-url"])
        self.assertEqual(code, 1)
        self.assertIn("ERROR", err.getvalue())


class SelfTest(unittest.TestCase):
    def test_graph_of_this_project(self) -> None:
        """The builder works on real code: this project's own source."""
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            result = ingest_directory(root, Config(data_dir=tmp))
        graph, report = build_graph(result)
        self.assertTrue(graph.find("OllamaClient", "Class"))
        gemini_send = graph.find("GeminiClient._send", "Function")
        gemini_open = graph.find("GeminiClient._open_stream", "Function")
        self.assertTrue(gemini_send and gemini_open)
        data = graph.g.get_edge_data(gemini_open[0], gemini_send[0], key=EDGE_CALLS)
        self.assertIsNotNone(data)
        self.assertEqual(data["confidence"], CERTAIN)  # self._send(...) inside the same class
        factory = graph.find("create_chat_client", "Function")[0]
        callee_labels = {graph.label(n) for n, _ in graph.callees(factory)}
        self.assertIn("common/llm_client.py::OllamaClient", callee_labels)
        self.assertGreater(report.calls.resolved, 50)


if __name__ == "__main__":
    unittest.main()
