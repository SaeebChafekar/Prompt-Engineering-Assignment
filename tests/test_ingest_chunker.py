"""Tests for the chunker (repo_qa.ingest.chunker).

Run from the project root:

    python -m unittest tests.test_ingest_chunker -v
"""

from __future__ import annotations

import textwrap
import unittest

from common.config import Config
from repo_qa.ingest.chunker import TRUNCATION_MARK, chunk_file
from repo_qa.ingest.models import SourceFile
from repo_qa.ingest.parse_python import parse_python

PY = '''"""Auth module: login and token checks."""
import os
import jwt

TOKEN_TTL = 3600


class AuthService(Base):
    """Handles login."""

    def __init__(self, db):
        self.db = db

    def verify_token(self, token):
        """Check a token."""
        data = jwt.decode(token, os.environ["KEY"])

        def inner():
            return data

        return inner()


def main():
    svc = AuthService(None)
    return svc


if __name__ == "__main__":
    main()
'''


def make_file(path: str, text: str, language: str, kind: str = "code", is_test: bool = False) -> SourceFile:
    return SourceFile(path=path, language=language, kind=kind, text=text, size=len(text), is_test=is_test)


def chunks_for_python(text: str = PY, path: str = "pkg/auth.py", cfg: Config | None = None):
    sf = make_file(path, text, "python")
    return chunk_file(sf, parse_python(path, text), cfg or Config())


class PythonChunkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = chunks_for_python()
        self.by_symbol = {c.symbol: c for c in self.chunks if c.symbol}

    def test_expected_chunks_exist(self) -> None:
        self.assertEqual(
            set(self.by_symbol), {"AuthService", "AuthService.__init__", "AuthService.verify_token", "main"}
        )
        self.assertEqual(sum(c.kind == "module" for c in self.chunks), 2)  # top of file + __main__ block

    def test_nested_function_stays_inside_parent(self) -> None:
        self.assertNotIn("AuthService.verify_token.inner", self.by_symbol)
        self.assertIn("def inner():", self.by_symbol["AuthService.verify_token"].text)

    def test_function_chunk_matches_source_lines(self) -> None:
        lines = PY.splitlines()
        main = self.by_symbol["main"]
        self.assertEqual(main.kind, "function")
        self.assertEqual("\n".join(lines[main.start_line - 1 : main.end_line]), main.text)
        self.assertEqual(main.citation, f"pkg/auth.py:{main.start_line}-{main.end_line}")

    def test_method_chunk_is_dedented_and_labeled(self) -> None:
        method = self.by_symbol["AuthService.verify_token"]
        self.assertTrue(method.text.startswith("def verify_token(self, token):"))
        self.assertIn("Function: AuthService.verify_token (method of AuthService)", method.context)
        original = PY.splitlines()[method.start_line - 1 : method.end_line]
        self.assertEqual(textwrap.dedent("\n".join(original)), method.text)

    def test_class_summary_has_signature_docstring_and_method_list_but_no_bodies(self) -> None:
        cls = self.by_symbol["AuthService"]
        self.assertEqual(cls.kind, "class")
        self.assertIn("class AuthService(Base):", cls.text)
        self.assertIn("Handles login.", cls.text)
        self.assertIn("# verify_token(self, token)", cls.text)
        self.assertNotIn("jwt.decode", cls.text)
        lines = PY.splitlines()
        self.assertTrue(lines[cls.start_line - 1].startswith("class AuthService"))
        self.assertTrue(lines[cls.end_line - 1].strip().endswith("return inner()"))

    def test_module_chunks_hold_imports_and_script_block(self) -> None:
        modules = [c for c in self.chunks if c.kind == "module"]
        top = next(c for c in modules if c.start_line == 1)
        self.assertIn("import jwt", top.text)
        self.assertIn("TOKEN_TTL = 3600", top.text)
        self.assertNotIn("class AuthService", top.text)
        tail = next(c for c in modules if c.start_line > 1)
        self.assertIn('if __name__ == "__main__":', tail.text)

    def test_context_header_contains_path_for_embedding(self) -> None:
        for chunk in self.chunks:
            self.assertIn("File: pkg/auth.py", chunk.context)
            self.assertTrue(chunk.embed_text.startswith(chunk.context))

    def test_ids_are_unique_and_chunks_sorted(self) -> None:
        ids = [c.id for c in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))
        starts = [c.start_line for c in self.chunks]
        self.assertEqual(starts, sorted(starts))

    def test_decorators_are_included_in_function_span(self) -> None:
        text = "import x\n\n\n@app.route('/login')\n@login_required\ndef login():\n    return x.go()\n"
        chunks = chunks_for_python(text, "app.py")
        login = next(c for c in chunks if c.symbol == "login")
        self.assertEqual(login.start_line, 4)
        self.assertTrue(login.text.startswith("@app.route('/login')"))

    def test_test_flag_propagates(self) -> None:
        sf = make_file("tests/test_a.py", "def test_x():\n    assert True\n", "python", is_test=True)
        chunks = chunk_file(sf, parse_python(sf.path, sf.text), Config())
        self.assertTrue(chunks and all(c.is_test for c in chunks))

    def test_empty_or_trivial_files_give_no_chunks(self) -> None:
        self.assertEqual(chunks_for_python("", "e.py"), [])
        self.assertEqual(chunks_for_python("\n\n   \n", "e.py"), [])
        self.assertEqual(chunks_for_python("x = 1\n", "tiny.py"), [])  # too small to be useful


class SplittingTests(unittest.TestCase):
    def big_function(self, lines: int = 200) -> str:
        body = "\n".join(f"    value_{i} = compute({i}) + other_value_{i}" for i in range(lines))
        return f"def big():\n{body}\n    return value_0\n"

    def test_oversized_function_is_split_into_overlapping_parts(self) -> None:
        cfg = Config(max_chunk_chars=1000, chunk_overlap_lines=3)
        text = self.big_function()
        parts = [c for c in chunks_for_python(text, "big.py", cfg) if c.symbol == "big"]
        self.assertGreater(len(parts), 3)
        self.assertEqual([p.part for p in parts], list(range(1, len(parts) + 1)))
        self.assertTrue(all(p.parts == len(parts) for p in parts))
        for p in parts:
            self.assertLessEqual(len(p.text), 1000)
            self.assertIn(f"Part: {p.part} of {len(parts)}", p.context)
        for a, b in zip(parts, parts[1:]):
            self.assertEqual(b.start_line, a.end_line - 3 + 1)  # 3 lines of overlap
        lines = text.splitlines()
        self.assertEqual(parts[0].start_line, 1)
        self.assertEqual(parts[-1].end_line, len(lines))
        for p in parts:  # line numbers still match the real file
            self.assertEqual("\n".join(lines[p.start_line - 1 : p.end_line]), p.text)

    def test_overlap_zero_gives_disjoint_parts(self) -> None:
        cfg = Config(max_chunk_chars=1000, chunk_overlap_lines=0)
        parts = [c for c in chunks_for_python(self.big_function(), "big.py", cfg) if c.symbol == "big"]
        for a, b in zip(parts, parts[1:]):
            self.assertEqual(b.start_line, a.end_line + 1)

    def test_single_huge_line_is_clipped_and_terminates(self) -> None:
        cfg = Config(max_chunk_chars=500)
        text = "def f():\n    x = '" + "a" * 5000 + "'\n    return x\n"
        chunks = chunks_for_python(text, "long.py", cfg)
        f_chunks = [c for c in chunks if c.symbol == "f"]
        self.assertTrue(f_chunks)
        self.assertTrue(any(TRUNCATION_MARK in c.text for c in f_chunks))
        self.assertTrue(all(len(c.text) <= 500 + len(TRUNCATION_MARK) for c in f_chunks))


class MarkdownChunkTests(unittest.TestCase):
    MD = textwrap.dedent(
        """\
        # Project

        Intro paragraph that is long enough to keep.

        ## Install

        Run the installer.

        ### Linux

        ```bash
        # this is a comment, not a heading
        sudo apt install thing
        ```

        ### Windows

        ## Empty

        ## Usage

        Call the tool with the right arguments to do the work.
        """
    )

    def setUp(self) -> None:
        sf = make_file("README.md", self.MD, "markdown", kind="doc")
        self.chunks = chunk_file(sf, None, Config())
        self.by_name = {c.name: c for c in self.chunks}

    def test_sections_use_heading_breadcrumbs(self) -> None:
        self.assertEqual(
            set(self.by_name),
            {"Project", "Project > Install", "Project > Install > Linux", "Project > Usage"},
        )
        self.assertTrue(all(c.kind == "doc" for c in self.chunks))

    def test_comment_inside_code_fence_is_not_a_heading(self) -> None:
        linux = self.by_name["Project > Install > Linux"]
        self.assertIn("# this is a comment, not a heading", linux.text)
        self.assertIn("sudo apt install thing", linux.text)

    def test_empty_sections_are_dropped(self) -> None:
        self.assertNotIn("Project > Install > Windows", self.by_name)
        self.assertNotIn("Project > Empty", self.by_name)

    def test_line_numbers_match(self) -> None:
        lines = self.MD.splitlines()
        for chunk in self.chunks:
            self.assertEqual("\n".join(lines[chunk.start_line - 1 : chunk.end_line]), chunk.text)

    def test_context_contains_section_path(self) -> None:
        self.assertIn("Section: Project > Usage", self.by_name["Project > Usage"].context)

    def test_text_before_first_heading_uses_file_name(self) -> None:
        sf = make_file("docs/guide.md", "Plain text before any heading, long enough to keep.\n\n# Later\n\nBody text for the later section here.\n", "markdown", "doc")
        names = [c.name for c in chunk_file(sf, None, Config())]
        self.assertEqual(names, ["guide.md", "Later"])


class OtherFileTests(unittest.TestCase):
    def test_non_python_code_uses_windows(self) -> None:
        text = "\n".join(f"function f{i}() {{ return {i}; }}" for i in range(300))
        sf = make_file("web/app.js", text, "javascript")
        chunks = chunk_file(sf, None, Config(max_chunk_chars=1000))
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(c.kind == "block" and c.language == "javascript" for c in chunks))

    def test_config_and_text_kinds(self) -> None:
        cfg = Config()
        self.assertEqual(chunk_file(make_file("a.yaml", "key: value\nother: 1\n", "yaml", "config"), None, cfg)[0].kind, "config")
        self.assertEqual(chunk_file(make_file("notes.txt", "some notes\nmore notes\n", "text", "doc"), None, cfg)[0].kind, "doc")

    def test_python_with_syntax_error_falls_back_to_blocks(self) -> None:
        text = "def broken(:\n    pass\n"
        sf = make_file("bad.py", text, "python")
        parsed = parse_python(sf.path, text)
        self.assertIsNotNone(parsed.error)
        chunks = chunk_file(sf, parsed, Config())
        self.assertEqual([c.kind for c in chunks], ["block"])
        self.assertIn("def broken", chunks[0].text)

    def test_whitespace_only_file(self) -> None:
        self.assertEqual(chunk_file(make_file("x.js", "\n\n  \n", "javascript"), None, Config()), [])


if __name__ == "__main__":
    unittest.main()
