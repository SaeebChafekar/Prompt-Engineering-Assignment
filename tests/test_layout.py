"""Tests for check_layout.py, including repairing a flat extraction.

    python -m unittest tests.test_layout -v
"""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

import check_layout

ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_every_manifest_file_exists(self) -> None:
        self.assertEqual(check_layout.find_missing(ROOT), [])

    def test_every_source_file_is_in_the_manifest(self) -> None:
        """Adding a source file without adding it to the manifest fails here, so the checker never goes stale."""
        self.assertEqual(check_layout.unlisted_source_files(ROOT), [])

    def test_your_own_files_in_the_folder_never_matter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "project"
            copy.mkdir()
            for rel in check_layout.MANIFEST:
                (copy / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ROOT / rel, copy / rel)
            for junk in ("RAG.zip", "notes.txt", "scratch.py", "data/index/x/chunks.jsonl", ".venv/lib/site.py",
                         "venv/lib/other.py", "repo_qa/__pycache__/x.py", "tests/__pycache__/y.py"):
                (copy / junk).parent.mkdir(parents=True, exist_ok=True)
                (copy / junk).write_text("x = 1\n")
            self.assertEqual(check_layout.unlisted_source_files(copy), [])
            self.assertEqual(check_layout.find_missing(copy), [])
            (copy / "repo_qa" / "forgotten.py").write_text("x = 1\n")  # a real new source file IS noticed
            self.assertEqual(check_layout.unlisted_source_files(copy), ["repo_qa/forgotten.py"])


class RepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def copy_project(self, flat: bool) -> None:
        for rel in check_layout.MANIFEST:
            target = self.root / (rel.rsplit("/", 1)[-1] if flat else rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, target)  # in a flat copy, same-named files overwrite each other

    def run_main(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = check_layout.main(list(args), root=self.root)
        return code, out.getvalue()

    def test_complete_layout_passes(self) -> None:
        self.copy_project(flat=False)
        code, out = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn(f"{len(check_layout.MANIFEST)}/{len(check_layout.MANIFEST)} files in place", out)
        self.assertIn("Layout OK", out)

    def test_wrong_folder_is_detected(self) -> None:
        self.copy_project(flat=False)
        (self.root / "repo_qa" / "ingest" / "chunker.py").unlink()
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("MISSING: repo_qa/ingest/chunker.py", out)

    def test_not_in_project_root(self) -> None:
        code, out = self.run_main()  # an empty folder
        self.assertEqual(code, 2)
        self.assertIn("project root", out)

    def test_flat_extraction_is_reported_and_unique_files_are_repaired(self) -> None:
        self.copy_project(flat=True)
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("python check_layout.py --fix", out)
        self.assertIn("cannot be\nrecovered", out)  # names shared by several folders are ambiguous once flat

        code, out = self.run_main("--fix")
        self.assertEqual(code, 1)  # the ambiguous files remain missing, honestly reported
        self.assertIn("moved config.py -> common/config.py", out)
        self.assertTrue((self.root / "common" / "config.py").is_file())
        self.assertTrue((self.root / "repo_qa" / "ingest" / "chunker.py").is_file())
        self.assertTrue((self.root / "tests" / "test_layout.py").is_file())
        self.assertFalse((self.root / "config.py").exists())  # moved, not copied
        remaining = check_layout.find_missing(self.root)
        self.assertTrue(remaining)
        names = [rel.rsplit("/", 1)[-1] for rel in check_layout.MANIFEST]
        ambiguous = {n for n in names if names.count(n) > 1}  # __init__.py, __main__.py, schema.py, ...
        self.assertTrue(all(rel.rsplit("/", 1)[-1] in ambiguous for rel in remaining))
        self.assertIn("__init__.py", ambiguous)

    def test_fix_is_harmless_on_a_correct_layout(self) -> None:
        self.copy_project(flat=False)
        code, out = self.run_main("--fix")
        self.assertEqual(code, 0)
        self.assertNotIn("moved", out)


if __name__ == "__main__":
    unittest.main()
