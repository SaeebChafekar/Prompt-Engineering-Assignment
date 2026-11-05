"""Tests for the Python parser (repo_qa.ingest.parse_python).

Run from the project root:

    python -m unittest tests.test_ingest_parse -v
"""

from __future__ import annotations

import unittest

from repo_qa.ingest.parse_python import module_name_from_path, parse_python

SAMPLE = '''"""Auth module."""
import os
import jwt as pyjwt
from . import utils
from .models import User as U, Session
from pkg.sub import *
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pkg.types import Token

try:
    import ujson
except ImportError:
    ujson = None

TOKEN_TTL = 3600


@decorator
class AuthService(Base):
    """Handles login."""

    def __init__(self, db):
        self.db = db

    @login_required
    async def verify_token(self, token: str) -> bool:
        """Check a token."""
        data = pyjwt.decode(token, key)
        user = self.db.get_user(data["id"])
        helper().strip()

        def inner():
            return deep()

        return U.check(user)


def main():
    import lazy_dep

    svc = AuthService(db)
    svc.verify_token("x")


if __name__ == "__main__":
    main()
'''


def line_of(text: str) -> int:
    """1-based line number of the first line of SAMPLE containing text."""
    for number, line in enumerate(SAMPLE.splitlines(), 1):
        if text in line:
            return number
    raise AssertionError(f"{text!r} not in SAMPLE")


class ModuleNameTests(unittest.TestCase):
    def test_module_names(self) -> None:
        self.assertEqual(module_name_from_path("src/auth/jwt.py"), "src.auth.jwt")
        self.assertEqual(module_name_from_path("pkg/__init__.py"), "pkg")
        self.assertEqual(module_name_from_path("script.py"), "script")


class ParseSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parsed = parse_python("pkg/auth.py", SAMPLE)
        cls.funcs = {f.qualname: f for f in cls.parsed.functions}
        cls.classes = {c.qualname: c for c in cls.parsed.classes}

    def test_basics(self) -> None:
        self.assertIsNone(self.parsed.error)
        self.assertEqual(self.parsed.module, "pkg.auth")
        self.assertEqual(self.parsed.docstring, "Auth module.")

    def test_class(self) -> None:
        cls = self.classes["AuthService"]
        self.assertEqual(cls.bases, ("Base",))
        self.assertEqual(cls.decorators, ("decorator",))
        self.assertEqual(cls.docstring, "Handles login.")
        self.assertEqual(cls.methods, ("AuthService.__init__", "AuthService.verify_token"))
        self.assertEqual(cls.start_line, line_of("@decorator"))  # decorators are part of the span
        self.assertEqual(cls.def_line, line_of("class AuthService"))
        self.assertEqual(cls.end_line, line_of("return U.check"))
        self.assertFalse(cls.in_function)

    def test_function_list_and_nesting(self) -> None:
        self.assertEqual(
            set(self.funcs),
            {"AuthService.__init__", "AuthService.verify_token", "AuthService.verify_token.inner", "main"},
        )
        self.assertTrue(self.funcs["AuthService.verify_token.inner"].is_nested)
        self.assertFalse(self.funcs["AuthService.verify_token"].is_nested)
        self.assertFalse(self.funcs["main"].is_nested)

    def test_method_details(self) -> None:
        method = self.funcs["AuthService.verify_token"]
        self.assertEqual(method.signature, "async verify_token(self, token: str) -> bool")
        self.assertEqual(method.decorators, ("login_required",))
        self.assertEqual(method.parent_class, "AuthService")
        self.assertEqual(method.docstring, "Check a token.")
        self.assertEqual(method.start_line, line_of("@login_required"))
        self.assertEqual(method.def_line, line_of("async def verify_token"))

    def test_calls_in_source_order_and_scoped(self) -> None:
        method = self.funcs["AuthService.verify_token"]
        self.assertEqual(method.calls[:2], ("pyjwt.decode", "self.db.get_user"))
        self.assertIn("helper", method.calls)
        self.assertIn("?.strip", method.calls)  # method call on the result of an expression
        self.assertIn("U.check", method.calls)
        self.assertNotIn("deep", method.calls)  # belongs to the nested function
        self.assertEqual(self.funcs["AuthService.verify_token.inner"].calls, ("deep",))
        self.assertEqual(self.funcs["main"].calls, ("AuthService", "svc.verify_token"))

    def test_module_level_calls(self) -> None:
        self.assertEqual(self.parsed.module_calls, ("main",))

    def test_imports_and_bindings(self) -> None:
        bindings = {}
        for imp in self.parsed.imports:
            for local, target in imp.bindings:
                bindings[local] = (target, imp.level, imp.scope)
        self.assertEqual(bindings["os"], ("os", 0, ""))
        self.assertEqual(bindings["pyjwt"], ("jwt", 0, ""))  # import jwt as pyjwt
        self.assertEqual(bindings["utils"], (".utils", 1, ""))  # from . import utils
        self.assertEqual(bindings["U"], (".models.User", 1, ""))  # from .models import User as U
        self.assertEqual(bindings["Session"], (".models.Session", 1, ""))
        self.assertEqual(bindings["*"], ("pkg.sub.*", 0, ""))
        self.assertEqual(bindings["Token"], ("pkg.types.Token", 0, ""))  # inside `if TYPE_CHECKING`
        self.assertEqual(bindings["ujson"], ("ujson", 0, ""))  # inside try/except
        self.assertEqual(bindings["lazy_dep"], ("lazy_dep", 0, "main"))  # import inside a function

    def test_relative_import_without_name_module(self) -> None:
        imp = next(i for i in self.parsed.imports if i.level == 1 and i.module == "")
        self.assertEqual(imp.bindings, (("utils", ".utils"),))


class ParseEdgeCaseTests(unittest.TestCase):
    def test_syntax_error_is_reported_not_raised(self) -> None:
        parsed = parse_python("bad.py", "def (:\n  pass")
        self.assertIsNotNone(parsed.error)
        self.assertTrue(parsed.error.startswith("SyntaxError"))
        self.assertEqual(parsed.functions, ())

    def test_python2_code(self) -> None:
        self.assertIsNotNone(parse_python("old.py", 'print "hello"\n').error)

    def test_null_bytes(self) -> None:
        self.assertIsNotNone(parse_python("x.py", "a = 1\x00\n").error)

    def test_extreme_nesting_does_not_crash(self) -> None:
        parsed = parse_python("deep.py", "x = " + "(" * 3000 + "1" + ")" * 3000 + "\n")
        # Either parses or reports an error; the point is that it never raises.
        self.assertIsInstance(parsed.module, str)

    def test_empty_file(self) -> None:
        parsed = parse_python("empty.py", "")
        self.assertIsNone(parsed.error)
        self.assertEqual((parsed.classes, parsed.functions, parsed.imports), ((), (), ()))

    def test_class_inside_function_is_flagged(self) -> None:
        parsed = parse_python("m.py", "def make():\n    class Local:\n        def run(self):\n            pass\n    return Local\n")
        local = next(c for c in parsed.classes if c.name == "Local")
        self.assertTrue(local.in_function)
        run = next(f for f in parsed.functions if f.name == "run")
        self.assertTrue(run.is_nested)

    def test_invalid_escape_warning_is_silenced(self) -> None:
        parsed = parse_python("re.py", 'import re\nPATTERN = re.compile("\\d+")\n')
        self.assertIsNone(parsed.error)

    def test_to_dict_is_json_friendly(self) -> None:
        import json

        parsed = parse_python("pkg/auth.py", SAMPLE)
        json.dumps(parsed.to_dict())


if __name__ == "__main__":
    unittest.main()
