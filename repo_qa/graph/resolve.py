"""Resolve names written in the source to the definitions they refer to.

The parser only records text: "self.save", "utils.helper", "U.check". This module decides
what each one most likely points at, and how sure it is:

    certain ... found through definitions and imports ("helper()" defined in this file,
                "utils.helper()" through `from . import utils`, "self.save()" in the class)
    likely .... found, but through inheritance, a re-export or a star import
    guess ..... matched only by method name, because the receiver's type is unknown
                ("svc.verify_token()" where svc is a local variable)

Anything we cannot resolve is reported as such, never invented. Python is dynamic, so
some edges will be missing or wrong; the build reports how many were resolved at each
confidence level so that this limitation can be measured and stated honestly.

Two things make the problem harder than it looks, and are handled explicitly:
- circular imports (Flask apps import each other): names are resolved lazily, with
  cycle detection, instead of in one eager pass;
- source roots and scripts: `from lib.core import x` may live in `src/lib/core.py`, and
  scripts import their siblings by bare name. Folders that are not packages (no
  __init__.py) are treated as roots, and a standard-library name is never matched to a
  repository file by accident.
"""

from __future__ import annotations

import ast
import builtins
import sys
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from repo_qa.ingest.models import ClassInfo, ParsedFile

from .schema import CERTAIN, GUESS, LIKELY

BUILTIN_NAMES = frozenset(dir(builtins))
STDLIB_NAMES = frozenset(getattr(sys, "stdlib_module_names", ()))
MAX_METHOD_GUESSES = 3  # a method name matching more classes than this is "ambiguous"
MAX_REEXPORT_DEPTH = 6

# Method names that belong to built-in types (str, list, dict, set, files, paths) far more
# often than to repository classes. Matching them by name would create false edges.
GENERIC_METHODS = frozenset(
    """append extend insert remove pop clear copy update get keys values items setdefault add
    discard union intersection difference join split rsplit strip lstrip rstrip replace format
    startswith endswith lower upper title capitalize encode decode read write close open flush
    readline readlines writelines seek tell sort sorted reverse index count find rfind isdigit
    isalpha isalnum isspace splitlines partition mkdir exists is_file is_dir resolve parent
    group groups match search sub findall finditer fullmatch strftime isoformat total_seconds
    __init__ __enter__ __exit__""".split()
)


# ---------------------------------------------------------------- result types
@dataclass(frozen=True)
class Binding:
    """What a name introduced by an import (or a local definition) refers to."""

    # "module" | "symbol" | "instance" | "variable" | "external" | "unknown"
    # instance = a module-level variable assigned from a constructor call, e.g. `db = SQLAlchemy()`
    kind: str
    path: str = ""  # file holding the module or definition (for an instance: its class's file)
    symbol: str = ""  # qualname of the definition (kind == "symbol")
    symbol_kind: str = ""  # "function" | "class"
    external: str = ""  # top-level package name (kind == "external")
    module_path: str = ""  # the file the import statement itself points at


@dataclass(frozen=True)
class CallTarget:
    """The outcome of resolving one call."""

    # func, class, external   -> an edge is created
    # builtin, skipped        -> deliberately ignored (len(), str.strip(), nested helper)
    # unresolved, ambiguous, foreign -> could not be resolved; counted in the statistics
    kind: str
    path: str = ""
    qualname: str = ""
    external: str = ""
    confidence: str = ""
    via: str = ""


@dataclass(frozen=True)
class CallContext:
    caller_class: str | None  # qualname of the enclosing class for methods
    caller_name: str  # simple name of the calling function
    local_names: frozenset[str] = frozenset()  # nested functions/classes defined inside the caller


# ---------------------------------------------------------------- the index
class RepoIndex:
    """Lookup tables over all parsed Python files, plus lazy import resolution."""

    def __init__(self, python_paths: list[str], parsed: list[ParsedFile]) -> None:
        self.python_paths = set(python_paths)
        self.parsed = {p.path: p for p in parsed if p.error is None}
        self.top_functions: dict[str, dict[str, str]] = {}
        self.top_classes: dict[str, dict[str, str]] = {}
        self.classes: dict[tuple[str, str], ClassInfo] = {}
        self.class_methods: dict[tuple[str, str], dict[str, str]] = {}
        self.method_index: dict[str, list[tuple[str, str]]] = {}
        self._raw: dict[str, dict[str, str]] = {}
        self._stars: dict[str, list[str]] = {}
        self._vars: dict[str, dict[str, str]] = {}
        self._inst_cache: dict[tuple[str, str], Binding | None] = {}
        self._inst_active: set[tuple[str, str]] = set()
        self._cache: dict[tuple[str, str], Binding | None] = {}
        self._mro_cache: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self.full_index: dict[str, str] = {}
        self.module_index: dict[str, list[str]] = {}

        for path, pf in self.parsed.items():
            self._index_definitions(path, pf)
            self._index_imports(path, pf)
        self._build_module_index()

    # ---- indexing
    def _index_definitions(self, path: str, pf: ParsedFile) -> None:
        funcs: dict[str, str] = {}
        classes: dict[str, str] = {}
        for fn in pf.functions:
            if "." not in fn.qualname and not fn.is_nested:
                funcs[fn.name] = fn.qualname
        for cls in pf.classes:
            if cls.in_function:
                continue  # classes defined inside functions are folded into the function
            if "." not in cls.qualname:
                classes[cls.name] = cls.qualname
            self.classes[(path, cls.qualname)] = cls
        self.top_functions[path] = funcs
        self.top_classes[path] = classes
        method_qualnames = {fn.qualname: fn for fn in pf.functions if not fn.is_nested}
        for cls in pf.classes:
            if cls.in_function:
                continue
            table: dict[str, str] = {}
            for qualname in cls.methods:
                if qualname in method_qualnames:
                    name = qualname.rsplit(".", 1)[-1]
                    table[name] = qualname
                    self.method_index.setdefault(name, []).append((path, qualname))
            self.class_methods[(path, cls.qualname)] = table

    def _index_imports(self, path: str, pf: ParsedFile) -> None:
        raw: dict[str, str] = {}
        stars: list[str] = []
        for imp in pf.imports:
            for local, target in imp.bindings:
                if local == "*":
                    stars.append(target[:-2] if target.endswith(".*") else target)
                else:
                    raw[local] = target
        self._raw[path] = raw
        self._stars[path] = stars
        self._vars[path] = dict(pf.assignments)  # later assignments win

    def _build_module_index(self) -> None:
        package_dirs = {
            str(PurePosixPath(p).parent) if str(PurePosixPath(p).parent) != "." else ""
            for p in self.python_paths
            if PurePosixPath(p).name == "__init__.py"
        }
        for path in sorted(self.python_paths):
            parts = list(PurePosixPath(path).with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts.pop()
            if not parts:
                continue  # a top-level __init__.py has no module name
            full = ".".join(parts)
            self.full_index.setdefault(full, path)
            variants = [full]
            dirs = list(PurePosixPath(path).parent.parts)
            stripped = parts[:]
            for i in range(len(dirs)):
                if "/".join(dirs[: i + 1]) in package_dirs:
                    break  # a real package: its name is part of the import path
                stripped = stripped[1:]  # a source root (src/, scripts/, tests/): not importable
                if not stripped:
                    break
                variants.append(".".join(stripped))
            for variant in variants:
                self.module_index.setdefault(variant, []).append(path)

    # ---- module resolution
    def resolve_module(self, from_path: str, module: str, level: int) -> str | None:
        """The repository file a module name refers to, or None (external or unknown)."""
        if level > 0:
            package = list(PurePosixPath(from_path).parent.parts)
            if level - 1 > len(package):
                return None
            base = package[: len(package) - (level - 1)]
            parts = base + [p for p in module.split(".") if p]
            return self.full_index.get(".".join(parts)) if parts else None
        if not module:
            return None
        parts = module.split(".")
        src_dir = str(PurePosixPath(from_path).parent)
        prefix = "" if src_dir == "." else src_dir + "/"
        if prefix + "__init__.py" not in self.python_paths:  # script-style sibling import
            for candidate in (prefix + "/".join(parts) + ".py", prefix + "/".join(parts) + "/__init__.py"):
                if candidate in self.python_paths:
                    return candidate
        candidates = self.module_index.get(module, [])
        if parts[0] in STDLIB_NAMES:  # never mistake `import types` for scripts/types.py
            candidates = [c for c in candidates if self.full_index.get(module) == c]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return self._closest(from_path, candidates)
        return None

    @staticmethod
    def _closest(from_path: str, candidates: list[str]) -> str | None:
        here = PurePosixPath(from_path).parent.parts

        def shared(candidate: str) -> int:
            n = 0
            for a, b in zip(here, PurePosixPath(candidate).parent.parts):
                if a != b:
                    break
                n += 1
            return n

        scored = sorted(((shared(c), c) for c in candidates), reverse=True)
        if scored[0][0] > scored[1][0]:
            return scored[0][1]
        return None  # ambiguous: refuse to guess

    def _submodule(self, package_init: str, name: str) -> str | None:
        if PurePosixPath(package_init).name != "__init__.py":
            return None
        package = ".".join(PurePosixPath(package_init).parent.parts)
        return self.full_index.get(f"{package}.{name}" if package else name)

    # ---- name bindings (lazy, cycle-safe)
    def binding_for(self, path: str, local: str, _stack: tuple = ()) -> Binding | None:
        key = (path, local)
        if key in self._cache:
            return self._cache[key]
        if key in _stack:
            return None  # circular import chain: give up on this name
        target = self._raw.get(path, {}).get(local)
        binding = None if target is None else self.resolve_target(path, target, _stack + (key,))
        self._cache[key] = binding
        return binding

    def resolve_target(self, path: str, target: str, _stack: tuple = ()) -> Binding:
        """Resolve a dotted import target such as '.models.User' or 'flask_login.UserMixin'."""
        level = len(target) - len(target.lstrip("."))
        parts = [p for p in target[level:].split(".") if p]
        for k in range(len(parts), -1, -1):
            if k == 0 and level == 0:
                break
            mod_path = self.resolve_module(path, ".".join(parts[:k]), level)
            if mod_path is None:
                continue
            remaining = parts[k:]
            if not remaining:
                return Binding("module", path=mod_path, module_path=mod_path)
            return self._walk_remaining(mod_path, remaining, _stack, mod_path)
        if level > 0:
            return Binding("unknown")
        return Binding("external", external=parts[0] if parts else "")

    def _walk_remaining(self, mod_path: str, remaining: list[str], stack: tuple, module_path: str) -> Binding:
        binding = self._member(mod_path, remaining[0], stack, module_path)
        if len(remaining) > 1 and binding.kind == "module":
            return self._walk_remaining(binding.path, remaining[1:], stack, module_path)
        return binding

    def _member(self, mod_path: str, name: str, stack: tuple, module_path: str) -> Binding:
        """What `name` is inside the module at mod_path: a definition, re-export, submodule, or variable."""
        if name in self.top_classes.get(mod_path, {}):
            return Binding("symbol", path=mod_path, symbol=self.top_classes[mod_path][name],
                           symbol_kind="class", module_path=module_path)
        if name in self.top_functions.get(mod_path, {}):
            return Binding("symbol", path=mod_path, symbol=self.top_functions[mod_path][name],
                           symbol_kind="function", module_path=module_path)
        instance = self.instance_binding(mod_path, name)
        if instance is not None:
            return replace(instance, module_path=module_path)
        if len(stack) < MAX_REEXPORT_DEPTH:
            reexport = self.binding_for(mod_path, name, stack)
            if reexport is not None and reexport.kind != "unknown":
                return replace(reexport, module_path=module_path)
        sub = self._submodule(mod_path, name)
        if sub:
            return Binding("module", path=sub, module_path=sub)
        return Binding("variable", path=mod_path, module_path=module_path)

    def instance_binding(self, path: str, name: str) -> Binding | None:
        """If `name` is a module-level variable built from a constructor, what kind of object it is."""
        key = (path, name)
        if key in self._inst_cache:
            return self._inst_cache[key]
        constructor = self._vars.get(path, {}).get(name)
        if constructor is None or key in self._inst_active:
            return None
        self._inst_active.add(key)
        try:
            result: Binding | None = None
            targets = self.resolve_call(path, CallContext(None, "<module>"), constructor, allow_fallback=False)
            for target in targets:
                if target.kind == "class":
                    result = Binding("instance", path=target.path, symbol=target.qualname,
                                     symbol_kind="class", module_path=path)
                    break
                if target.kind == "external":
                    result = Binding("instance", external=target.external, module_path=path)
                    break
        finally:
            self._inst_active.discard(key)
        self._inst_cache[key] = result
        return result

    def name_binding(self, path: str, name: str) -> tuple[Binding, str] | None:
        """What a bare name means in a file, and how we know: (binding, 'local'|'import'|'star-import')."""
        if name in self.top_classes.get(path, {}):
            return Binding("symbol", path=path, symbol=self.top_classes[path][name], symbol_kind="class",
                           module_path=path), "local"
        if name in self.top_functions.get(path, {}):
            return Binding("symbol", path=path, symbol=self.top_functions[path][name], symbol_kind="function",
                           module_path=path), "local"
        instance = self.instance_binding(path, name)
        if instance is not None:
            return instance, "local"
        imported = self.binding_for(path, name)
        if imported is not None and imported.kind != "unknown":
            return imported, "import"
        for star in self._stars.get(path, []):
            module = self.resolve_target(path, star)
            if module.kind != "module":
                continue
            found = self._member(module.path, name, (), module.path)
            if found.kind in ("symbol", "module"):
                return found, "star-import"
        return None

    def resolve_ref(self, path: str, dotted: str) -> Binding | None:
        """Resolve a dotted reference (a base class, a decorator) by walking modules."""
        parts = dotted.split(".")
        named = self.name_binding(path, parts[0])
        if named is None:
            return None
        current = named[0]
        for part in parts[1:]:
            if current.kind == "module":
                current = self._member(current.path, part, (), current.module_path)
            elif current.kind == "external":
                return current
            elif current.kind == "instance" and current.external:
                # symbol_kind "instance" tells callers the type was inferred from an assignment
                return Binding("external", external=current.external, symbol_kind="instance",
                               module_path=current.module_path)
            else:
                return None
        return current

    # ---- classes
    def class_mro(self, path: str, qualname: str, _stack: frozenset = frozenset()) -> list[tuple[str, str]]:
        """The class followed by its resolvable base classes, depth-first, left to right."""
        key = (path, qualname)
        if key in self._mro_cache:
            return self._mro_cache[key]
        result = [key]
        if key in self.classes and key not in _stack:
            for base in self.classes[key].bases:
                ref = self.resolve_ref(path, base)
                if ref and ref.kind == "symbol" and ref.symbol_kind == "class":
                    for item in self.class_mro(ref.path, ref.symbol, _stack | {key}):
                        if item not in result:
                            result.append(item)
        if not _stack:
            self._mro_cache[key] = result
        return result

    def find_method(self, path: str, class_qualname: str, name: str) -> tuple[str, str, bool] | None:
        """(file, method qualname, defined_on_this_class) searching the class then its bases."""
        for index, (p, q) in enumerate(self.class_mro(path, class_qualname)):
            method = self.class_methods.get((p, q), {}).get(name)
            if method:
                return p, method, index == 0
        return None

    # ---- calls
    def resolve_call(
        self, path: str, ctx: CallContext, raw: str, *, allow_fallback: bool = True
    ) -> list[CallTarget]:
        """Resolve one call target as written in the source ("foo", "self.save", "utils.run")."""
        parts = raw.split(".")
        head = parts[0]
        if len(parts) == 1:
            return self._bare(path, ctx, head)
        if head in ("self", "cls") and ctx.caller_class:
            return self._self_call(path, ctx, parts[1:], allow_fallback)
        if head == "?":
            return self._unknown_receiver(path, ctx, parts[1:], allow_fallback)
        if head in ctx.local_names:
            return self._fallback(parts[-1], allow_fallback)
        named = self.name_binding(path, head)
        if named is None:
            return self._fallback(parts[-1], allow_fallback)  # head is a local variable or parameter
        current, how = named
        confidence = LIKELY if how == "star-import" else CERTAIN
        consumed = 1
        while consumed < len(parts) and current.kind == "module":
            current = self._member(current.path, parts[consumed], (), current.module_path)
            consumed += 1
        return self._finish_chain(current, parts[consumed:], confidence, consumed > 1, allow_fallback)

    def _finish_chain(
        self, cur: Binding, remaining: list[str], confidence: str, through_module: bool, allow_fallback: bool
    ) -> list[CallTarget]:
        if cur.kind == "external":
            return [CallTarget("external", external=cur.external, confidence=confidence, via="import")]
        if cur.kind == "instance":  # attribute or method call on a module-level object
            if cur.external:
                return [CallTarget("external", external=cur.external, confidence=LIKELY, via="instance")]
            if len(remaining) == 1:
                hit = self.find_method(cur.path, cur.symbol, remaining[0])
                if hit:
                    return [CallTarget("func", hit[0], hit[1], confidence=LIKELY, via="instance")]
            return self._fallback(remaining[-1], allow_fallback) if remaining else [
                CallTarget("unresolved", via="not-a-function")
            ]
        if cur.kind == "symbol" and cur.symbol_kind == "class":
            if not remaining:
                return [CallTarget("class", cur.path, cur.symbol, confidence=confidence, via="instantiation")]
            if len(remaining) == 1:
                hit = self.find_method(cur.path, cur.symbol, remaining[0])
                if hit:
                    p, q, own = hit
                    return [CallTarget("func", p, q, confidence=confidence if own else LIKELY,
                                       via="class-attr" if own else "inherited")]
            return self._fallback(remaining[-1], allow_fallback)
        if cur.kind == "symbol":  # a function
            if not remaining:
                return [CallTarget("func", cur.path, cur.symbol, confidence=confidence,
                                   via="module-attr" if through_module else "import")]
            return self._fallback(remaining[-1], allow_fallback)
        if remaining:
            return self._fallback(remaining[-1], allow_fallback)
        return [CallTarget("unresolved", via="not-a-function")]

    def _bare(self, path: str, ctx: CallContext, name: str) -> list[CallTarget]:
        if name in ctx.local_names:
            return [CallTarget("skipped", via="nested")]
        named = self.name_binding(path, name)
        if named is not None:
            binding, how = named
            confidence = LIKELY if how == "star-import" else CERTAIN
            if binding.kind == "symbol":
                if binding.symbol_kind == "class":
                    return [CallTarget("class", binding.path, binding.symbol, confidence=confidence, via="instantiation")]
                return [CallTarget("func", binding.path, binding.symbol, confidence=confidence, via=how)]
            if binding.kind == "external":
                return [CallTarget("external", external=binding.external, confidence=confidence, via="import")]
            return [CallTarget("unresolved", via="not-a-function")]
        if name in BUILTIN_NAMES:
            return [CallTarget("builtin")]
        return [CallTarget("unresolved", via="unknown-name")]

    def _self_call(self, path: str, ctx: CallContext, rest: list[str], allow_fallback: bool) -> list[CallTarget]:
        if len(rest) == 1 and ctx.caller_class:
            hit = self.find_method(path, ctx.caller_class, rest[0])
            if hit:
                p, q, own = hit
                return [CallTarget("func", p, q, confidence=CERTAIN if own else LIKELY,
                                   via="self" if own else "inherited")]
        return self._fallback(rest[-1], allow_fallback)

    def _unknown_receiver(self, path: str, ctx: CallContext, rest: list[str], allow_fallback: bool) -> list[CallTarget]:
        if rest == ["__init__"] and ctx.caller_name == "__init__" and ctx.caller_class:
            for p, q in self.class_mro(path, ctx.caller_class)[1:]:  # super().__init__()
                init = self.class_methods.get((p, q), {}).get("__init__")
                if init:
                    return [CallTarget("func", p, init, confidence=LIKELY, via="super")]
            return [CallTarget("skipped", via="dunder")]
        return self._fallback(rest[-1], allow_fallback)

    def _fallback(self, name: str, allow_fallback: bool) -> list[CallTarget]:
        """Receiver type unknown: match by method name, but only when that is safe."""
        if name.startswith("__") and name.endswith("__"):
            return [CallTarget("skipped", via="dunder")]
        if name in GENERIC_METHODS:
            return [CallTarget("skipped", via="generic")]
        if not allow_fallback:
            return [CallTarget("unresolved", via="no-fallback")]
        candidates = self.method_index.get(name, [])
        if not candidates:
            return [CallTarget("foreign", via="no-candidate")]  # probably a library method
        if len(candidates) > MAX_METHOD_GUESSES:
            return [CallTarget("ambiguous", via=f"{len(candidates)}-candidates")]
        return [CallTarget("func", p, q, confidence=GUESS, via="method-name") for p, q in candidates]


# ---------------------------------------------------------------- decorators
def decorator_reference(text: str) -> str | None:
    """'bp.route("/x")' -> 'bp.route'; '@login_required' -> 'login_required'; None if unnameable."""
    try:
        node = ast.parse(text.strip().lstrip("@"), mode="eval").body
    except (SyntaxError, ValueError, RecursionError):
        return None
    if isinstance(node, ast.Call):
        node = node.func
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))
