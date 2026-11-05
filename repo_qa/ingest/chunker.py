"""Cut files into chunks: the units that get embedded, retrieved and cited.

Code is chunked by structure, not by character count, because a function is the natural
answer to "where is X handled?":

    Python .......... one chunk per function/method, one summary chunk per class
                      (signature, docstring, method list), plus chunks for the module-
                      level code in between (imports, constants, scripts)
    Markdown ........ one chunk per heading section, with the heading path as context
    everything else . fixed-size windows with a few lines of overlap

Every chunk carries a short `context` header (file path, symbol name) that is embedded
together with the text, so a question mentioning "auth" can match src/auth/jwt.py even
when the function body never uses that word.

Chunks that exceed max_chunk_chars are split into parts. Line numbers always refer to
real lines of the original file, so citations stay accurate.
"""

from __future__ import annotations

import re
import textwrap

from common.config import Config

from .models import Chunk, ClassInfo, FunctionInfo, ParsedFile, SourceFile

MIN_MODULE_CHUNK_CHARS = 25  # keeps `if __name__ == "__main__": main()`, drops lone `x = 1`
MIN_DOC_SECTION_CHARS = 3  # only drops heading-only sections; short real content is kept
TRUNCATION_MARK = " ...[line truncated]"
MAX_DOC_IN_CLASS_CHUNK = 500
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def chunk_file(sf: SourceFile, parsed: ParsedFile | None, cfg: Config) -> list[Chunk]:
    """Cut one file into chunks. `parsed` is the Python parse result for .py files."""
    builder = _Builder(sf, cfg)
    if sf.language == "python" and parsed is not None and parsed.error is None:
        _chunk_python(builder, parsed)
    elif sf.language == "markdown":
        _chunk_markdown(builder)
    else:
        kind = {"code": "block", "doc": "doc", "config": "config"}.get(sf.kind, "block")
        builder.add(kind, sf.path.rsplit("/", 1)[-1], 1, len(builder.lines), sf.text, f"File: {sf.path}")
    return builder.chunks


class _Builder:
    def __init__(self, sf: SourceFile, cfg: Config) -> None:
        self.sf = sf
        self.cfg = cfg
        self.lines = sf.text.splitlines()
        self.chunks: list[Chunk] = []
        self._ids: set[str] = set()

    def add(
        self,
        kind: str,
        name: str,
        start: int,
        end: int,
        text: str,
        header: str,
        *,
        symbol: str | None = None,
        split: bool = True,
    ) -> None:
        """Add a chunk; long source text is split into overlapping parts.

        Pass split=False for synthesized text (class summaries) whose lines do not map
        one-to-one onto the file; it is clipped instead of split.
        """
        text = text.rstrip("\n")  # a trailing newline would make the last line number off by one
        if not text.strip():
            return
        if split:
            pieces = _split_text(text, start, self.cfg.max_chunk_chars, self.cfg.chunk_overlap_lines)
        else:
            pieces = [(text[: self.cfg.max_chunk_chars], start, end)]
        total = len(pieces)
        for number, (piece, p_start, p_end) in enumerate(pieces, 1):
            context = header + (f"\nPart: {number} of {total}" if total > 1 else "")
            self.chunks.append(
                Chunk(
                    id=self._unique_id(f"{self.sf.path}:{p_start}-{p_end}"),
                    kind=kind,
                    path=self.sf.path,
                    start_line=p_start,
                    end_line=p_end,
                    name=name,
                    language=self.sf.language,
                    text=piece,
                    context=context,
                    symbol=symbol,
                    is_test=self.sf.is_test,
                    part=number,
                    parts=total,
                )
            )

    def _unique_id(self, base: str) -> str:
        candidate, n = base, 1
        while candidate in self._ids:
            n += 1
            candidate = f"{base}#{n}"
        self._ids.add(candidate)
        return candidate


# ---------------------------------------------------------------- python
def _chunk_python(b: _Builder, parsed: ParsedFile) -> None:
    path = b.sf.path
    functions = {f.qualname: f for f in parsed.functions}
    total_lines = len(b.lines)

    # 1. functions and methods (nested functions stay inside their parent's chunk)
    for fn in parsed.functions:
        if fn.is_nested:
            continue
        start, end = fn.start_line, min(fn.end_line, total_lines)
        source = textwrap.dedent("\n".join(b.lines[start - 1 : end]))
        owner = f" (method of {fn.parent_class})" if fn.parent_class else ""
        b.add(
            "function", fn.qualname, start, end, source,
            f"File: {path}\nFunction: {fn.qualname}{owner}", symbol=fn.qualname,
        )

    # 2. one summary chunk per class: signature, docstring, list of methods
    for cls in parsed.classes:
        if cls.in_function:
            continue
        b.add(
            "class", cls.qualname, cls.start_line, min(cls.end_line, total_lines),
            _class_summary(cls, functions), f"File: {path}\nClass: {cls.qualname}",
            symbol=cls.qualname, split=False,
        )

    # 3. module-level code: the stretches of the file outside any top-level definition
    covered = [False] * (total_lines + 2)
    for info in (*parsed.classes, *parsed.functions):
        if "." in info.qualname:  # nested definitions lie inside a top-level one
            continue
        for line_no in range(info.start_line, min(info.end_line, total_lines) + 1):
            covered[line_no] = True
    line_no = 1
    while line_no <= total_lines:
        if covered[line_no]:
            line_no += 1
            continue
        run_start = line_no
        while line_no <= total_lines and not covered[line_no]:
            line_no += 1
        run_end = line_no - 1
        while run_start <= run_end and not b.lines[run_start - 1].strip():
            run_start += 1
        while run_end >= run_start and not b.lines[run_end - 1].strip():
            run_end -= 1
        if run_start > run_end:
            continue
        text = "\n".join(b.lines[run_start - 1 : run_end])
        if len(text.strip()) < MIN_MODULE_CHUNK_CHARS:
            continue
        b.add(
            "module", parsed.module or path, run_start, run_end, text,
            f"File: {path}\nModule-level code (imports, constants, setup)",
        )

    b.chunks.sort(key=lambda c: (c.start_line, c.end_line, c.kind))


def _class_summary(cls: ClassInfo, functions: dict[str, FunctionInfo]) -> str:
    bases = f"({', '.join(cls.bases)})" if cls.bases else ""
    parts = [*[f"@{d}" for d in cls.decorators], f"class {cls.name}{bases}:"]
    if cls.docstring:
        parts.append(textwrap.indent(f'"""{cls.docstring[:MAX_DOC_IN_CLASS_CHUNK]}"""', "    "))
    signatures = [functions[m].signature for m in cls.methods if m in functions]
    if signatures:
        parts.append("    # methods:")
        budget = 1500
        for index, signature in enumerate(signatures):
            if budget <= 0:
                parts.append(f"    # ... and {len(signatures) - index} more methods")
                break
            parts.append(f"    # {signature}")
            budget -= len(signature) + 6
    return "\n".join(parts)


# ---------------------------------------------------------------- markdown
def _chunk_markdown(b: _Builder) -> None:
    path = b.sf.path
    stack: list[tuple[int, str]] = []
    title = path.rsplit("/", 1)[-1]
    start = 1
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        lines = list(current)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return
        body = lines[1:] if _HEADING_RE.match(lines[0]) else lines
        if len("\n".join(body).strip()) < MIN_DOC_SECTION_CHARS:
            return
        b.add(
            "doc", title, start, start + len(lines) - 1, "\n".join(lines),
            f"File: {path}\nSection: {title}",
        )

    for line_no, line in enumerate(b.lines, 1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING_RE.match(line)
        if match:
            flush()
            level, heading = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            title = " > ".join(h for _, h in stack)
            start, current = line_no, [line]
        else:
            current.append(line)
    flush()


# ---------------------------------------------------------------- splitting
def _clip(line: str, max_chars: int) -> str:
    return line if len(line) <= max_chars else line[:max_chars] + TRUNCATION_MARK


def _split_text(text: str, first_line: int, max_chars: int, overlap: int) -> list[tuple[str, int, int]]:
    """Split text into pieces of at most ~max_chars, cutting only at line boundaries.

    Returns (piece, first_line_number, last_line_number) tuples. Consecutive pieces share
    `overlap` lines so a statement cut at a boundary appears whole in one of them.
    """
    if len(text) <= max_chars:
        return [(text, first_line, first_line + text.count("\n"))]
    lines = [_clip(line, max_chars) for line in text.split("\n")]
    pieces: list[tuple[str, int, int]] = []
    i, n = 0, len(lines)
    while i < n:
        j, size = i, 0
        while j < n and (j == i or size + len(lines[j]) + 1 <= max_chars):
            size += len(lines[j]) + 1
            j += 1
        pieces.append(("\n".join(lines[i:j]), first_line + i, first_line + j - 1))
        if j >= n:
            break
        i = max(j - overlap, i + 1)  # always move forward, even with a large overlap
    return pieces
