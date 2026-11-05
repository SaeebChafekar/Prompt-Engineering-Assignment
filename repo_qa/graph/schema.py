"""Vocabulary of the knowledge graph: node types, edge types, ids and confidence.

Nodes
    File ........ a source, doc or config file                  id  file:<path>
    Class ....... a class                                       id  class:<path>::<qualname>
    Function .... a function or method (nested ones are folded) id  func:<path>::<qualname>
    DocSection .. a section of a Markdown file                  id  doc:<path>:<start line>
    External .... a library outside the repository              id  ext:<top-level name>

Edges (every edge has a confidence: certain, likely or guess)
    CONTAINS ..... File -> Class/Function/DocSection, Class -> method or nested Class
    IMPORTS ...... File -> File (inside the repo) or File -> External
    CALLS ........ Function/File -> Function or Class (calling a class means creating one)
    USES ......... Function/File -> External (calls into a third-party or standard library)
    INHERITS ..... Class -> Class or External
    DECORATED_BY . Function/Class -> Function or External (e.g. @login_required)
    DOCUMENTS .... DocSection -> File/Class/Function it mentions
"""

from __future__ import annotations

NODE_FILE = "File"
NODE_CLASS = "Class"
NODE_FUNCTION = "Function"
NODE_DOC = "DocSection"
NODE_EXTERNAL = "External"

EDGE_CONTAINS = "CONTAINS"
EDGE_IMPORTS = "IMPORTS"
EDGE_CALLS = "CALLS"
EDGE_USES = "USES"
EDGE_INHERITS = "INHERITS"
EDGE_DECORATED_BY = "DECORATED_BY"
EDGE_DOCUMENTS = "DOCUMENTS"

NODE_TYPES = (NODE_FILE, NODE_CLASS, NODE_FUNCTION, NODE_DOC, NODE_EXTERNAL)
EDGE_TYPES = (
    EDGE_CONTAINS, EDGE_IMPORTS, EDGE_CALLS, EDGE_USES,
    EDGE_INHERITS, EDGE_DECORATED_BY, EDGE_DOCUMENTS,
)

# How sure we are that an edge is real. Python is dynamic, so many edges are inferred.
CERTAIN = "certain"  # resolved through definitions and imports
LIKELY = "likely"  # resolved, but through inheritance, re-exports or star imports
GUESS = "guess"  # matched only by method name (receiver type unknown)

CONFIDENCE_RANK = {GUESS: 1, LIKELY: 2, CERTAIN: 3}
# Used to score graph neighbours during retrieval: a path is only as good as its weakest edge.
CONFIDENCE_WEIGHT = {CERTAIN: 1.0, LIKELY: 0.7, GUESS: 0.3}


def file_id(path: str) -> str:
    return f"file:{path}"


def class_id(path: str, qualname: str) -> str:
    return f"class:{path}::{qualname}"


def function_id(path: str, qualname: str) -> str:
    return f"func:{path}::{qualname}"


def doc_id(path: str, start_line: int) -> str:
    return f"doc:{path}:{start_line}"


def external_id(name: str) -> str:
    return f"ext:{name}"


def best_confidence(a: str, b: str) -> str:
    """The more certain of two confidence levels."""
    return a if CONFIDENCE_RANK[a] >= CONFIDENCE_RANK[b] else b


def at_least(confidence: str, minimum: str) -> bool:
    """True if `confidence` is as certain as `minimum` or more."""
    return CONFIDENCE_RANK[confidence] >= CONFIDENCE_RANK[minimum]
