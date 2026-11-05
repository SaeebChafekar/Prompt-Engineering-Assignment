"""Hybrid search over a RepoIndex.

Two searches run side by side and their rankings are merged:

    vector ..... embeds the question and finds chunks with similar MEANING
                 ("where is login handled?" -> the login view)
    keyword .... BM25 over words and identifier parts, finds EXACT names
                 ("verify_password" -> the chunk that defines it)

They are merged with reciprocal rank fusion: a chunk scores 1/(60 + rank) in each list it
appears in, so chunks ranked well by both float to the top and neither search's raw scores
(which are on different scales) matter. Two small boosts then help code questions:
a chunk whose function/class name appears in the question, and a chunk from a file the
question names. Finally, chunks from test files are down-weighted unless the question is
about tests, because tests mention everything and would crowd out the real code.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from common.backends import Embedder
from common.config import Config
from repo_qa.ingest.models import Chunk

from .bm25 import query_tokens
from .index import RepoIndex

log = logging.getLogger(__name__)

RRF_K = 60
SYMBOL_BOOST = 0.02  # a bit more than a rank-1 appearance in one list (1/61)
PATH_BOOST = 0.01
MODES = ("hybrid", "vector", "keyword")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_TEST_WORDS = frozenset({"test", "tests", "testing", "pytest", "unittest", "spec"})


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    vector_rank: int | None  # 1-based position in the vector results, if present
    keyword_rank: int | None  # 1-based position in the keyword results, if present
    signals: tuple[str, ...] = ()  # extra reasons: "name match", "file match", "test penalty"


class HybridSearcher:
    def __init__(self, index: RepoIndex, embedder: Embedder | None, cfg: Config) -> None:
        self.index = index
        self.embedder = embedder
        self.cfg = cfg
        self.note: str | None = None  # set when a search had to fall back (e.g. no vectors)

    def search(self, query: str, *, k: int | None = None, mode: str = "hybrid") -> list[SearchHit]:
        """The top-k chunks for a question, best first."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        text = query.strip()
        if not text:
            raise ValueError("The question is empty.")
        top_k = k if k is not None else self.cfg.search_top_k
        if top_k < 1:
            raise ValueError("k must be at least 1")
        self.note = None
        index, n = self.index, self.cfg.search_candidates

        use_vectors = mode in ("hybrid", "vector")
        if use_vectors and (self.embedder is None or not index.has_vectors):
            if mode == "vector":
                raise ValueError("Vector search needs an index built with embeddings and an embedder.")
            self.note = "No vectors available: using keyword search only."
            use_vectors = False

        vector_hits: list[tuple[int, float]] = []
        if use_vectors:
            assert self.embedder is not None and index.store is not None
            query_vector = self.embedder.embed([self.cfg.embed_query_prefix + text])[0]
            vector_hits = [
                (index.by_id[index.store.ids[row]], score) for row, score in index.store.search(query_vector, n)
            ]
        keyword_hits = index.bm25.search(query_tokens(text), n) if mode in ("hybrid", "keyword") else []

        scores: dict[int, float] = {}
        vector_rank = {pos: r for r, (pos, _) in enumerate(vector_hits, 1)}
        keyword_rank = {pos: r for r, (pos, _) in enumerate(keyword_hits, 1)}
        for ranks in (vector_rank, keyword_rank):
            for pos, rank in ranks.items():
                scores[pos] = scores.get(pos, 0.0) + 1.0 / (RRF_K + rank)

        signals: dict[int, list[str]] = {}
        for pos in self._named_chunks(text, index.symbols):
            scores[pos] = scores.get(pos, 0.0) + SYMBOL_BOOST
            signals.setdefault(pos, []).append("name match")
        for pos in self._named_chunks(text, index.paths):
            scores[pos] = scores.get(pos, 0.0) + PATH_BOOST
            signals.setdefault(pos, []).append("file match")

        if not _TEST_WORDS & set(query_tokens(text)) and self.cfg.test_file_penalty < 1:
            for pos in list(scores):
                if index.chunks[pos].is_test:
                    scores[pos] *= self.cfg.test_file_penalty
                    signals.setdefault(pos, []).append("test penalty")

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            SearchHit(index.chunks[pos], score, vector_rank.get(pos), keyword_rank.get(pos), tuple(signals.get(pos, ())))
            for pos, score in ranked
        ]

    @staticmethod
    def _named_chunks(query: str, table: dict[str, list[int]]) -> set[int]:
        """Chunks whose symbol or file is named in the question (exact, case-sensitive)."""
        found: set[int] = set()
        for word in _IDENT_RE.findall(query):
            word = word.strip(".")
            candidates = {word, *word.split(".")} if "." in word else {word}
            for candidate in candidates:
                found.update(table.get(candidate, ()))
        return found
