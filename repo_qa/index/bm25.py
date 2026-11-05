"""Keyword search (BM25) with a tokenizer that understands code.

Meaning-based (vector) search is good at "where is login handled?" but weak at exact
identifiers like `verify_password`. Keyword search is the opposite. The tokenizer here
keeps whole identifiers AND their parts, so a chunk containing `verifyToken` or
`verify_token` matches queries for "verify_token", "verifyToken", "verify" and "token".
"""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

# Dropped from queries only (documents keep every word; IDF already down-weights them).
QUERY_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its of on or
    that the this to was what when where which who why will with you your me my tell show
    find about get give list explain""".split()
)


def split_identifier(word: str) -> list[str]:
    """'verifyToken' -> ['verify', 'token']; 'HTTPServer' -> ['http', 'server']."""
    parts: list[str] = []
    for piece in word.split("_"):
        parts.extend(p.lower() for p in _PART_RE.findall(piece))
    return parts


def tokenize(text: str) -> list[str]:
    """Lower-cased tokens: each identifier whole, plus its parts when it has several."""
    tokens: list[str] = []
    for word in _WORD_RE.findall(text):
        whole = word.lower().strip("_")
        if whole:
            tokens.append(whole)
        parts = split_identifier(word)
        if len(parts) > 1 or (parts and parts[0] != whole):
            tokens.extend(parts)
    return tokens


def query_tokens(query: str) -> list[str]:
    """Tokens for a question: like tokenize(), minus common English words."""
    return [t for t in tokenize(query) if t not in QUERY_STOPWORDS and len(t) > 1]


class BM25Index:
    """Okapi BM25 over a fixed list of documents (documents are identified by position)."""

    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.size = len(documents)
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._lengths: list[int] = []
        for doc_index, text in enumerate(documents):
            tokens = tokenize(text)
            self._lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                self._postings.setdefault(term, []).append((doc_index, count))
        self._avg_length = (sum(self._lengths) / self.size) if self.size else 0.0

    def _idf(self, term: str) -> float:
        df = len(self._postings.get(term, ()))
        return math.log(1.0 + (self.size - df + 0.5) / (df + 0.5))

    def search(self, tokens: list[str], limit: int) -> list[tuple[int, float]]:
        """Best-first (document position, score) pairs for the given query tokens."""
        if not tokens or not self.size or limit < 1:
            return []
        scores: dict[int, float] = {}
        for term in set(tokens):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for doc_index, tf in postings:
                norm = self.k1 * (1 - self.b + self.b * self._lengths[doc_index] / (self._avg_length or 1.0))
                scores[doc_index] = scores.get(doc_index, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:limit]
