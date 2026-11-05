"""A small vector store: a normalized float32 matrix searched by cosine similarity.

A repository has thousands of chunks, not millions, so exact brute-force search with
numpy takes milliseconds and needs no database. Vectors are normalized once on the way in,
so a dot product is the cosine similarity.

The store remembers which embedding model produced its vectors and refuses to load for a
different one: vectors from different models are not comparable, and silently mixing them
would return nonsense. The fix is to re-index, and the error says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from common.errors import IngestError, InvalidOutputError

FORMAT_VERSION = 1


class VectorStore:
    def __init__(self, vectors: np.ndarray, ids: list[str], *, model: str) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError("vectors must be a 2-D array with one row per id")
        self.vectors = vectors
        self.ids = ids
        self.model = model

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def from_lists(cls, vectors: Sequence[Sequence[float]], ids: list[str], *, model: str) -> "VectorStore":
        """Build a store from raw embedding vectors (validates shape, normalizes each row)."""
        if len(vectors) != len(ids):
            raise InvalidOutputError(f"Got {len(vectors)} vectors for {len(ids)} chunks")
        if not vectors:
            return cls(np.zeros((0, 0), dtype=np.float32), [], model=model)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise InvalidOutputError("Embedding vectors have inconsistent or empty dimensions")
        if not np.isfinite(matrix).all():
            raise InvalidOutputError("Embedding vectors contain NaN or infinite values")
        return cls(_normalize(matrix), list(ids), model=model)

    def query_vector(self, vector: Sequence[float]) -> np.ndarray:
        q = np.asarray(vector, dtype=np.float32)
        if q.ndim != 1 or (self.dim and q.shape[0] != self.dim):
            raise InvalidOutputError(
                f"The query embedding has {q.shape[-1] if q.ndim else 0} dimensions but the index "
                f"has {self.dim}. Was the index built with a different embedding model?"
            )
        if not np.isfinite(q).all():
            raise InvalidOutputError("The query embedding contains NaN or infinite values")
        return _normalize(q.reshape(1, -1))[0]

    def search(self, vector: Sequence[float], limit: int) -> list[tuple[int, float]]:
        """Best-first (row position, cosine similarity) pairs."""
        if not len(self) or limit < 1:
            return []
        scores = self.vectors @ self.query_vector(vector)
        limit = min(limit, len(scores))
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top]

    # ---------------------------------------------------------------- storage
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors, allow_pickle=False)
        meta = {"format": FORMAT_VERSION, "model": self.model, "dim": self.dim, "count": len(self), "ids": self.ids}
        tmp = directory / "store.json.tmp"
        tmp.write_text(json.dumps(meta), encoding="utf-8")
        tmp.replace(directory / "store.json")  # written last: a crash leaves no half-valid store

    @classmethod
    def load(cls, directory: Path, *, expect_model: str | None = None) -> "VectorStore":
        try:
            meta = json.loads((directory / "store.json").read_text(encoding="utf-8"))
            vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise IngestError(f"Cannot read the vector store in {directory}: {type(exc).__name__}") from exc
        if not isinstance(meta, dict) or meta.get("format") != FORMAT_VERSION:
            raise IngestError("The vector store was saved by an incompatible version. Re-index the repository.")
        if expect_model is not None and meta.get("model") != expect_model:
            raise IngestError(
                f"This index was built with embedding model {meta.get('model')!r} but {expect_model!r} is "
                "configured. Re-index the repository (vectors from different models cannot be compared)."
            )
        ids = meta.get("ids")
        if not isinstance(ids, list) or vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise IngestError("The vector store is damaged (vectors and ids do not match). Re-index.")
        return cls(vectors.astype(np.float32, copy=False), ids, model=str(meta.get("model")))


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # an all-zero vector stays zero and matches nothing
    return (matrix / norms).astype(np.float32, copy=False)
