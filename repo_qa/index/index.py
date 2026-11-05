"""The searchable index of one repository: chunks, vectors, keyword index and graph.

    ingest result + graph + embedder  --build_index-->  RepoIndex  --save-->  folder on disk

Embedding is the slow step (every chunk goes through the local model), so it is guarded by
a time limit and reports progress, and vectors are cached by content: re-indexing a
repository only embeds chunks that are new or changed.

Saved layout (all plain files):
    index.json     summary and settings (written last, so a crash leaves no half-valid index)
    chunks.jsonl   one chunk per line
    graph.json     the knowledge graph
    vectors.npy + store.json   the vectors and the model that produced them (if embedded)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from common.backends import Embedder
from common.config import Config
from common.errors import IngestError
from repo_qa.graph.model import KnowledgeGraph
from repo_qa.ingest.models import Chunk, IngestResult

from .bm25 import BM25Index
from .store import VectorStore

log = logging.getLogger(__name__)
FORMAT_VERSION = 1
Progress = Callable[[int, int], None]


def keyword_text(chunk: Chunk) -> str:
    """What keyword search sees: the header twice (so path and symbol words weigh more) and the text."""
    return f"{chunk.context}\n{chunk.context}\n{chunk.text}"


@dataclass
class RepoIndex:
    repo_name: str
    source: str
    chunks: list[Chunk]
    graph: KnowledgeGraph
    store: VectorStore | None = None
    meta: dict = field(default_factory=dict)
    bm25: BM25Index = field(init=False)
    by_id: dict[str, int] = field(init=False)
    symbols: dict[str, list[int]] = field(init=False)  # function/class names -> chunk positions
    paths: dict[str, list[int]] = field(init=False)  # file paths and basenames -> chunk positions

    def __post_init__(self) -> None:
        self.by_id = {c.id: i for i, c in enumerate(self.chunks)}
        if len(self.by_id) != len(self.chunks):
            raise IngestError("Chunk ids are not unique; the index is damaged. Re-index.")
        if self.store is not None:
            missing = [cid for cid in self.store.ids if cid not in self.by_id]
            if missing or len(self.store) != len(self.chunks):
                raise IngestError("The vectors do not match the chunks. Re-index the repository.")
        self.bm25 = BM25Index([keyword_text(c) for c in self.chunks])
        self.symbols, self.paths = {}, {}
        for i, chunk in enumerate(self.chunks):
            if chunk.symbol and chunk.part == 1:
                for key in {chunk.symbol, chunk.symbol.rsplit(".", 1)[-1]}:
                    self.symbols.setdefault(key, []).append(i)
            for key in {chunk.path, chunk.path.rsplit("/", 1)[-1]}:
                self.paths.setdefault(key, []).append(i)

    @property
    def has_vectors(self) -> bool:
        return self.store is not None

    def chunk(self, chunk_id: str) -> Chunk:
        return self.chunks[self.by_id[chunk_id]]

    # ---------------------------------------------------------------- storage
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "chunks.jsonl.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(directory / "chunks.jsonl")
        self.graph.save(directory / "graph.json")
        if self.store is not None:
            self.store.save(directory)
        else:  # never leave vectors from an older build next to a keyword-only index
            for name in ("vectors.npy", "store.json"):
                (directory / name).unlink(missing_ok=True)
        meta = {**self.meta, "format": FORMAT_VERSION, "repo": self.repo_name, "source": self.source,
                "chunks": len(self.chunks), "has_vectors": self.store is not None,
                "embed_model": self.store.model if self.store else None}
        tmp_meta = directory / "index.json.tmp"
        tmp_meta.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        tmp_meta.replace(directory / "index.json")

    @classmethod
    def load(cls, directory: Path, *, expect_model: str | None = None) -> "RepoIndex":
        directory = Path(directory)
        try:
            meta = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IngestError(f"{directory} is not an index folder (no readable index.json). Build one first.") from exc
        if not isinstance(meta, dict) or meta.get("format") != FORMAT_VERSION:
            raise IngestError("The index was saved by an incompatible version. Re-index the repository.")
        try:
            with open(directory / "chunks.jsonl", encoding="utf-8") as handle:
                chunks = [Chunk.from_dict(json.loads(line)) for line in handle if line.strip()]
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise IngestError(f"The chunk file in {directory} is unreadable or damaged. Re-index.") from exc
        graph = KnowledgeGraph.load(directory / "graph.json")
        store = (
            VectorStore.load(directory, expect_model=expect_model) if meta.get("has_vectors") else None
        )
        return cls(meta.get("repo", directory.name), meta.get("source", ""), chunks, graph, store, meta)


# ---------------------------------------------------------------- building
def _cache_key(model: str, text: str) -> str:
    return hashlib.sha1(f"{model}\0{text}".encode("utf-8")).hexdigest()


def _load_cache(previous_dir: Path | None, cfg: Config, model: str) -> dict[str, list[float]]:
    """Vectors from an earlier index of this repository, keyed by content hash."""
    if previous_dir is None or not (previous_dir / "store.json").exists():
        return {}
    try:
        store = VectorStore.load(previous_dir, expect_model=model)
        with open(previous_dir / "chunks.jsonl", encoding="utf-8") as handle:
            old = {c.id: c for c in (Chunk.from_dict(json.loads(line)) for line in handle if line.strip())}
        cache = {}
        for row, chunk_id in enumerate(store.ids):
            if chunk_id in old:
                key = _cache_key(model, cfg.embed_doc_prefix + old[chunk_id].embed_text)
                cache[key] = store.vectors[row].tolist()
        return cache
    except (IngestError, OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        log.warning("Ignoring the previous index (cannot reuse its vectors): %s", exc)
        return {}


def build_index(
    result: IngestResult,
    graph: KnowledgeGraph,
    embedder: Embedder | None,
    cfg: Config,
    *,
    previous_dir: Path | None = None,
    progress: Progress | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RepoIndex:
    """Embed the chunks (unless embedder is None: keyword-only) and assemble the index."""
    chunks = list(result.chunks)
    store: VectorStore | None = None
    stats = {"embedded": 0, "reused": 0}
    if embedder is not None and chunks:
        vectors = _embed_chunks(chunks, embedder, cfg, previous_dir, progress, clock, stats)
        store = VectorStore.from_lists(vectors, [c.id for c in chunks], model=embedder.model_name)
    meta = {"vectors_embedded": stats["embedded"], "vectors_reused": stats["reused"],
            "embed_doc_prefix": cfg.embed_doc_prefix, "files": len(result.files)}
    return RepoIndex(result.repo_name, result.source, chunks, graph, store, meta)


def _embed_chunks(
    chunks: list[Chunk], embedder: Embedder, cfg: Config, previous_dir: Path | None,
    progress: Progress | None, clock: Callable[[], float], stats: dict[str, int],
) -> list[list[float]]:
    model = embedder.model_name
    texts = [cfg.embed_doc_prefix + c.embed_text for c in chunks]
    keys = [_cache_key(model, t) for t in texts]
    cache = _load_cache(previous_dir, cfg, model)
    vectors: list[list[float] | None] = [cache.get(k) for k in keys]
    todo = [i for i, v in enumerate(vectors) if v is None]
    stats["reused"] = len(chunks) - len(todo)
    deadline = clock() + cfg.index_timeout_s
    batch_size = cfg.embed_batch_size
    for start in range(0, len(todo), batch_size):
        if clock() > deadline:
            raise IngestError(
                f"Indexing exceeded {cfg.index_timeout_s:.0f}s after embedding {start} of {len(todo)} chunks. "
                "Use a smaller repository, or raise CHATBOT_INDEX_TIMEOUT_S."
            )
        batch = todo[start : start + batch_size]
        embedded = embedder.embed([texts[i] for i in batch])  # raises on timeout / server errors
        if len(embedded) != len(batch):
            raise IngestError(f"The embedder returned {len(embedded)} vectors for {len(batch)} chunks.")
        for i, vector in zip(batch, embedded):
            vectors[i] = vector
        stats["embedded"] += len(batch)
        if progress:
            progress(min(start + batch_size, len(todo)), len(todo))
    return [v for v in vectors if v is not None]  # every slot is filled by now
