"""Operations behind the web page (and reusable from the command line).

    index_repository(...)  ingest -> graph -> embed -> save, with progress callbacks
    list_indexes(cfg)      what has been indexed already (data/index/*)
    RepoChat               an open index plus a conversation, ready to answer questions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from common.backends import ChatBackend, Embedder, create_chat_client, create_embedder
from common.config import Config
from common.errors import IngestError
from repo_qa.graph.build import build_graph
from repo_qa.index.index import RepoIndex, build_index
from repo_qa.index.search import HybridSearcher
from repo_qa.ingest.pipeline import ingest_repo
from repo_qa.rag.answer import Answer, Answerer
from repo_qa.rag.retrieve import Retriever

log = logging.getLogger(__name__)
MAX_HISTORY = 6


def default_index_dir(cfg: Config, repo_name: str) -> Path:
    return Path(cfg.data_dir) / "index" / repo_name.replace("/", "__")


@dataclass(frozen=True)
class IndexInfo:
    name: str  # folder name, e.g. miguelgrinberg__microblog
    path: Path
    repo: str
    chunks: int
    has_vectors: bool
    embed_model: str | None


def list_indexes(cfg: Config) -> list[IndexInfo]:
    """Indexes that already exist under data/index. Unreadable folders are skipped."""
    root = Path(cfg.data_dir) / "index"
    found: list[IndexInfo] = []
    if not root.is_dir():
        return found
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            meta = json.loads((folder / "index.json").read_text(encoding="utf-8"))
            found.append(IndexInfo(folder.name, folder, str(meta.get("repo", folder.name)), int(meta.get("chunks", 0)),
                                   bool(meta.get("has_vectors")), meta.get("embed_model")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            log.warning("Skipping %s: not a readable index", folder)
    return found


@dataclass
class IndexReport:
    path: Path
    repo_name: str
    summary: str  # the ingestion summary text
    graph_counts: dict
    call_stats: dict
    chunks: int
    has_vectors: bool
    vectors_embedded: int = 0
    vectors_reused: int = 0
    warnings: list[str] = field(default_factory=list)


def index_repository(
    source: str,
    cfg: Config,
    *,
    refresh: bool = False,
    keyword_only: bool = False,
    out_dir: Path | None = None,
    embedder_factory: Callable[[Config], Embedder] = create_embedder,
    progress: Callable[[int, int], None] | None = None,
    stage: Callable[[str], None] | None = None,
) -> IndexReport:
    """Index a GitHub URL or local folder and save it. Raises ChatbotError subclasses on failure."""
    say = stage or (lambda _msg: None)
    say("Reading the repository ...")
    result = ingest_repo(source, cfg, refresh=refresh)
    say(f"Building the knowledge graph from {len(result.files)} files ...")
    graph, report = build_graph(result)
    target = out_dir or default_index_dir(cfg, result.repo_name)

    embedder: Embedder | None = None
    if not keyword_only:
        embedder = embedder_factory(cfg)
        try:
            embedder.health_check()  # fail early with a helpful message if Ollama or the model is missing
        except Exception:
            embedder.close()
            raise
        say(f"Embedding {len(result.chunks)} chunks with {embedder.model_name} (the first batch can be slow) ...")
    try:
        index = build_index(result, graph, embedder, cfg, previous_dir=target, progress=progress)
    finally:
        if embedder is not None:
            embedder.close()
    say("Saving the index ...")
    index.save(target)
    return IndexReport(
        path=target, repo_name=result.repo_name, summary=result.summary(), graph_counts=graph.counts(),
        call_stats=report.to_dict()["calls"], chunks=len(index.chunks), has_vectors=index.has_vectors,
        vectors_embedded=int(index.meta.get("vectors_embedded", 0)),
        vectors_reused=int(index.meta.get("vectors_reused", 0)), warnings=list(result.warnings),
    )


class RepoChat:
    """An opened index with its search, graph and chat model, plus the running conversation."""

    def __init__(self, index: RepoIndex, answerer: Answerer, chat: ChatBackend, embedder: Embedder | None) -> None:
        self.index = index
        self.answerer = answerer
        self._chat = chat
        self._embedder = embedder
        self.history: list[tuple[str, str]] = []

    @classmethod
    def open(
        cls,
        index_dir: Path,
        cfg: Config,
        *,
        chat_factory: Callable[[Config], ChatBackend] = create_chat_client,
        embedder_factory: Callable[[Config], Embedder] = create_embedder,
    ) -> "RepoChat":
        """Load an index and connect to the models. Raises ChatbotError subclasses if anything is wrong."""
        index = RepoIndex.load(Path(index_dir))
        embedder: Embedder | None = None
        chat: ChatBackend | None = None
        try:
            if index.has_vectors:
                embedder = embedder_factory(cfg)
                embedder.health_check()
                assert index.store is not None
                if index.store.model != embedder.model_name:
                    raise IngestError(
                        f"The index was built with embedding model {index.store.model!r} but "
                        f"{embedder.model_name!r} is configured. Re-index the repository."
                    )
            chat = chat_factory(cfg)
            chat.health_check()
        except Exception:
            for resource in (chat, embedder):
                if resource is not None:
                    resource.close()
            raise
        answerer = Answerer(Retriever(index, HybridSearcher(index, embedder, cfg), cfg), chat, cfg)
        return cls(index, answerer, chat, embedder)

    def ask(
        self,
        question: str,
        *,
        use_graph: bool = True,
        mode: str = "hybrid",
        effort: str = "medium",
        on_token: Callable[[str], None] | None = None,
    ) -> Answer:
        answer = self.answerer.ask(question, self.history, use_graph=use_graph, mode=mode, effort=effort, on_token=on_token)
        self.history.append((question.strip(), answer.text))
        del self.history[:-MAX_HISTORY]
        return answer

    def clear(self) -> None:
        self.history.clear()

    def close(self) -> None:
        for resource in (self._chat, self._embedder):
            if resource is not None:
                resource.close()
