"""Backend interfaces and factories.

The rest of the code base depends only on the two small interfaces below, never on
OllamaClient or GeminiClient directly:

    ChatBackend ... chat, streaming chat, JSON chat, health check
    Embedder ...... turns texts into vectors

Chat and embeddings are chosen independently. Chat can be local (Ollama) or cloud
(Gemini), selected with CHATBOT_BACKEND. Embeddings always run locally through Ollama:
vectors from different embedding models are incompatible, so keeping them fixed lets
you switch the chat backend at any time without re-indexing a repository.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from .config import Config
from .errors import ConfigError
from .gemini_client import GeminiClient
from .llm_client import Message, OllamaClient


class ChatBackend(Protocol):
    """What the chatbots need from any chat model backend."""

    backend_name: str

    @property
    def model_name(self) -> str: ...

    def health_check(self) -> None:
        """Raise a ChatbotError if the backend is unreachable, rejects us, or lacks the model."""

    def list_models(self) -> list[str]: ...

    def chat(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        total_timeout_s: float | None = None,
    ) -> str: ...

    def chat_json(
        self,
        messages: Sequence[Message],
        schema: dict[str, Any],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        total_timeout_s: float | None = None,
    ) -> Any: ...

    def chat_stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        total_timeout_s: float | None = None,
    ) -> Any: ...

    def close(self) -> None: ...


class Embedder(Protocol):
    """What the indexing and retrieval code need to embed text."""

    @property
    def model_name(self) -> str: ...

    def health_check(self) -> None:
        """Raise a ChatbotError if the embedding model cannot be used."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def close(self) -> None: ...


class OllamaEmbedder:
    """Local embeddings through Ollama (the only embedding backend for now)."""

    def __init__(self, config: Config | None = None) -> None:
        self._client = OllamaClient(config)

    @property
    def model_name(self) -> str:
        return self._client.config.embed_model

    def health_check(self) -> None:
        self._client.require_models(self._client.config.embed_model)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._client.embed(texts)

    def close(self) -> None:
        self._client.close()


def create_chat_client(config: Config | None = None) -> ChatBackend:
    """Build the chat backend selected by config.backend."""
    cfg = config if config is not None else Config.from_env()
    if cfg.backend == "ollama":
        return OllamaClient(cfg)
    if cfg.backend == "gemini":
        return GeminiClient(cfg)
    raise ConfigError(f"Unknown backend {cfg.backend!r}")  # unreachable if Config validated


def create_embedder(config: Config | None = None) -> Embedder:
    """Build the embedder. Always local, whichever chat backend is active."""
    return OllamaEmbedder(config if config is not None else Config.from_env())
