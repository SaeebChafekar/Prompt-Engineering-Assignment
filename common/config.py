"""Central configuration.

Defaults are tuned for an 8 GB GPU (RTX 4060 class) running one ~8B model at Q4.
Every field can be overridden with an environment variable named CHATBOT_<FIELD>,
for example:

    CHATBOT_CHAT_MODEL=qwen2.5-coder:7b CHATBOT_NUM_CTX=4096 python check_setup.py

To use the Gemini cloud backend instead of local Ollama:

    export GEMINI_API_KEY=...        # never put the key in code or commit it
    CHATBOT_BACKEND=gemini python check_setup.py

Embeddings always run locally through Ollama, so switching the chat backend never
forces a re-index.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from .errors import ConfigError

ENV_PREFIX = "CHATBOT_"
BACKENDS = ("ollama", "gemini")
GEMINI_THINKING_LEVELS = ("low", "medium", "high", "none")


@dataclass(frozen=True)
class RetrievalProfile:
    """How much context the RAG pipeline may gather per question."""

    top_k: int  # chunks returned by search
    graph_hops: int  # how far to walk the knowledge graph from the search hits
    context_tokens: int  # approximate token budget for the retrieved context


# Starting values; they are tuned against the eval set in a later phase.
LOCAL_PROFILE = RetrievalProfile(top_k=5, graph_hops=1, context_tokens=3500)
CLOUD_PROFILE = RetrievalProfile(top_k=12, graph_hops=2, context_tokens=16000)


@dataclass(frozen=True)
class Config:
    # --- Ollama server and models ---
    ollama_host: str = "http://localhost:11434"
    chat_model: str = "qwen3:8b"
    embed_model: str = "nomic-embed-text"

    # --- generation settings ---
    # Context window. It costs VRAM; if `ollama ps` shows a CPU/GPU split, lower this.
    num_ctx: int = 8192
    temperature: float = 0.1
    max_output_tokens: int = 1024
    # For "thinking" models (e.g. Qwen3). False = answer directly, which is much faster.
    think: bool = False
    # How long Ollama keeps the model in VRAM after the last request.
    keep_alive: str = "30m"

    # --- timeouts (seconds) ---
    connect_timeout_s: float = 5.0
    # Max wait for the next bytes from the server. The first request after loading a
    # model can take a while (model load + prompt processing), so this is generous.
    read_timeout_s: float = 120.0
    # Hard cap on one whole chat generation, however many chunks keep arriving.
    chat_total_timeout_s: float = 300.0
    embed_timeout_s: float = 60.0

    # --- embeddings and retries ---
    embed_batch_size: int = 32
    # Retries apply only to connection failures, never to timeouts (those are costly).
    max_retries: int = 1
    retry_backoff_s: float = 1.0

    # --- chat backend selection ---
    backend: str = "ollama"  # "ollama" (local) or "gemini" (cloud)

    # --- Gemini cloud backend (only used when backend == "gemini") ---
    # repr=False keeps the key out of logs, tracebacks and printed configs.
    gemini_api_key: str = field(default="", repr=False)
    # Model names change often; python check_setup.py lists what your key can use.
    gemini_model: str = "gemini-3.8-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    # Gemini 3.x models take a thinking LEVEL (not a token budget): low, medium or high.
    # "minimal" is not supported by 3.8 Flash and is rejected by the API. "low" is the
    # fastest allowed level. "none" sends no thinking setting (the model default is medium).
    gemini_thinking_level: str = "low"
    # Gemini 3 guidance is to leave temperature at the model default, so a negative value
    # (the default here) means "do not send it". Set 0..2 to force a value.
    gemini_temperature: float = -1.0
    # Thinking tokens count against the output limit, so this is more generous than the
    # local max_output_tokens.
    gemini_max_output_tokens: int = 4096
    gemini_read_timeout_s: float = 60.0
    gemini_total_timeout_s: float = 120.0
    # Client-side rate limit (requests per minute). 0 disables it. Free-tier limits are
    # low and change; check yours in AI Studio.
    gemini_rpm: int = 10
    # Longest the client will wait for a free rate-limit slot before giving up.
    gemini_rate_wait_s: float = 60.0
    gemini_max_retries: int = 3
    # Longest single wait between retries. A 429 asking for a longer wait than this is
    # treated as an exhausted quota (for example the daily limit) and not retried.
    gemini_max_backoff_s: float = 30.0
    # Optional hard caps for one process/session. 0 = unlimited.
    session_max_requests: int = 0
    session_max_tokens: int = 0

    # --- repository ingestion (RAG project) ---
    # Cloned repositories are stored under <data_dir>/repos.
    data_dir: str = "data"
    git_executable: str = "git"
    clone_timeout_s: float = 180.0
    # Hard cap on the whole ingest step (walk + read + parse + chunk).
    ingest_timeout_s: float = 300.0
    # Limits that reject repositories too big for a laptop demo.
    max_repo_files: int = 3000
    max_repo_bytes: int = 50_000_000
    max_file_bytes: int = 200_000
    # Chunk size cap in characters (about 800 tokens at 3000). Smaller chunks retrieve
    # more precisely; larger ones keep more context together.
    max_chunk_chars: int = 3000
    chunk_overlap_lines: int = 3
    # False drops test files from the index entirely. True keeps them, flagged as tests.
    include_tests: bool = True

    # --- indexing and search (RAG project) ---
    # nomic-embed-text was trained with task prefixes. For another embedding model, set both
    # to empty. (Environment values are stripped, so a trailing space cannot be set that way.)
    embed_doc_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "
    index_timeout_s: float = 1800.0  # hard cap on embedding a whole repository
    search_top_k: int = 8
    search_candidates: int = 40  # taken from each of vector and keyword search before fusion
    # Chunks from test files are multiplied by this unless the query is about tests (1 = off).
    test_file_penalty: float = 0.6

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ConfigError if any setting is unusable."""
        if not self.ollama_host.startswith(("http://", "https://")):
            raise ConfigError(
                f"ollama_host must start with http:// or https://, got {self.ollama_host!r}"
            )
        if not self.chat_model.strip():
            raise ConfigError("chat_model must not be empty")
        if not self.embed_model.strip():
            raise ConfigError("embed_model must not be empty")
        if self.num_ctx < 512:
            raise ConfigError(f"num_ctx must be at least 512, got {self.num_ctx}")
        if self.max_output_tokens < 1:
            raise ConfigError(f"max_output_tokens must be positive, got {self.max_output_tokens}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError(f"temperature must be between 0 and 2, got {self.temperature}")
        for name in (
            "connect_timeout_s",
            "read_timeout_s",
            "chat_total_timeout_s",
            "embed_timeout_s",
        ):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")
        if self.embed_batch_size < 1:
            raise ConfigError(f"embed_batch_size must be at least 1, got {self.embed_batch_size}")
        if self.max_retries < 0:
            raise ConfigError(f"max_retries must not be negative, got {self.max_retries}")
        if self.retry_backoff_s < 0:
            raise ConfigError(f"retry_backoff_s must not be negative, got {self.retry_backoff_s}")
        self._validate_backend()
        self._validate_ingest()

    def _validate_ingest(self) -> None:
        if not self.data_dir.strip():
            raise ConfigError("data_dir must not be empty")
        if not self.git_executable.strip():
            raise ConfigError("git_executable must not be empty")
        for name in ("clone_timeout_s", "ingest_timeout_s"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("max_repo_files", "max_repo_bytes", "max_file_bytes"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.max_chunk_chars < 200:
            raise ConfigError(f"max_chunk_chars must be at least 200, got {self.max_chunk_chars}")
        if self.index_timeout_s <= 0:
            raise ConfigError(f"index_timeout_s must be positive, got {self.index_timeout_s}")
        if self.search_top_k < 1 or self.search_candidates < 1:
            raise ConfigError("search_top_k and search_candidates must be at least 1")
        if not 0 < self.test_file_penalty <= 1:
            raise ConfigError(f"test_file_penalty must be in (0, 1], got {self.test_file_penalty}")
        if not 0 <= self.chunk_overlap_lines <= 20:
            raise ConfigError(
                f"chunk_overlap_lines must be between 0 and 20, got {self.chunk_overlap_lines}"
            )

    def _validate_backend(self) -> None:
        if self.backend not in BACKENDS:
            raise ConfigError(f"backend must be one of {', '.join(BACKENDS)}, got {self.backend!r}")
        for name in ("gemini_read_timeout_s", "gemini_total_timeout_s", "gemini_rate_wait_s"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")
        for name in (
            "gemini_rpm",
            "gemini_max_retries",
            "gemini_max_backoff_s",
            "session_max_requests",
            "session_max_tokens",
        ):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must not be negative, got {getattr(self, name)}")
        if not self.gemini_model.strip():
            raise ConfigError("gemini_model must not be empty")
        if self.gemini_thinking_level not in GEMINI_THINKING_LEVELS:
            hint = (
                " ('minimal' is not supported by Gemini 3.8 Flash)"
                if self.gemini_thinking_level == "minimal" else ""
            )
            raise ConfigError(
                f"gemini_thinking_level must be one of {', '.join(GEMINI_THINKING_LEVELS)}, "
                f"got {self.gemini_thinking_level!r}{hint}"
            )
        if self.gemini_temperature > 2.0:
            raise ConfigError(f"gemini_temperature must be at most 2 (or negative to omit), got {self.gemini_temperature}")
        if self.gemini_max_output_tokens < 1:
            raise ConfigError(f"gemini_max_output_tokens must be positive, got {self.gemini_max_output_tokens}")
        base = self.gemini_base_url
        # Plain http is allowed only for local test servers, so the key never travels
        # unencrypted over a real network.
        if not (
            base.startswith("https://")
            or base.startswith("http://127.0.0.1")
            or base.startswith("http://localhost")
        ):
            raise ConfigError(f"gemini_base_url must use https://, got {base!r}")
        if self.backend == "gemini" and not self.gemini_api_key.strip():
            raise ConfigError(
                "backend is 'gemini' but no API key is set. Create a key in Google AI Studio "
                "and export it as GEMINI_API_KEY (do not put it in code or commit it)."
            )

    @property
    def retrieval_profile(self) -> RetrievalProfile:
        """Context limits for the active backend: small for local, larger for cloud."""
        return CLOUD_PROFILE if self.backend == "gemini" else LOCAL_PROFILE

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Config":
        """Build a Config from defaults plus CHATBOT_* environment variables."""
        env = os.environ if environ is None else environ
        defaults = cls()
        overrides: dict[str, Any] = {}
        for f in fields(cls):
            var = ENV_PREFIX + f.name.upper()
            raw = env.get(var)
            if raw is None or not raw.strip():
                continue
            overrides[f.name] = _parse_value(var, raw, getattr(defaults, f.name))
        # Google's own SDKs and docs use GEMINI_API_KEY, so honor it as well.
        if "gemini_api_key" not in overrides:
            standard = env.get("GEMINI_API_KEY", "").strip()
            if standard:
                overrides["gemini_api_key"] = standard
        return cls(**overrides)


def _parse_value(var: str, raw: str, default: Any) -> Any:
    """Convert an environment string to the same type as the field's default."""
    text = raw.strip()
    try:
        if isinstance(default, bool):  # must come before int: bool is a subclass of int
            lowered = text.lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off"):
                return False
            raise ValueError(text)
        if isinstance(default, int):
            return int(text)
        if isinstance(default, float):
            return float(text)
        return text
    except ValueError as exc:
        raise ConfigError(
            f"Environment variable {var}={raw!r} is not a valid {type(default).__name__}"
        ) from exc
