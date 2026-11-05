"""Exception hierarchy shared by both chatbot projects.

Every error raised on purpose by this code base derives from ChatbotError, so the
UI layer can catch one base class and show a clean message instead of a traceback.
"""

from __future__ import annotations


class ChatbotError(Exception):
    """Base class for all expected, user-presentable errors."""


# --- configuration ---------------------------------------------------------
class ConfigError(ChatbotError):
    """A setting is missing, malformed, or out of range."""


# --- LLM / Ollama ----------------------------------------------------------
class LLMError(ChatbotError):
    """Generic failure while talking to the model server."""


class BackendUnavailableError(LLMError):
    """The model backend cannot be reached (server down, no internet, wrong address)."""


class OllamaUnavailableError(BackendUnavailableError):
    """The Ollama server cannot be reached (not running, wrong host or port)."""


class AuthenticationError(LLMError):
    """A cloud API rejected the key (missing, invalid, revoked, or restricted)."""


class RateLimitError(LLMError):
    """A rate limit or quota was hit, either locally or reported by the cloud API.

    retry_after is the suggested wait in seconds when the server or limiter gave one.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class BudgetExceededError(LLMError):
    """The per-session request or token budget set in the config has been used up."""


class ContentBlockedError(LLMError):
    """The cloud API refused or stopped the reply for safety or policy reasons."""


class ModelNotFoundError(LLMError):
    """The requested model is not installed in Ollama."""


class LLMTimeoutError(LLMError):
    """The model server did not answer within the configured time limit."""


class InvalidOutputError(LLMError):
    """The server replied, but the content was empty, malformed, or the wrong shape."""


# --- ingestion (used by the RAG project in later phases) -------------------
class IngestError(ChatbotError):
    """Base class for repository ingestion failures."""


class CloneError(IngestError):
    """Cloning the repository failed (bad URL, private repo, network, git missing)."""


class RepoTooLargeError(IngestError):
    """The repository exceeds the configured size or file-count limits."""
