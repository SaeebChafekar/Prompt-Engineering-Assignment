"""The only module that talks to Ollama.

Both chatbot projects call the model through OllamaClient, so timeouts, retries and
error translation live in exactly one place. To switch to a different backend later
(for example a cloud API), only this file needs to change.

Error mapping (all subclasses of ChatbotError, see errors.py):
    server not reachable ............ OllamaUnavailableError (after max_retries retries)
    model not installed ............. ModelNotFoundError
    no response / stalled / too long  LLMTimeoutError
    empty or malformed reply ........ InvalidOutputError
    anything else from the server ... LLMError
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterator, Sequence

import requests

from .config import Config
from .errors import (
    InvalidOutputError,
    LLMError,
    LLMTimeoutError,
    ModelNotFoundError,
    OllamaUnavailableError,
)

log = logging.getLogger(__name__)

Message = dict[str, str]
_VALID_ROLES = {"system", "user", "assistant"}


class OllamaClient:
    """Thin, defensive wrapper around Ollama's REST API."""

    backend_name = "ollama"

    def __init__(
        self,
        config: Config | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config if config is not None else Config.from_env()
        self._session = session if session is not None else requests.Session()
        self._base_url = self.config.ollama_host.rstrip("/")
        # Flipped to False if the server rejects the "think" option for a model.
        self._think_supported = True

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ health
    @property
    def model_name(self) -> str:
        """The chat model this client uses by default."""
        return self.config.chat_model

    def health_check(self) -> None:
        """Raise a ChatbotError if the server is unreachable or the chat model is missing."""
        self.require_models(self.config.chat_model)

    def list_models(self) -> list[str]:
        """Return the names of models installed in Ollama (e.g. 'qwen3:8b')."""
        resp = self._request("GET", "/api/tags", read_timeout=10.0)
        self._check(resp)
        data = self._json(resp)
        models = data.get("models", [])
        if not isinstance(models, list):
            raise InvalidOutputError("Ollama /api/tags returned an unexpected format")
        return [m["name"] for m in models if isinstance(m, dict) and isinstance(m.get("name"), str)]

    def require_models(self, *names: str) -> None:
        """Raise ModelNotFoundError (with the pull command) if any model is missing."""
        installed = self.list_models()
        missing = [n for n in names if not _is_installed(n, installed)]
        if missing:
            commands = "; ".join(f"ollama pull {n}" for n in missing)
            raise ModelNotFoundError(
                f"Model(s) not installed: {', '.join(missing)}. Install with: {commands}"
            )

    # ------------------------------------------------------------------ chat
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
    ) -> str:
        """Run a chat completion and return the full reply text."""
        text = "".join(
            self.chat_stream(
                messages,
                system=system,
                json_schema=json_schema,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                total_timeout_s=total_timeout_s,
            )
        )
        if not text.strip():
            raise InvalidOutputError(
                "The model returned an empty response. If you use a thinking model, "
                "make sure CHATBOT_THINK=false or raise CHATBOT_MAX_OUTPUT_TOKENS."
            )
        return text

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
    ) -> Any:
        """Chat with Ollama's structured-output mode and return the parsed JSON.

        The schema constrains the model's output, but small models can still produce
        values you must validate (e.g. with Pydantic); that is the caller's job.
        """
        text = self.chat(
            messages,
            system=system,
            json_schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            total_timeout_s=total_timeout_s,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            snippet = text[:200].replace("\n", " ")
            raise InvalidOutputError(f"Model output was not valid JSON: {snippet!r}") from exc

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
    ) -> Iterator[str]:
        """Start a chat completion and return an iterator of text pieces.

        The connection is opened immediately, so errors such as a missing model or an
        unreachable server are raised by this call rather than on first iteration.
        """
        msgs = _build_messages(messages, system)
        payload = self._chat_payload(msgs, json_schema, temperature, max_tokens, model)
        limit = total_timeout_s if total_timeout_s is not None else self.config.chat_total_timeout_s
        deadline = time.monotonic() + limit
        resp = self._open_chat(payload, model=payload["model"])
        return self._iter_chat(resp, deadline, limit)

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Embed texts in batches. Returns one vector per input text, in order."""
        if not texts:
            return []
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"embed() got an empty or non-string text at index {i}")

        model_name = model or self.config.embed_model
        batch_size = self.config.embed_batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            payload = {
                "model": model_name,
                "input": batch,
                "truncate": True,
                "keep_alive": self.config.keep_alive,
            }
            resp = self._request(
                "POST", "/api/embed", json_body=payload, read_timeout=self.config.embed_timeout_s
            )
            self._check(resp, model=model_name)
            data = self._json(resp)
            batch_vectors = data.get("embeddings")
            _validate_embeddings(batch_vectors, expected=len(batch))
            vectors.extend(batch_vectors)
        return vectors

    # ------------------------------------------------------------------ internals
    def _chat_payload(
        self,
        messages: list[Message],
        json_schema: dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        model: str | None,
    ) -> dict[str, Any]:
        cfg = self.config
        payload: dict[str, Any] = {
            "model": model or cfg.chat_model,
            "messages": messages,
            "stream": True,
            "keep_alive": cfg.keep_alive,
            "options": {
                "num_ctx": cfg.num_ctx,
                "temperature": cfg.temperature if temperature is None else temperature,
                "num_predict": cfg.max_output_tokens if max_tokens is None else max_tokens,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema
        if self._think_supported:
            payload["think"] = cfg.think
        return payload

    def _open_chat(self, payload: dict[str, Any], *, model: str) -> requests.Response:
        """POST /api/chat, retrying once without 'think' if the model rejects it."""
        for _ in range(2):
            resp = self._request(
                "POST",
                "/api/chat",
                json_body=payload,
                read_timeout=self.config.read_timeout_s,
                stream=True,
            )
            if resp.status_code == 200:
                return resp
            message = _error_message(resp)
            status = resp.status_code
            resp.close()
            if status == 400 and "think" in payload and "think" in message.lower():
                log.warning("Model %s does not support 'think'; retrying without it", model)
                self._think_supported = False
                payload.pop("think")
                continue
            raise _status_error(status, message, model)
        raise LLMError("Ollama kept rejecting the chat request")  # pragma: no cover

    def _iter_chat(
        self, resp: requests.Response, deadline: float, limit: float
    ) -> Iterator[str]:
        finished = False
        try:
            # chunk_size=1: by default requests waits to fill a 512-byte block before
            # yielding, which stalls short streamed replies until the server closes the
            # connection. Reading as data arrives keeps streaming responsive.
            for raw in resp.iter_lines(chunk_size=1):
                if time.monotonic() > deadline:
                    raise LLMTimeoutError(
                        f"Generation exceeded the total limit of {limit:.0f}s. "
                        "Try a shorter context or raise CHATBOT_CHAT_TOTAL_TIMEOUT_S."
                    )
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise InvalidOutputError("Ollama sent a malformed stream line") from exc
                if not isinstance(obj, dict):
                    raise InvalidOutputError("Ollama sent an unexpected stream line")
                if "error" in obj:
                    raise LLMError(f"Ollama reported an error during generation: {obj['error']}")
                piece = (obj.get("message") or {}).get("content") or ""
                if piece:
                    yield piece
                if obj.get("done"):
                    finished = True
                    if obj.get("done_reason") == "length":
                        log.warning("Reply was cut off by the max_output_tokens limit")
                    break
        except requests.RequestException as exc:
            # requests reports a mid-stream read timeout as a ConnectionError.
            if "timed out" in str(exc).lower():
                raise LLMTimeoutError(
                    f"Ollama stopped sending data for {self.config.read_timeout_s:.0f}s mid-reply"
                ) from exc
            raise LLMError(f"Connection to Ollama was lost mid-reply: {exc}") from exc
        finally:
            resp.close()
        if not finished:
            raise LLMError("Ollama closed the stream before the reply was complete")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        read_timeout: float,
        stream: bool = False,
    ) -> requests.Response:
        """Send a request; retry connection failures only; translate exceptions."""
        url = self._base_url + path
        timeout = (self.config.connect_timeout_s, read_timeout)
        attempts = self.config.max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._session.request(
                    method, url, json=json_body, timeout=timeout, stream=stream
                )
            except requests.ReadTimeout as exc:
                raise LLMTimeoutError(
                    f"Ollama did not respond within {read_timeout:.0f}s ({path}). The model "
                    "may still be loading; try again or raise CHATBOT_READ_TIMEOUT_S."
                ) from exc
            except requests.ConnectionError as exc:  # also covers ConnectTimeout
                last_exc = exc
                log.warning("Cannot reach Ollama (attempt %d/%d): %s", attempt, attempts, exc)
                if attempt < attempts:
                    time.sleep(self.config.retry_backoff_s * attempt)
            except requests.RequestException as exc:
                raise LLMError(f"Request to Ollama failed: {exc}") from exc
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {self._base_url}. Is it running? Start it with "
            "'ollama serve' (or check CHATBOT_OLLAMA_HOST)."
        ) from last_exc

    def _check(self, resp: requests.Response, *, model: str | None = None) -> None:
        if resp.status_code == 200:
            return
        message = _error_message(resp)
        status = resp.status_code
        resp.close()
        raise _status_error(status, message, model)

    @staticmethod
    def _json(resp: requests.Response) -> dict[str, Any]:
        try:
            data = resp.json()
        except ValueError as exc:
            raise InvalidOutputError("Ollama returned a non-JSON response") from exc
        if not isinstance(data, dict):
            raise InvalidOutputError("Ollama returned an unexpected JSON shape")
        return data


# ---------------------------------------------------------------------- helpers
def _build_messages(messages: Sequence[Message], system: str | None) -> list[Message]:
    if not messages:
        raise ValueError("messages must not be empty")
    built: list[Message] = []
    if system is not None:
        built.append({"role": "system", "content": system})
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{i}] must be a dict with 'role' and 'content'")
        role, content = msg.get("role"), msg.get("content")
        if role not in _VALID_ROLES:
            raise ValueError(f"messages[{i}] has invalid role {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"messages[{i}] content must be a string")
        built.append({"role": role, "content": content})
    return built


def _is_installed(name: str, installed: list[str]) -> bool:
    if name in installed:
        return True
    # "nomic-embed-text" is listed by Ollama as "nomic-embed-text:latest".
    return ":" not in name and f"{name}:latest" in installed


def _error_message(resp: requests.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
    except ValueError:
        pass
    return (resp.text or "").strip()[:300] or f"HTTP {resp.status_code}"


def _status_error(status: int, message: str, model: str | None) -> LLMError:
    if status == 404 and "model" in message.lower():
        hint = f" Install it with: ollama pull {model}" if model else ""
        return ModelNotFoundError(f"{message}.{hint}")
    return LLMError(f"Ollama returned HTTP {status}: {message}")


def _validate_embeddings(vectors: Any, *, expected: int) -> None:
    if not isinstance(vectors, list) or len(vectors) != expected:
        got = len(vectors) if isinstance(vectors, list) else type(vectors).__name__
        raise InvalidOutputError(f"Expected {expected} embeddings from Ollama, got {got}")
    for i, vec in enumerate(vectors):
        if not isinstance(vec, list) or not vec or not all(
            isinstance(x, (int, float)) for x in vec
        ):
            raise InvalidOutputError(f"Embedding {i} from Ollama is empty or malformed")
