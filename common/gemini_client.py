"""Chat backend for the Gemini Developer API (Google AI Studio keys).

Uses the native REST API through `requests`, like the Ollama client, so there is no SDK
dependency and every timeout and error is under our control. The API key is sent in the
`x-goog-api-key` header (never in the URL), is excluded from the config's repr, and is
never written to logs or error messages.

Error mapping (all subclasses of ChatbotError, see errors.py):
    cannot reach the API ............ BackendUnavailableError (after retries)
    key rejected (400/401/403) ...... AuthenticationError
    model name unknown (404) ........ ModelNotFoundError
    429 / local limiter ............. RateLimitError (retried with backoff when it is
                                      a short wait; a long wait is treated as an
                                      exhausted quota and raised immediately)
    500/502/503/504 ................. retried with backoff, then LLMError
    no response / stalled / too long  LLMTimeoutError
    safety block ..................... ContentBlockedError
    empty or malformed reply ........ InvalidOutputError
    session budget used up .......... BudgetExceededError

Only chat is provided here. Embeddings stay on the local Ollama backend (see
backends.py) so that switching chat backends never invalidates an index.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Iterator, Sequence
from urllib.parse import quote

import requests

from .config import Config
from .errors import (
    AuthenticationError,
    BackendUnavailableError,
    ConfigError,
    ContentBlockedError,
    InvalidOutputError,
    LLMError,
    LLMTimeoutError,
    ModelNotFoundError,
    RateLimitError,
)
from .limits import RateLimiter, UsageTracker
from .llm_client import Message, _build_messages  # shared message validation

log = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_BLOCKED_FINISH_REASONS = {
    "SAFETY",
    "RECITATION",
    "LANGUAGE",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
}
_RETRY_DELAY_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)s\s*$")  # e.g. "27s" or "27.5s"
_MAX_MODEL_LIST_PAGES = 10


class GeminiClient:
    """Defensive wrapper around the Gemini REST API (chat only)."""

    backend_name = "gemini"

    def __init__(
        self,
        config: Config | None = None,
        session: requests.Session | None = None,
        *,
        limiter: RateLimiter | None = None,
        usage: UsageTracker | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config if config is not None else Config.from_env()
        if not self.config.gemini_api_key.strip():
            raise ConfigError(
                "No Gemini API key. Export GEMINI_API_KEY (create one in Google AI Studio)."
            )
        self._session = session if session is not None else requests.Session()
        self._base_url = self.config.gemini_base_url.rstrip("/")
        self._sleep = sleep
        self._clock = clock
        self.limiter = limiter or RateLimiter(
            self.config.gemini_rpm, 60.0, clock=clock, sleep=sleep
        )
        self.usage = usage or UsageTracker(
            self.config.session_max_requests, self.config.session_max_tokens
        )
        # Flipped to False if the server rejects the thinking option for a model.
        self._send_thinking = self.config.gemini_thinking_level != "none"

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "GeminiClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ health
    @property
    def model_name(self) -> str:
        """The chat model this client uses by default."""
        return self.config.gemini_model

    def list_models(self) -> list[str]:
        """Names of models your key can use for chat, without the 'models/' prefix."""
        names: list[str] = []
        page_token: str | None = None
        for _ in range(_MAX_MODEL_LIST_PAGES):
            params: dict[str, Any] = {"pageSize": 1000}
            if page_token:
                params["pageToken"] = page_token
            resp = self._send("GET", "/v1beta/models", params=params, read_timeout=15.0)
            self._check(resp, model=None)
            data = self._json(resp)
            for item in data.get("models") or []:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    continue
                methods = item.get("supportedGenerationMethods")
                if isinstance(methods, list) and "generateContent" not in methods:
                    continue
                names.append(item["name"].removeprefix("models/"))
            page_token = data.get("nextPageToken") or None
            if not page_token:
                break
        return names

    def health_check(self) -> None:
        """Raise a ChatbotError if the key is rejected or the configured model is unavailable."""
        names = self.list_models()
        model = self.config.gemini_model
        if model not in names:
            suggestions = [n for n in names if n.startswith("gemini")][:8]
            hint = f" Models available to your key include: {', '.join(suggestions)}." if suggestions else ""
            raise ModelNotFoundError(
                f"Model '{model}' is not available for your key.{hint} "
                "Set CHATBOT_GEMINI_MODEL to one of them."
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
                "Gemini returned an empty response. Thinking tokens count against the output "
                "limit: use CHATBOT_GEMINI_THINKING_LEVEL=low or raise CHATBOT_GEMINI_MAX_OUTPUT_TOKENS."
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
        """Chat in structured-output mode and return the parsed JSON.

        The schema constrains the reply, but callers should still validate the values.
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

        The request is sent immediately, so errors such as a bad key or an unknown model
        are raised by this call rather than on first iteration.
        """
        msgs = _build_messages(messages, system)
        payload = self._payload(msgs, json_schema, temperature, max_tokens)
        model_name = model or self.config.gemini_model
        limit = total_timeout_s if total_timeout_s is not None else self.config.gemini_total_timeout_s
        deadline = self._clock() + limit
        resp = self._open_stream(model_name, payload)
        return self._iter_stream(resp, model_name, deadline, limit)

    # ------------------------------------------------------------------ internals
    def _payload(
        self,
        messages: list[Message],
        json_schema: dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        cfg = self.config
        system_text = [m["content"] for m in messages if m["role"] == "system"]
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
            if m["role"] != "system"
        ]
        if not contents:
            raise ValueError("At least one user or assistant message is required")
        generation: dict[str, Any] = {
            "maxOutputTokens": cfg.gemini_max_output_tokens if max_tokens is None else max_tokens,
        }
        if temperature is not None:
            generation["temperature"] = temperature
        elif cfg.gemini_temperature >= 0:
            generation["temperature"] = cfg.gemini_temperature  # otherwise: let Gemini use its default
        if json_schema is not None:
            generation["responseMimeType"] = "application/json"
            generation["responseJsonSchema"] = json_schema
        if self._send_thinking:
            generation["thinkingConfig"] = {"thinkingLevel": cfg.gemini_thinking_level}
        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation}
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_text)}]}
        return payload

    def _open_stream(self, model: str, payload: dict[str, Any]) -> requests.Response:
        """POST the streaming request, handling retries, limits and the thinking fallback."""
        cfg = self.config
        path = f"/v1beta/models/{quote(model, safe='-._')}:streamGenerateContent"
        generation = payload["generationConfig"]
        attempt = 0
        while True:
            self.usage.check()
            self.limiter.acquire(cfg.gemini_rate_wait_s)
            resp = self._send(
                "POST",
                path,
                json_body=payload,
                params={"alt": "sse"},
                read_timeout=cfg.gemini_read_timeout_s,
                stream=True,
            )
            if resp.status_code == 200:
                return resp
            status = resp.status_code
            message, delay = _error_info(resp)
            resp.close()

            if status == 400 and "thinkingConfig" in generation and "thinking" in message.lower():
                log.warning("Model %s rejected the thinking option; retrying without it", model)
                self._send_thinking = False
                generation.pop("thinkingConfig")
                continue

            if status == 429 and delay is not None and delay > cfg.gemini_max_backoff_s:
                # A long requested wait usually means the daily quota is gone.
                raise _status_error(status, message, model, delay)

            if status in _RETRYABLE_STATUS and attempt < cfg.gemini_max_retries:
                wait = delay if delay is not None else cfg.retry_backoff_s * (2**attempt)
                wait = min(wait, cfg.gemini_max_backoff_s)
                log.warning(
                    "Gemini returned HTTP %d (attempt %d/%d); waiting %.1fs. Reason: %s",
                    status,
                    attempt + 1,
                    cfg.gemini_max_retries + 1,
                    wait,
                    message[:300],
                )
                self._sleep(wait)
                attempt += 1
                continue

            raise _status_error(status, message, model, delay)

    def _iter_stream(
        self, resp: requests.Response, model: str, deadline: float, limit: float
    ) -> Iterator[str]:
        finished = False
        prompt_tokens = 0
        output_tokens = 0
        try:
            # chunk_size=1: by default requests waits to fill a 512-byte block before
            # yielding, which stalls short streamed replies until the server closes the
            # connection. Reading as data arrives keeps streaming responsive.
            for raw in resp.iter_lines(chunk_size=1):
                if self._clock() > deadline:
                    raise LLMTimeoutError(
                        f"Generation exceeded the total limit of {limit:.0f}s. "
                        "Raise CHATBOT_GEMINI_TOTAL_TIMEOUT_S or shorten the context."
                    )
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                line = line.strip()
                if not line.startswith("data:"):
                    continue  # SSE comments and keep-alives
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    finished = True
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise InvalidOutputError("Gemini sent a malformed stream event") from exc
                if not isinstance(obj, dict):
                    raise InvalidOutputError("Gemini sent an unexpected stream event")

                if isinstance(obj.get("error"), dict):
                    err = obj["error"]
                    code = err.get("code")
                    raise _status_error(
                        code if isinstance(code, int) else 500,
                        str(err.get("message") or "unknown error"),
                        model,
                        None,
                    )

                meta = obj.get("usageMetadata")
                if isinstance(meta, dict):
                    prompt_tokens = _as_int(meta.get("promptTokenCount"))
                    output_tokens = _as_int(meta.get("candidatesTokenCount")) + _as_int(
                        meta.get("thoughtsTokenCount")
                    )

                feedback = obj.get("promptFeedback")
                if isinstance(feedback, dict) and feedback.get("blockReason"):
                    raise ContentBlockedError(
                        f"Gemini blocked the prompt (reason: {feedback['blockReason']}). "
                        "Try rephrasing or using different code as context."
                    )

                candidates = obj.get("candidates")
                if not isinstance(candidates, list) or not candidates:
                    continue
                candidate = candidates[0] if isinstance(candidates[0], dict) else {}
                content = candidate.get("content")
                parts = content.get("parts") if isinstance(content, dict) else None
                for part in parts or []:
                    if not isinstance(part, dict) or part.get("thought"):
                        continue  # skip internal reasoning parts
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        yield text

                reason = candidate.get("finishReason")
                if reason:
                    if reason in _BLOCKED_FINISH_REASONS:
                        raise ContentBlockedError(
                            f"Gemini stopped the reply for safety or policy reasons ({reason})."
                        )
                    if reason == "MAX_TOKENS":
                        log.warning("Reply was cut off by the max_output_tokens limit")
                    elif reason != "STOP":
                        raise LLMError(f"Gemini stopped the reply unexpectedly (finish reason: {reason}).")
                    finished = True
                    break  # the reply is complete; do not wait for the server to close the stream
        except requests.RequestException as exc:
            if finished:
                log.debug("Ignoring a late connection error after a complete reply: %s", exc)
                return
            # requests reports a mid-stream read timeout as a ConnectionError.
            if "timed out" in str(exc).lower():
                raise LLMTimeoutError(
                    f"Gemini stopped sending data for {self.config.gemini_read_timeout_s:g}s mid-reply"
                ) from exc
            raise LLMError(f"Connection to Gemini was lost mid-reply: {exc}") from exc
        finally:
            resp.close()
            self.usage.record(prompt_tokens, output_tokens)
        if not finished:
            raise LLMError("Gemini closed the stream before the reply was complete")

    def _send(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        read_timeout: float,
        stream: bool = False,
    ) -> requests.Response:
        """Send one request. Retries connection failures only; translates exceptions."""
        cfg = self.config
        url = self._base_url + path
        headers = {"x-goog-api-key": cfg.gemini_api_key, "Content-Type": "application/json"}
        timeout = (cfg.connect_timeout_s, read_timeout)
        attempts = cfg.gemini_max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._session.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    params=params,
                    timeout=timeout,
                    stream=stream,
                )
            except requests.ReadTimeout as exc:
                raise LLMTimeoutError(
                    f"Gemini did not respond within {read_timeout:.0f}s. "
                    "Try again or raise CHATBOT_GEMINI_READ_TIMEOUT_S."
                ) from exc
            except requests.ConnectionError as exc:  # also covers ConnectTimeout
                last_exc = exc
                log.warning("Cannot reach Gemini (attempt %d/%d)", attempt, attempts)
                if attempt < attempts:
                    self._sleep(min(cfg.retry_backoff_s * (2 ** (attempt - 1)), cfg.gemini_max_backoff_s))
            except requests.RequestException as exc:
                raise LLMError(f"Request to Gemini failed: {type(exc).__name__}") from exc
        raise BackendUnavailableError(
            f"Cannot reach the Gemini API at {self._base_url}. Check your internet connection "
            "(or CHATBOT_GEMINI_BASE_URL)."
        ) from last_exc

    def _check(self, resp: requests.Response, *, model: str | None) -> None:
        if resp.status_code == 200:
            return
        status = resp.status_code
        message, delay = _error_info(resp)
        resp.close()
        raise _status_error(status, message, model, delay)

    @staticmethod
    def _json(resp: requests.Response) -> dict[str, Any]:
        try:
            data = resp.json()
        except ValueError as exc:
            raise InvalidOutputError("Gemini returned a non-JSON response") from exc
        if not isinstance(data, dict):
            raise InvalidOutputError("Gemini returned an unexpected JSON shape")
        return data


# ---------------------------------------------------------------------- helpers
def _as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _error_info(resp: requests.Response) -> tuple[str, float | None]:
    """Return (message, suggested_wait_seconds) from an error response."""
    message = ""
    delay: float | None = None
    try:
        data = resp.json()
    except ValueError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        err = data["error"]
        message = str(err.get("message") or "")
        status_name = err.get("status")
        if status_name and message:
            message = f"{status_name}: {message}"
        for detail in err.get("details") or []:
            if isinstance(detail, dict) and isinstance(detail.get("retryDelay"), str):
                match = _RETRY_DELAY_RE.match(detail["retryDelay"])
                if match:
                    delay = float(match.group(1))
    if not message:
        message = (resp.text or "").strip()[:300] or f"HTTP {resp.status_code}"
    header = resp.headers.get("Retry-After")
    if header:
        try:
            delay = float(header)
        except ValueError:
            pass
    return message, delay


def _status_error(status: int, message: str, model: str | None, delay: float | None) -> LLMError:
    lowered = message.lower()
    if status in (401, 403) or (
        status == 400 and ("api key not valid" in lowered or "api_key_invalid" in lowered)
    ):
        return AuthenticationError(
            f"Gemini rejected the API key ({message}). Check GEMINI_API_KEY, and that any key "
            "restrictions in Google Cloud allow the Gemini API."
        )
    if status == 404:
        return ModelNotFoundError(
            f"{message}. Check the model name (CHATBOT_GEMINI_MODEL); list the models your "
            "key can use with: python check_setup.py"
        )
    if status == 429:
        wait = f" Try again in about {delay:.0f}s." if delay is not None else ""
        return RateLimitError(
            f"Gemini rate limit or quota reached ({message}).{wait} Free-tier limits are low; "
            "see your limits in Google AI Studio.",
            retry_after=delay,
        )
    return LLMError(f"Gemini API returned HTTP {status}: {message}")
