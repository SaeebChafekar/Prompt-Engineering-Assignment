"""Tests for GeminiClient, run against a fake Gemini REST server.

No internet, API key, or quota is needed. Run from the project root:

    python -m unittest tests.test_gemini_client -v
"""

from __future__ import annotations

import json
import re
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from common.config import Config
from common.errors import (
    AuthenticationError,
    BackendUnavailableError,
    BudgetExceededError,
    ContentBlockedError,
    InvalidOutputError,
    LLMError,
    LLMTimeoutError,
    ModelNotFoundError,
    RateLimitError,
)
from common.gemini_client import GeminiClient

GOOD_KEY = "test-key"
USAGE = {"promptTokenCount": 10, "candidatesTokenCount": 2, "totalTokenCount": 12}
SCHEMA = {"type": "object", "properties": {"language": {"type": "string"}}}
RETRY_429 = {
    "error": {
        "code": 429,
        "message": "Quota exceeded",
        "status": "RESOURCE_EXHAUSTED",
        "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "2s"}],
    }
}


def _chunk(text=None, *, finish=None, usage=None, thought=False) -> dict:
    parts = []
    if text is not None:
        part = {"text": text}
        if thought:
            part["thought"] = True
        parts.append(part)
    candidate: dict = {"content": {"role": "model", "parts": parts}}
    if finish:
        candidate["finishReason"] = finish
    obj: dict = {"candidates": [candidate]}
    if usage:
        obj["usageMetadata"] = usage
    return obj


class FakeGemini(BaseHTTPRequestHandler):
    """Behaviour is selected by the model name in the URL."""

    seen: list[tuple[str, dict, dict]] = []  # (model, headers, payload)
    calls: dict[str, int] = {}

    def log_message(self, *args: object) -> None:
        pass

    def _json(self, status: int, obj: dict, headers: dict | None = None) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, obj: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()

    def _auth_ok(self) -> bool:
        key = self.headers.get("x-goog-api-key")
        if key == GOOD_KEY:
            return True
        if key == "restricted-key":
            self._json(403, {"error": {"code": 403, "message": "Requests are blocked", "status": "PERMISSION_DENIED"}})
        else:
            self._json(400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}})
        return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if not self._auth_ok():
            return
        if parsed.path == "/v1beta/models":
            if parse_qs(parsed.query).get("pageToken") == ["p2"]:
                return self._json(200, {"models": [{"name": "models/gemini-b", "supportedGenerationMethods": ["generateContent"]}]})
            return self._json(
                200,
                {
                    "models": [
                        {"name": "models/normal", "supportedGenerationMethods": ["generateContent", "countTokens"]},
                        {"name": "models/embed-only", "supportedGenerationMethods": ["embedContent"]},
                    ],
                    "nextPageToken": "p2",
                },
            )
        self._json(404, {"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        match = re.match(r"^/v1beta/models/(.+):streamGenerateContent$", parsed.path)
        if not match:
            return self._json(404, {"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}})
        model = match.group(1)
        FakeGemini.seen.append((model, dict(self.headers), payload))
        FakeGemini.calls[model] = FakeGemini.calls.get(model, 0) + 1
        n = FakeGemini.calls[model]
        if not self._auth_ok():
            return

        if model == "missing":
            return self._json(404, {"error": {"code": 404, "message": "models/missing is not found for API version v1beta", "status": "NOT_FOUND"}})
        if model == "rate-limit-once" and n == 1:
            return self._json(429, RETRY_429)
        if model == "rate-limit-header" and n == 1:
            return self._json(429, {"error": {"code": 429, "message": "slow down", "status": "RESOURCE_EXHAUSTED"}}, {"Retry-After": "3"})
        if model == "rate-limit-daily":
            err = json.loads(json.dumps(RETRY_429))
            err["error"]["details"][0]["retryDelay"] = "3600s"
            return self._json(429, err)
        if model == "always-503":
            return self._json(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
        if model == "nothink" and "thinkingConfig" in payload.get("generationConfig", {}):
            return self._json(400, {"error": {"code": 400, "message": "Thinking budget is not supported for this model", "status": "INVALID_ARGUMENT"}})
        if model == "slow-start":
            time.sleep(1.5)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            if model == "blocked-prompt":
                self._sse({"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": USAGE})
            elif model == "blocked-finish":
                self._sse(_chunk(None, finish="SAFETY"))
            elif model == "weird-finish":
                self._sse(_chunk("x", finish="OTHER"))
            elif model == "max-tokens":
                self._sse(_chunk("partial", finish="MAX_TOKENS", usage=USAGE))
            elif model == "empty-max":
                self._sse(_chunk(None, finish="MAX_TOKENS"))
            elif model == "thought":
                self._sse(_chunk("internal reasoning", thought=True))
                self._sse(_chunk("Answer", finish="STOP", usage=USAGE))
            elif model == "error-mid":
                self._sse(_chunk("partial"))
                self._sse({"error": {"code": 503, "message": "boom", "status": "UNAVAILABLE"}})
            elif model == "no-finish":
                self._sse(_chunk("partial"))
            elif model == "stall-mid":
                self._sse(_chunk("partial"))
                time.sleep(1.5)
                self._sse(_chunk("x", finish="STOP"))
            elif model == "endless":
                while True:
                    self._sse(_chunk("x"))
                    time.sleep(0.1)
            elif model == "hang-after-finish":
                # complete reply, but the server keeps the connection open afterwards
                self._sse(_chunk("Hello", finish="STOP", usage=USAGE))
                time.sleep(1.5)
            elif model == "bad-sse":
                self.wfile.write(b"data: not json\n\n")
            elif payload.get("generationConfig", {}).get("responseMimeType") == "application/json":
                if model == "badjson":
                    self._sse(_chunk("not json at all", finish="STOP", usage=USAGE))
                else:
                    self._sse(_chunk('{"language":'))
                    self._sse(_chunk(' "python"}', finish="STOP", usage=USAGE))
            else:
                # keep-alive comment line and blank lines must be ignored
                self.wfile.write(b": keep-alive\n\n")
                self._sse(_chunk("Hel"))
                self._sse(_chunk("lo", finish="STOP", usage=USAGE))
        except (BrokenPipeError, ConnectionResetError):
            pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class GeminiTestCase(unittest.TestCase):
    server: ThreadingHTTPServer
    port: int

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGemini)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def config(self, **overrides: object) -> Config:
        values: dict = dict(
            backend="gemini",
            gemini_api_key=GOOD_KEY,
            gemini_base_url=f"http://127.0.0.1:{self.port}",
            gemini_model="normal",
            connect_timeout_s=1.0,
            gemini_read_timeout_s=0.5,
            gemini_total_timeout_s=5.0,
            gemini_rpm=0,
            gemini_max_retries=2,
            gemini_max_backoff_s=10.0,
            retry_backoff_s=0.5,
        )
        values.update(overrides)
        return Config(**values)

    def setUp(self) -> None:
        FakeGemini.seen = []
        FakeGemini.calls = {}
        self.sleeps: list[float] = []
        self.client = GeminiClient(self.config(), sleep=self.sleeps.append)

    def tearDown(self) -> None:
        self.client.close()

    msgs = [{"role": "user", "content": "hi"}]


class ChatTests(GeminiTestCase):
    def test_chat_returns_full_text(self) -> None:
        self.assertEqual(self.client.chat(self.msgs), "Hello")

    def test_chat_stream_yields_pieces(self) -> None:
        self.assertEqual(list(self.client.chat_stream(self.msgs)), ["Hel", "lo"])

    def test_request_shape(self) -> None:
        self.client.chat(
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ],
            system="be brief",
            max_tokens=7,
            temperature=0.3,
        )
        model, headers, payload = FakeGemini.seen[-1]
        self.assertEqual(model, "normal")
        self.assertEqual(headers.get("x-goog-api-key"), GOOD_KEY)
        self.assertEqual(payload["systemInstruction"], {"parts": [{"text": "be brief"}]})
        self.assertEqual([c["role"] for c in payload["contents"]], ["user", "model", "user"])
        gen = payload["generationConfig"]
        self.assertEqual(gen["maxOutputTokens"], 7)
        self.assertEqual(gen["temperature"], 0.3)  # an explicit per-call value is always sent
        self.assertEqual(gen["thinkingConfig"], {"thinkingLevel": "low"})  # Gemini 3.x: a level, not a budget
        self.assertNotIn("thinkingBudget", gen["thinkingConfig"])

    def test_defaults_follow_gemini_3_guidance(self) -> None:
        self.client.chat(self.msgs)
        gen = FakeGemini.seen[-1][2]["generationConfig"]
        self.assertNotIn("temperature", gen)  # left at the model default unless configured
        self.assertEqual(gen["maxOutputTokens"], 4096)  # generous: thinking tokens count against it

    def test_temperature_and_limits_can_be_configured(self) -> None:
        client = GeminiClient(self.config(gemini_temperature=0.2, gemini_max_output_tokens=900), sleep=self.sleeps.append)
        client.chat(self.msgs)
        client.close()
        gen = FakeGemini.seen[-1][2]["generationConfig"]
        self.assertEqual((gen["temperature"], gen["maxOutputTokens"]), (0.2, 900))

    def test_each_thinking_level_is_sent_as_given(self) -> None:
        for level in ("low", "medium", "high"):
            client = GeminiClient(self.config(gemini_thinking_level=level), sleep=self.sleeps.append)
            client.chat(self.msgs)
            client.close()
            self.assertEqual(FakeGemini.seen[-1][2]["generationConfig"]["thinkingConfig"], {"thinkingLevel": level})

    def test_system_only_messages_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.client.chat([{"role": "system", "content": "only system"}])

    def test_chat_json_sends_schema_and_parses(self) -> None:
        self.assertEqual(self.client.chat_json(self.msgs, SCHEMA), {"language": "python"})
        gen = FakeGemini.seen[-1][2]["generationConfig"]
        self.assertEqual(gen["responseMimeType"], "application/json")
        self.assertEqual(gen["responseJsonSchema"], SCHEMA)

    def test_chat_json_invalid_raises(self) -> None:
        with self.assertRaises(InvalidOutputError):
            self.client.chat_json(self.msgs, SCHEMA, model="badjson")

    def test_returns_as_soon_as_reply_is_finished(self) -> None:
        # The server holds the connection open for 1.5s (read timeout is 0.5s) after
        # sending the final event. The client must not wait for it to close.
        start = time.monotonic()
        self.assertEqual(self.client.chat(self.msgs, model="hang-after-finish"), "Hello")
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertEqual(self.client.usage.requests, 1)
        self.assertEqual(self.client.usage.prompt_tokens, 10)

    def test_thought_parts_are_skipped(self) -> None:
        self.assertEqual(self.client.chat(self.msgs, model="thought"), "Answer")

    def test_max_tokens_returns_partial_text(self) -> None:
        self.assertEqual(self.client.chat(self.msgs, model="max-tokens"), "partial")

    def test_empty_reply_raises(self) -> None:
        with self.assertRaises(InvalidOutputError):
            self.client.chat(self.msgs, model="empty-max")

    def test_thinking_option_falls_back_once(self) -> None:
        self.assertEqual(self.client.chat(self.msgs, model="nothink"), "Hello")
        self.client.chat(self.msgs, model="nothink")
        payloads = [p for m, _, p in FakeGemini.seen if m == "nothink"]
        self.assertEqual(len(payloads), 3)  # rejected, retried, then the second chat call
        self.assertNotIn("thinkingConfig", payloads[1]["generationConfig"])
        self.assertNotIn("thinkingConfig", payloads[2]["generationConfig"])

    def test_thinking_can_be_left_enabled(self) -> None:
        client = GeminiClient(self.config(gemini_thinking_level="none"), sleep=self.sleeps.append)
        client.chat(self.msgs)
        client.close()
        self.assertNotIn("thinkingConfig", FakeGemini.seen[-1][2]["generationConfig"])


class AuthAndModelTests(GeminiTestCase):
    def test_invalid_key(self) -> None:
        client = GeminiClient(self.config(gemini_api_key="wrong"), sleep=self.sleeps.append)
        with self.assertRaises(AuthenticationError):
            client.chat(self.msgs)
        self.assertEqual(self.sleeps, [])  # auth errors are never retried
        client.close()

    def test_restricted_key(self) -> None:
        client = GeminiClient(self.config(gemini_api_key="restricted-key"), sleep=self.sleeps.append)
        with self.assertRaises(AuthenticationError):
            client.chat(self.msgs)
        client.close()

    def test_unknown_model(self) -> None:
        with self.assertRaises(ModelNotFoundError) as ctx:
            self.client.chat(self.msgs, model="missing")
        self.assertIn("CHATBOT_GEMINI_MODEL", str(ctx.exception))

    def test_list_models_follows_pages_and_filters(self) -> None:
        self.assertEqual(self.client.list_models(), ["normal", "gemini-b"])

    def test_health_check_ok(self) -> None:
        self.client.health_check()

    def test_health_check_reports_alternatives(self) -> None:
        client = GeminiClient(self.config(gemini_model="gemini-ghost"), sleep=self.sleeps.append)
        with self.assertRaises(ModelNotFoundError) as ctx:
            client.health_check()
        self.assertIn("gemini-b", str(ctx.exception))
        client.close()

    def test_health_check_bad_key(self) -> None:
        client = GeminiClient(self.config(gemini_api_key="wrong"), sleep=self.sleeps.append)
        with self.assertRaises(AuthenticationError):
            client.health_check()
        client.close()


class RetryTests(GeminiTestCase):
    def test_retry_delay_from_body(self) -> None:
        self.assertEqual(self.client.chat(self.msgs, model="rate-limit-once"), "Hello")
        self.assertEqual(self.sleeps, [2.0])

    def test_retry_after_header(self) -> None:
        self.assertEqual(self.client.chat(self.msgs, model="rate-limit-header"), "Hello")
        self.assertEqual(self.sleeps, [3.0])

    def test_long_wait_is_treated_as_exhausted_quota(self) -> None:
        with self.assertRaises(RateLimitError) as ctx:
            self.client.chat(self.msgs, model="rate-limit-daily")
        self.assertEqual(ctx.exception.retry_after, 3600.0)
        self.assertEqual(self.sleeps, [])
        self.assertEqual(FakeGemini.calls["rate-limit-daily"], 1)

    def test_server_errors_retry_with_exponential_backoff_then_fail(self) -> None:
        with self.assertRaises(LLMError) as ctx:
            self.client.chat(self.msgs, model="always-503")
        self.assertNotIsInstance(ctx.exception, RateLimitError)
        self.assertEqual(self.sleeps, [0.5, 1.0])
        self.assertEqual(FakeGemini.calls["always-503"], 3)

    def test_unreachable_server_retries_then_reports(self) -> None:
        cfg = self.config(gemini_base_url=f"http://127.0.0.1:{_free_port()}", connect_timeout_s=0.5)
        client = GeminiClient(cfg, sleep=self.sleeps.append)
        with self.assertRaises(BackendUnavailableError) as ctx:
            client.chat(self.msgs)
        self.assertEqual(self.sleeps, [0.5, 1.0])
        client.close()
        self.assertNotIn(GOOD_KEY, str(ctx.exception))
        self.assertNotIn(GOOD_KEY, repr(cfg))


class SafetyAndStreamFailureTests(GeminiTestCase):
    def test_blocked_prompt(self) -> None:
        with self.assertRaises(ContentBlockedError):
            self.client.chat(self.msgs, model="blocked-prompt")

    def test_blocked_finish_reason(self) -> None:
        with self.assertRaises(ContentBlockedError):
            self.client.chat(self.msgs, model="blocked-finish")

    def test_unexpected_finish_reason(self) -> None:
        with self.assertRaises(LLMError) as ctx:
            self.client.chat(self.msgs, model="weird-finish")
        self.assertIn("OTHER", str(ctx.exception))

    def test_error_event_mid_stream(self) -> None:
        with self.assertRaises(LLMError) as ctx:
            self.client.chat(self.msgs, model="error-mid")
        self.assertIn("boom", str(ctx.exception))

    def test_stream_without_finish_reason_is_error(self) -> None:
        with self.assertRaises(LLMError):
            self.client.chat(self.msgs, model="no-finish")

    def test_malformed_event(self) -> None:
        with self.assertRaises(InvalidOutputError):
            self.client.chat(self.msgs, model="bad-sse")


class TimeoutTests(GeminiTestCase):
    def test_no_response_times_out(self) -> None:
        with self.assertRaises(LLMTimeoutError):
            self.client.chat(self.msgs, model="slow-start")

    def test_stall_mid_stream_times_out(self) -> None:
        with self.assertRaises(LLMTimeoutError):
            self.client.chat(self.msgs, model="stall-mid")

    def test_total_deadline_stops_endless_stream(self) -> None:
        start = time.monotonic()
        with self.assertRaises(LLMTimeoutError):
            self.client.chat(self.msgs, model="endless", total_timeout_s=0.5)
        self.assertLess(time.monotonic() - start, 3.0)


class LimitIntegrationTests(GeminiTestCase):
    def test_usage_is_recorded_from_stream(self) -> None:
        self.client.chat(self.msgs)
        self.assertEqual(self.client.usage.requests, 1)
        self.assertEqual(self.client.usage.prompt_tokens, 10)
        self.assertEqual(self.client.usage.output_tokens, 2)

    def test_request_budget_blocks_before_network(self) -> None:
        client = GeminiClient(self.config(session_max_requests=1), sleep=self.sleeps.append)
        client.chat(self.msgs)
        before = len(FakeGemini.seen)
        with self.assertRaises(BudgetExceededError):
            client.chat(self.msgs)
        self.assertEqual(len(FakeGemini.seen), before)
        client.close()

    def test_local_rate_limiter_is_applied(self) -> None:
        client = GeminiClient(
            self.config(gemini_rpm=1, gemini_rate_wait_s=1.0), sleep=self.sleeps.append
        )
        client.chat(self.msgs)
        with self.assertRaises(RateLimitError):
            client.chat(self.msgs)  # second call within the minute needs ~60s of waiting
        client.close()


if __name__ == "__main__":
    unittest.main()
