"""Tests for Config and OllamaClient, run against a fake Ollama server.

No real Ollama or GPU is needed. Run from the project root:

    python -m unittest tests.test_llm_client -v
"""

from __future__ import annotations

import json
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from common.config import Config
from common.errors import (
    ConfigError,
    InvalidOutputError,
    LLMError,
    LLMTimeoutError,
    ModelNotFoundError,
    OllamaUnavailableError,
)
from common.llm_client import OllamaClient

SCHEMA = {"type": "object", "properties": {"language": {"type": "string"}}}


class FakeOllama(BaseHTTPRequestHandler):
    """Behaviour is selected by the model name in the request."""

    seen: list[tuple[str, dict]] = []

    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass

    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _line(self, obj: dict) -> None:
        self.wfile.write((json.dumps(obj) + "\n").encode())
        self.wfile.flush()

    def _piece(self, text: str) -> None:
        self._line({"message": {"role": "assistant", "content": text}, "done": False})

    def _done(self) -> None:
        self._line({"message": {"role": "assistant", "content": ""}, "done": True})

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            self._send_json(
                200, {"models": [{"name": "qwen3:8b"}, {"name": "nomic-embed-text:latest"}]}
            )
        else:
            self._send_json(404, {"error": "404 page not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        FakeOllama.seen.append((self.path, payload))
        if self.path == "/api/chat":
            self._chat(payload)
        elif self.path == "/api/embed":
            self._embed(payload)
        else:
            self._send_json(404, {"error": "404 page not found"})

    def _chat(self, p: dict) -> None:
        model = p.get("model")
        if model == "missing":
            return self._send_json(404, {"error": "model 'missing' not found"})
        if model == "nothink" and "think" in p:
            return self._send_json(400, {"error": '"nothink" does not support thinking'})
        if model == "server-error":
            return self._send_json(500, {"error": "llama runner process has terminated"})
        if model == "slow-start":
            time.sleep(1.5)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        try:
            if model == "error-mid":
                self._piece("partial")
                self._line({"error": "boom"})
            elif model == "no-done":
                self._piece("partial")
            elif model == "stall-mid":
                self._piece("partial")
                time.sleep(1.5)
                self._done()
            elif model == "endless":
                while True:
                    self._piece("x")
                    time.sleep(0.1)
            elif model == "empty":
                self._done()
            elif model == "hang-after-done":
                # complete reply, but the server keeps the connection open afterwards
                self._piece("Hello")
                self._done()
                time.sleep(1.5)
            elif p.get("format") is not None:
                if model == "badjson":
                    self._piece("not json at all")
                else:
                    self._piece('{"language":')
                    self._piece(' "python"}')
                self._done()
            else:
                self._piece("Hel")
                self._piece("lo")
                self._done()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _embed(self, p: dict) -> None:
        model, inputs = p.get("model"), p.get("input", [])
        if model == "bad-count":
            return self._send_json(200, {"embeddings": [[0.1, 0.2]] * (len(inputs) - 1)})
        if model == "bad-shape":
            return self._send_json(200, {"embeddings": [[] for _ in inputs]})
        if model == "missing":
            return self._send_json(404, {"error": "model 'missing' not found"})
        self._send_json(200, {"embeddings": [[0.1, 0.2, 0.3] for _ in inputs]})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ClientTestCase(unittest.TestCase):
    server: ThreadingHTTPServer
    port: int

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        FakeOllama.seen = []
        self.cfg = Config(
            ollama_host=f"http://127.0.0.1:{self.port}",
            connect_timeout_s=1.0,
            read_timeout_s=0.5,
            chat_total_timeout_s=5.0,
            embed_batch_size=2,
            max_retries=1,
            retry_backoff_s=0.01,
        )
        self.client = OllamaClient(self.cfg)

    def tearDown(self) -> None:
        self.client.close()

    msgs = [{"role": "user", "content": "hi"}]


class HealthTests(ClientTestCase):
    def test_list_models(self) -> None:
        self.assertEqual(self.client.list_models(), ["qwen3:8b", "nomic-embed-text:latest"])

    def test_require_models_accepts_latest_alias(self) -> None:
        self.client.require_models("qwen3:8b", "nomic-embed-text")

    def test_require_models_reports_pull_command(self) -> None:
        with self.assertRaises(ModelNotFoundError) as ctx:
            self.client.require_models("qwen3:8b", "llama3.1:8b")
        self.assertIn("ollama pull llama3.1:8b", str(ctx.exception))

    def test_unreachable_server(self) -> None:
        cfg = Config(
            ollama_host=f"http://127.0.0.1:{_free_port()}",
            connect_timeout_s=0.5,
            max_retries=1,
            retry_backoff_s=0.01,
        )
        with OllamaClient(cfg) as client:
            with self.assertRaises(OllamaUnavailableError):
                client.list_models()


class ChatTests(ClientTestCase):
    def test_chat_returns_full_text(self) -> None:
        self.assertEqual(self.client.chat(self.msgs), "Hello")

    def test_chat_stream_yields_pieces(self) -> None:
        self.assertEqual(list(self.client.chat_stream(self.msgs)), ["Hel", "lo"])

    def test_returns_as_soon_as_reply_is_done(self) -> None:
        # Server holds the connection open 1.5s (read timeout is 0.5s) after "done".
        start = time.monotonic()
        self.assertEqual(self.client.chat(self.msgs, model="hang-after-done"), "Hello")
        self.assertLess(time.monotonic() - start, 1.0)

    def test_system_prompt_is_prepended_and_options_sent(self) -> None:
        self.client.chat(self.msgs, system="be brief", max_tokens=7)
        _, payload = FakeOllama.seen[-1]
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "be brief"})
        self.assertEqual(payload["options"]["num_predict"], 7)
        self.assertEqual(payload["options"]["num_ctx"], self.cfg.num_ctx)
        self.assertIs(payload["think"], False)

    def test_chat_json_parses(self) -> None:
        self.assertEqual(self.client.chat_json(self.msgs, SCHEMA), {"language": "python"})

    def test_chat_json_invalid_raises(self) -> None:
        with self.assertRaises(InvalidOutputError):
            self.client.chat_json(self.msgs, SCHEMA, model="badjson")

    def test_empty_reply_raises(self) -> None:
        with self.assertRaises(InvalidOutputError):
            self.client.chat(self.msgs, model="empty")

    def test_missing_model(self) -> None:
        with self.assertRaises(ModelNotFoundError) as ctx:
            self.client.chat(self.msgs, model="missing")
        self.assertIn("ollama pull missing", str(ctx.exception))

    def test_server_error_is_llm_error(self) -> None:
        with self.assertRaises(LLMError) as ctx:
            self.client.chat(self.msgs, model="server-error")
        self.assertNotIsInstance(ctx.exception, (ModelNotFoundError, LLMTimeoutError))

    def test_think_option_falls_back_once(self) -> None:
        self.assertEqual(self.client.chat(self.msgs, model="nothink"), "Hello")
        self.client.chat(self.msgs, model="nothink")
        chat_calls = [p for path, p in FakeOllama.seen if path == "/api/chat"]
        self.assertEqual(len(chat_calls), 3)  # rejected, retried, then second chat call
        self.assertIn("think", chat_calls[0])
        self.assertNotIn("think", chat_calls[1])
        self.assertNotIn("think", chat_calls[2])

    def test_invalid_messages(self) -> None:
        with self.assertRaises(ValueError):
            self.client.chat([])
        with self.assertRaises(ValueError):
            self.client.chat([{"role": "robot", "content": "hi"}])
        with self.assertRaises(ValueError):
            self.client.chat([{"role": "user", "content": 5}])  # type: ignore[dict-item]


class TimeoutAndStreamFailureTests(ClientTestCase):
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

    def test_error_line_mid_stream(self) -> None:
        with self.assertRaises(LLMError) as ctx:
            self.client.chat(self.msgs, model="error-mid")
        self.assertIn("boom", str(ctx.exception))

    def test_stream_without_done_is_error(self) -> None:
        with self.assertRaises(LLMError):
            self.client.chat(self.msgs, model="no-done")


class EmbedTests(ClientTestCase):
    def test_batches_and_preserves_count(self) -> None:
        vectors = self.client.embed(["a", "b", "c", "d", "e"])
        self.assertEqual(len(vectors), 5)
        embed_calls = [p for path, p in FakeOllama.seen if path == "/api/embed"]
        self.assertEqual([len(p["input"]) for p in embed_calls], [2, 2, 1])

    def test_empty_list_makes_no_request(self) -> None:
        self.assertEqual(self.client.embed([]), [])
        self.assertEqual(FakeOllama.seen, [])

    def test_blank_text_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.client.embed(["ok", "   "])

    def test_wrong_count_rejected(self) -> None:
        with self.assertRaises(InvalidOutputError):
            self.client.embed(["a", "b"], model="bad-count")

    def test_malformed_vector_rejected(self) -> None:
        with self.assertRaises(InvalidOutputError):
            self.client.embed(["a"], model="bad-shape")

    def test_missing_embed_model(self) -> None:
        with self.assertRaises(ModelNotFoundError):
            self.client.embed(["a"], model="missing")


class ConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        self.assertEqual(Config().chat_model, "qwen3:8b")

    def test_env_overrides_with_types(self) -> None:
        cfg = Config.from_env(
            {
                "CHATBOT_NUM_CTX": "4096",
                "CHATBOT_THINK": "true",
                "CHATBOT_TEMPERATURE": "0.5",
                "CHATBOT_CHAT_MODEL": " llama3.1:8b ",
            }
        )
        self.assertEqual(cfg.num_ctx, 4096)
        self.assertIs(cfg.think, True)
        self.assertEqual(cfg.temperature, 0.5)
        self.assertEqual(cfg.chat_model, "llama3.1:8b")

    def test_blank_env_value_is_ignored(self) -> None:
        self.assertEqual(Config.from_env({"CHATBOT_NUM_CTX": "  "}).num_ctx, 8192)

    def test_bad_env_value(self) -> None:
        with self.assertRaises(ConfigError):
            Config.from_env({"CHATBOT_NUM_CTX": "lots"})
        with self.assertRaises(ConfigError):
            Config.from_env({"CHATBOT_THINK": "maybe"})

    def test_validation(self) -> None:
        with self.assertRaises(ConfigError):
            Config(ollama_host="localhost:11434")
        with self.assertRaises(ConfigError):
            Config(num_ctx=100)
        with self.assertRaises(ConfigError):
            Config(read_timeout_s=0)


if __name__ == "__main__":
    unittest.main()
