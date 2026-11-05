"""Tests for the rate limiter, session budget, backend config and factories.

Run from the project root:

    python -m unittest tests.test_limits_and_backends -v
"""

from __future__ import annotations

import unittest

from common.backends import OllamaEmbedder, create_chat_client, create_embedder
from common.config import CLOUD_PROFILE, LOCAL_PROFILE, Config
from common.errors import BudgetExceededError, ConfigError, RateLimitError
from common.gemini_client import GeminiClient
from common.limits import RateLimiter, UsageTracker
from common.llm_client import OllamaClient


class FakeClock:
    """A clock whose sleep() advances time, so limiter tests run instantly."""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


class RateLimiterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()

    def limiter(self, max_calls: int, period: float = 60.0) -> RateLimiter:
        return RateLimiter(max_calls, period, clock=self.clock.now, sleep=self.clock.sleep)

    def test_burst_within_limit_does_not_wait(self) -> None:
        lim = self.limiter(2)
        self.assertEqual(lim.acquire(10), 0.0)
        self.assertEqual(lim.acquire(10), 0.0)
        self.assertEqual(self.clock.slept, [])

    def test_waits_for_slot_when_full(self) -> None:
        lim = self.limiter(2)
        lim.acquire(100)
        lim.acquire(100)
        waited = lim.acquire(100)
        self.assertEqual(waited, 60.0)
        self.assertEqual(self.clock.slept, [60.0])

    def test_raises_instead_of_waiting_too_long(self) -> None:
        lim = self.limiter(1)
        lim.acquire(100)
        with self.assertRaises(RateLimitError) as ctx:
            lim.acquire(5)
        self.assertEqual(ctx.exception.retry_after, 60.0)
        self.assertEqual(self.clock.slept, [])

    def test_window_slides(self) -> None:
        lim = self.limiter(1)
        lim.acquire(100)
        self.clock.t = 61.0
        self.assertEqual(lim.acquire(1), 0.0)

    def test_zero_disables(self) -> None:
        lim = self.limiter(0)
        for _ in range(100):
            self.assertEqual(lim.acquire(0), 0.0)

    def test_invalid_period(self) -> None:
        with self.assertRaises(ValueError):
            RateLimiter(5, 0)


class UsageTrackerTests(unittest.TestCase):
    def test_unlimited_by_default(self) -> None:
        usage = UsageTracker()
        for _ in range(50):
            usage.check()
            usage.record(1000, 1000)
        self.assertEqual(usage.requests, 50)
        self.assertEqual(usage.total_tokens, 100_000)

    def test_request_cap(self) -> None:
        usage = UsageTracker(max_requests=2)
        usage.record()
        usage.check()
        usage.record()
        with self.assertRaises(BudgetExceededError):
            usage.check()

    def test_token_cap(self) -> None:
        usage = UsageTracker(max_tokens=100)
        usage.record(60, 30)
        usage.check()
        usage.record(5, 10)
        with self.assertRaises(BudgetExceededError):
            usage.check()

    def test_summary(self) -> None:
        usage = UsageTracker()
        usage.record(10, 5)
        self.assertIn("1 request", usage.summary())
        self.assertIn("15 tokens", usage.summary())


class BackendConfigTests(unittest.TestCase):
    def test_default_backend_is_local(self) -> None:
        cfg = Config()
        self.assertEqual(cfg.backend, "ollama")
        self.assertEqual(cfg.retrieval_profile, LOCAL_PROFILE)

    def test_gemini_requires_key(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            Config(backend="gemini")
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_gemini_with_key_uses_cloud_profile(self) -> None:
        cfg = Config(backend="gemini", gemini_api_key="k")
        self.assertEqual(cfg.retrieval_profile, CLOUD_PROFILE)
        self.assertGreater(CLOUD_PROFILE.context_tokens, LOCAL_PROFILE.context_tokens)

    def test_unknown_backend_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            Config(backend="openai")

    def test_key_not_in_repr_or_str(self) -> None:
        cfg = Config(backend="gemini", gemini_api_key="super-secret-key")
        self.assertNotIn("super-secret-key", repr(cfg))
        self.assertNotIn("super-secret-key", str(cfg))

    def test_standard_env_var_is_honored(self) -> None:
        cfg = Config.from_env({"CHATBOT_BACKEND": "gemini", "GEMINI_API_KEY": " abc "})
        self.assertEqual(cfg.gemini_api_key, "abc")

    def test_prefixed_env_var_wins(self) -> None:
        cfg = Config.from_env(
            {
                "CHATBOT_BACKEND": "gemini",
                "CHATBOT_GEMINI_API_KEY": "prefixed",
                "GEMINI_API_KEY": "standard",
            }
        )
        self.assertEqual(cfg.gemini_api_key, "prefixed")

    def test_env_overrides_gemini_settings(self) -> None:
        cfg = Config.from_env(
            {
                "CHATBOT_GEMINI_MODEL": "gemini-test",
                "CHATBOT_GEMINI_RPM": "5",
                "CHATBOT_GEMINI_THINKING_LEVEL": "high",
            }
        )
        self.assertEqual(cfg.gemini_model, "gemini-test")
        self.assertEqual(cfg.gemini_rpm, 5)
        self.assertEqual(cfg.gemini_thinking_level, "high")

    def test_gemini_defaults_target_gemini_3_8_flash(self) -> None:
        cfg = Config()
        self.assertEqual(cfg.gemini_model, "gemini-3.8-flash")
        self.assertEqual(cfg.gemini_thinking_level, "low")
        self.assertLess(cfg.gemini_temperature, 0)  # negative = do not send, use the model default

    def test_thinking_level_validation(self) -> None:
        for ok in ("low", "medium", "high", "none"):
            self.assertEqual(Config(gemini_thinking_level=ok).gemini_thinking_level, ok)
        with self.assertRaises(ConfigError) as ctx:
            Config(gemini_thinking_level="minimal")  # rejected by the Gemini 3.8 Flash API
        self.assertIn("not supported by Gemini 3.8 Flash", str(ctx.exception))
        with self.assertRaises(ConfigError):
            Config(gemini_thinking_level="extreme")
        with self.assertRaises(ConfigError):
            Config(gemini_temperature=3.0)
        with self.assertRaises(ConfigError):
            Config(gemini_max_output_tokens=0)

    def test_remote_http_base_url_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            Config(gemini_base_url="http://example.com")
        Config(gemini_base_url="http://127.0.0.1:9999")  # local test servers are fine
        Config(gemini_base_url="https://generativelanguage.googleapis.com")

    def test_negative_limits_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            Config(gemini_rpm=-1)
        with self.assertRaises(ConfigError):
            Config(session_max_tokens=-5)


class FactoryTests(unittest.TestCase):
    def test_factory_returns_expected_types(self) -> None:
        local = create_chat_client(Config())
        cloud = create_chat_client(Config(backend="gemini", gemini_api_key="k"))
        try:
            self.assertIsInstance(local, OllamaClient)
            self.assertEqual(local.backend_name, "ollama")
            self.assertEqual(local.model_name, "qwen3:8b")
            self.assertIsInstance(cloud, GeminiClient)
            self.assertEqual(cloud.backend_name, "gemini")
            self.assertEqual(cloud.model_name, "gemini-3.8-flash")
        finally:
            local.close()
            cloud.close()

    def test_embedder_is_local_even_with_gemini_chat(self) -> None:
        embedder = create_embedder(Config(backend="gemini", gemini_api_key="k"))
        try:
            self.assertIsInstance(embedder, OllamaEmbedder)
            self.assertEqual(embedder.model_name, "nomic-embed-text")
        finally:
            embedder.close()

    def test_gemini_client_refuses_missing_key(self) -> None:
        with self.assertRaises(ConfigError):
            GeminiClient(Config())  # default config has no key


if __name__ == "__main__":
    unittest.main()
