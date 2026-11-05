"""Client-side guards for cloud backends: a rate limiter and a session budget.

Free cloud tiers have low request-per-minute and request-per-day limits. Staying
under them locally is cheaper and clearer than being rejected by the server, and the
budget guard stops a runaway loop (for example during an evaluation run) from burning
through a quota.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

from .errors import BudgetExceededError, RateLimitError


class RateLimiter:
    """Sliding-window limiter: at most max_calls calls in any period_s seconds.

    acquire() blocks until a slot is free, but never longer than max_wait_s; past that
    it raises RateLimitError instead of hanging. max_calls <= 0 disables the limiter.
    The clock and sleep functions can be replaced, which keeps the tests instant.
    """

    def __init__(
        self,
        max_calls: int,
        period_s: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        self.max_calls = max_calls
        self.period_s = period_s
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self, max_wait_s: float) -> float:
        """Wait for a free slot and take it. Returns the seconds spent waiting."""
        if self.max_calls <= 0:
            return 0.0
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                while self._calls and now - self._calls[0] >= self.period_s:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return waited
                wait = max(self._calls[0] + self.period_s - now, 0.01)
            if waited + wait > max_wait_s:
                raise RateLimitError(
                    f"Local rate limit reached ({self.max_calls} requests per "
                    f"{self.period_s:.0f}s). The next slot opens in {wait:.0f}s, which is "
                    f"longer than the allowed wait of {max_wait_s:.0f}s.",
                    retry_after=wait,
                )
            self._sleep(wait)
            waited += wait


class UsageTracker:
    """Counts requests and tokens for one process, with optional hard caps (0 = none).

    The token cap is checked before each request, so a single request can overshoot it
    (token counts are only known after the reply).
    """

    def __init__(self, max_requests: int = 0, max_tokens: int = 0) -> None:
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.requests = 0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def check(self) -> None:
        """Raise BudgetExceededError if another request would exceed a cap."""
        with self._lock:
            if self.max_requests and self.requests >= self.max_requests:
                raise BudgetExceededError(
                    f"Session request budget used up ({self.requests}/{self.max_requests}). "
                    "Raise CHATBOT_SESSION_MAX_REQUESTS or restart the app."
                )
            if self.max_tokens and self.total_tokens >= self.max_tokens:
                raise BudgetExceededError(
                    f"Session token budget used up ({self.total_tokens}/{self.max_tokens}). "
                    "Raise CHATBOT_SESSION_MAX_TOKENS or restart the app."
                )

    def record(self, prompt_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record one completed request and its token usage."""
        with self._lock:
            self.requests += 1
            self.prompt_tokens += max(prompt_tokens, 0)
            self.output_tokens += max(output_tokens, 0)

    def summary(self) -> str:
        return (
            f"{self.requests} request(s), {self.prompt_tokens} prompt + "
            f"{self.output_tokens} output = {self.total_tokens} tokens"
        )
