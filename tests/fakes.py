"""Fake model backends shared by the tests (no Ollama, Gemini or GPU needed)."""

from __future__ import annotations

import json

from common.errors import InvalidOutputError, LLMError


class FakeChat:
    """A scripted chat backend. Records every call; can fail on chosen call numbers."""

    backend_name = "fake"
    model_name = "fake-chat"

    def __init__(self, replies=None, fail_on=()) -> None:
        self.replies = list(replies or ["It is handled in [app/auth.py:7-10]."])
        self.fail_on = set(fail_on)
        self.calls: list[dict] = []
        self.closed = False

    def _reply(self, messages, system, kind, kw=None):
        self.calls.append({"kind": kind, "system": system, "messages": messages, "kw": kw or {}})
        if len(self.calls) in self.fail_on:
            raise LLMError("simulated model failure")
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]

    def chat_stream(self, messages, *, system=None, **kw):
        text = self._reply(messages, system, "stream", kw)
        words = text.split(" ")
        return iter([w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words)])

    def chat(self, messages, *, system=None, **kw):
        return self._reply(messages, system, "chat", kw)

    def chat_json(self, messages, schema, *, system=None, **kw):
        """Replies are JSON text (or a ready dict) in `replies`; invalid text raises like a real backend."""
        reply = self._reply(messages, system, "json", {"schema": schema, **kw})
        if isinstance(reply, dict):
            return reply
        try:
            return json.loads(reply)
        except json.JSONDecodeError as exc:
            raise InvalidOutputError(f"Model output was not valid JSON: {reply[:80]!r}") from exc

    def health_check(self) -> None:
        pass

    def list_models(self):
        return ["fake-chat"]

    def close(self) -> None:
        self.closed = True
