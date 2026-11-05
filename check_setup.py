"""Smoke test for your setup, for either chat backend.

Run from the project root:

    python check_setup.py                         # local Ollama (default)
    CHATBOT_BACKEND=gemini python check_setup.py  # Gemini cloud (needs GEMINI_API_KEY)

It checks, in order: configuration, chat backend reachable and model available,
embedding model available, a short streamed chat (with timing), structured JSON
output, and embeddings (always local). Exit code 0 means everything passed; 1 means a
step failed (the message says how to fix it); 2 means bad configuration.

Note: with the Gemini backend this makes about 2 chat requests, which count against
your free-tier quota.
"""

from __future__ import annotations

import logging
import sys
import time

from common.backends import create_chat_client, create_embedder
from common.config import Config
from common.errors import ChatbotError

LANGUAGE_SCHEMA = {
    "type": "object",
    "properties": {"language": {"type": "string"}},
    "required": ["language"],
}


def _step(title: str) -> None:
    print(f"\n[{title}]")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    try:
        cfg = Config.from_env()
    except ChatbotError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2

    print("Settings")
    print(f"  chat backend : {cfg.backend}")
    if cfg.backend == "gemini":
        print(f"  chat model   : {cfg.gemini_model}   thinking: {cfg.gemini_thinking_level}")
        print(f"  api key      : set ({len(cfg.gemini_api_key)} characters, hidden)")
        print(f"  rate limit   : {cfg.gemini_rpm or 'off'} requests/min (client-side)")
    else:
        print(f"  chat model   : {cfg.chat_model}")
        print(f"  host         : {cfg.ollama_host}")
        print(f"  num_ctx      : {cfg.num_ctx}   think: {cfg.think}")
    print(f"  embed model  : {cfg.embed_model} (local Ollama)")
    profile = cfg.retrieval_profile
    print(f"  retrieval    : top_k={profile.top_k}, graph_hops={profile.graph_hops}, "
          f"context_tokens={profile.context_tokens}")

    try:
        chat = create_chat_client(cfg)
        embedder = create_embedder(cfg)
    except ChatbotError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2

    try:
        try:
            _step("1/5 Chat backend and model")
            names = chat.list_models()
            shown = ", ".join(names[:8]) + (" ..." if len(names) > 8 else "")
            print(f"  OK, {len(names)} model(s) available: {shown or 'none'}")
            chat.health_check()
            print(f"  OK, '{chat.model_name}' is available")

            _step("2/5 Embedding model (local Ollama)")
            embedder.health_check()
            print(f"  OK, '{embedder.model_name}' is installed")

            _step("3/5 Streaming chat (first Ollama call may be slow: the model is loading)")
            start = time.monotonic()
            first_token_at: float | None = None
            pieces: list[str] = []
            for piece in chat.chat_stream(
                [{"role": "user", "content": "Reply with exactly one word: ready"}],
            ):
                if first_token_at is None:
                    first_token_at = time.monotonic() - start
                pieces.append(piece)
            total = time.monotonic() - start
            reply = "".join(pieces).strip()
            if not reply:
                print("  FAIL: the model returned an empty reply")
                return 1
            print(f"  OK, reply: {reply!r}")
            print(f"  first token after {first_token_at:.1f}s, finished after {total:.1f}s")

            _step("4/5 Structured JSON output")
            data = chat.chat_json(
                [{"role": "user", "content": "Which programming language is this? print('hi')"}],
                LANGUAGE_SCHEMA,
            )
            if not isinstance(data, dict) or "language" not in data:
                print(f"  FAIL: unexpected JSON: {data!r}")
                return 1
            print(f"  OK, parsed: {data}")

            _step("5/5 Embeddings (local)")
            start = time.monotonic()
            vectors = embedder.embed(["def login(user): ...", "where is authentication handled?"])
            print(f"  OK, {len(vectors)} vectors of dimension {len(vectors[0])} "
                  f"in {time.monotonic() - start:.1f}s")

        except ChatbotError as exc:
            print(f"  FAIL ({type(exc).__name__}): {exc}")
            return 1

        usage = getattr(chat, "usage", None)
        if usage is not None:
            print(f"\nCloud usage this run: {usage.summary()}")
    finally:
        chat.close()
        embedder.close()

    print("\nAll checks passed.")
    if cfg.backend == "ollama":
        print("Tip: run `ollama ps` now. The PROCESSOR column should say 100% GPU.")
        print("If it shows a CPU/GPU split, lower CHATBOT_NUM_CTX (e.g. 4096) for more speed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
