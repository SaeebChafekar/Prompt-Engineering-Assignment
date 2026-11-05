"""Standalone Gemini diagnostic: finds out WHERE a hang or failure happens.

Run from the project root (with GEMINI_API_KEY set):

    python debug_gemini.py

It does not use the chatbot client, retries or rate limiter. It makes three plain
checks with short timeouts and prints exactly what comes back:

    A. can this machine open a connection to the Gemini server?
    B. a normal (non-streaming) request: status code, time, error text
    C. a streaming request: when each line of the reply arrives

The API key is never printed. Each request is tiny (about 20 output tokens), so it
uses very little quota. Exit code: 0 = all good, 1 = a check failed, 2 = bad config.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from typing import Any
from urllib.parse import quote, urlparse

import requests

from common.config import Config
from common.errors import ConfigError

CONNECT_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 30.0
STREAM_DEADLINE_S = 60.0
MAX_STREAM_LINES = 30
SNIPPET_CHARS = 700


def _request_body(cfg: Config, *, with_thinking_config: bool) -> dict[str, Any]:
    generation: dict[str, Any] = {"maxOutputTokens": 512}
    if with_thinking_config:
        generation["thinkingConfig"] = {"thinkingLevel": cfg.gemini_thinking_level}
    return {
        "contents": [{"role": "user", "parts": [{"text": "Reply with exactly one word: ready"}]}],
        "generationConfig": generation,
    }


def _snippet(resp: requests.Response) -> str:
    text = (resp.text or "").strip().replace("\n", " ")
    return text[:SNIPPET_CHARS] + (" ..." if len(text) > SNIPPET_CHARS else "")


def check_connection(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    proxies = [n for n in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY") if os.environ.get(n)]
    if proxies:
        print(f"  note: proxy environment variable(s) set: {', '.join(proxies)}")
    start = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_S):
            pass
    except OSError as exc:
        print(f"  FAIL: cannot open a connection to {host}:{port} ({type(exc).__name__}).")
        print("  Check your internet, DNS, VPN, firewall or college/office proxy.")
        return False
    print(f"  OK, connected to {host}:{port} in {time.monotonic() - start:.2f}s")
    return True


def check_plain_request(url: str, headers: dict[str, str], cfg: Config) -> tuple[int | None, bool]:
    """Returns (status_code or None, thinking_config_was_rejected)."""
    thinking = cfg.gemini_thinking_level != "none"
    for _ in range(2):
        body = _request_body(cfg, with_thinking_config=thinking)
        start = time.monotonic()
        try:
            resp = requests.post(
                url, headers=headers, json=body, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)
            )
        except requests.ReadTimeout:
            print(f"  TIMEOUT: connected, but no answer within {READ_TIMEOUT_S:.0f}s.")
            return None, False
        except requests.RequestException as exc:
            print(f"  FAIL: request error ({type(exc).__name__}).")
            return None, False
        elapsed = time.monotonic() - start
        print(f"  HTTP {resp.status_code} after {elapsed:.1f}s")
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            print(f"  Retry-After header: {retry_after}")
        if resp.status_code == 200:
            try:
                parts = resp.json()["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                print(f"  reply: {text.strip()!r}")
            except (ValueError, KeyError, IndexError, TypeError):
                print(f"  200 OK but unexpected body: {_snippet(resp)}")
            return 200, False
        print(f"  body: {_snippet(resp)}")
        if resp.status_code == 400 and thinking and "thinking" in resp.text.lower():
            print("  The model rejected the thinking option; retrying without it.")
            thinking = False
            continue
        return resp.status_code, False
    return None, True


def check_streaming(url: str, headers: dict[str, str], cfg: Config) -> int | None:
    body = _request_body(cfg, with_thinking_config=cfg.gemini_thinking_level != "none")
    start = time.monotonic()
    try:
        with requests.post(
            url,
            params={"alt": "sse"},
            headers=headers,
            json=body,
            stream=True,
            timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
        ) as resp:
            print(f"  HTTP {resp.status_code} after {time.monotonic() - start:.1f}s (headers received)")
            if resp.status_code != 200:
                print(f"  body: {_snippet(resp)}")
                return resp.status_code
            lines = 0
            for raw in resp.iter_lines(chunk_size=1):  # same as the client: no 512-byte buffering
                elapsed = time.monotonic() - start
                if elapsed > STREAM_DEADLINE_S:
                    print(f"  STOPPED: still streaming after {STREAM_DEADLINE_S:.0f}s.")
                    return None
                if not raw:
                    continue
                lines += 1
                text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                print(f"  +{elapsed:5.1f}s  line {lines}: {text[:110]!r}")
                if '"finishReason"' in text:
                    print("  (final event received; a real client can stop reading here)")
                    break
                if lines >= MAX_STREAM_LINES:
                    print(f"  (stopped after {MAX_STREAM_LINES} lines)")
                    break
            else:
                print(f"  stream closed by the server after {time.monotonic() - start:.1f}s")
            return 200
    except requests.ReadTimeout:
        print(f"  TIMEOUT: stream went silent for {READ_TIMEOUT_S:.0f}s.")
    except requests.RequestException as exc:
        if "timed out" in str(exc).lower():
            print(f"  TIMEOUT: stream went silent for {READ_TIMEOUT_S:.0f}s.")
        else:
            print(f"  FAIL: request error ({type(exc).__name__}).")
    return None


def explain(plain: int | None, stream: int | None) -> None:
    print("\nWhat this means")
    if plain == 200 and stream == 200:
        print("  The API works from this machine, both normal and streaming.")
        print("  So the problem is in how the chatbot client behaves. Run:")
        print("    CHATBOT_BACKEND=gemini CHATBOT_GEMINI_MAX_RETRIES=0 python check_setup.py")
        print("  and send me everything it prints.")
        return
    if plain == 200 and stream is None:
        print("  Normal requests work but streaming stalls. A proxy, antivirus or network")
        print("  filter is probably buffering streamed replies. Try another network (for")
        print("  example a phone hotspot), and send me this output.")
        return
    status = plain if plain is not None else stream
    if status == 429:
        print("  429 = quota or rate limit. Your key works, but this model has no quota left")
        print("  (or none on the free tier). Open Google AI Studio > usage/limits, try a")
        print("  different model (python check_setup.py lists them), or wait and retry.")
        print("  The chatbot was most likely not stuck: it was waiting to retry this error.")
    elif status in (401, 403):
        print("  The key was rejected or restricted. Check it in AI Studio, and that any key")
        print("  restriction allows the Gemini API.")
    elif status == 404:
        print("  The model name is not available. Set CHATBOT_GEMINI_MODEL to a model that")
        print("  python check_setup.py lists for your key.")
    elif status == 400:
        print("  The request was rejected (see the body above). If it mentions thinking, set")
        print("  CHATBOT_GEMINI_THINKING_LEVEL=none.")
    elif status is not None and status >= 500:
        print("  Google's side is overloaded or failing (5xx). Wait a few minutes and retry.")
    else:
        print("  No answer came back in time. Check your connection, VPN and proxy, then")
        print("  send me this output.")


def main() -> int:
    env = dict(os.environ)
    env["CHATBOT_BACKEND"] = "gemini"
    try:
        cfg = Config.from_env(env)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2

    base = cfg.gemini_base_url.rstrip("/")
    url = f"{base}/v1beta/models/{quote(cfg.gemini_model, safe='-._')}:generateContent"
    stream_url = f"{base}/v1beta/models/{quote(cfg.gemini_model, safe='-._')}:streamGenerateContent"
    headers = {"x-goog-api-key": cfg.gemini_api_key, "Content-Type": "application/json"}

    print(f"model: {cfg.gemini_model}")
    print(f"server: {base}")
    print(f"api key: set ({len(cfg.gemini_api_key)} characters, hidden)")

    print("\n[A] Connection to the server")
    if not check_connection(base):
        return 1

    print("\n[B] Normal (non-streaming) request")
    plain, _ = check_plain_request(url, headers, cfg)

    print("\n[C] Streaming request")
    stream = check_streaming(stream_url, headers, cfg)

    explain(plain, stream)
    return 0 if plain == 200 and stream == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
