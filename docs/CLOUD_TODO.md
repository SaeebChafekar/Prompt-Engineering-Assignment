# Cloud (Gemini) backend: status and how to resume

**Status: parked.** The Gemini backend is written and passes its tests against a fake
server, but it has never completed a run against the real API on the author's machine
(it stalled at the streaming-chat step of `check_setup.py`). Local Ollama is the default
and everything else in the project works without the cloud.

## What is already in place
- `common/gemini_client.py`: REST client (key in the `x-goog-api-key` header, retries with
  backoff, rate limiter, quota guard, safety-block handling, JSON mode).
- `common/backends.py`: `CHATBOT_BACKEND=gemini` switches chat to Gemini. Embeddings stay
  local, so switching never forces a re-index.
- `debug_gemini.py`: talks to the API directly, bypassing our client, with timestamps.
- Fixed already: streaming used to wait for the connection to close (512-byte read
  buffering). The client now reads each line as it arrives and stops at `finishReason`.

## Gemini 3.8 Flash (the model this project targets)
The author's account cannot use older Gemini versions, so the defaults target `gemini-3.8-flash`
(Google GA, 2 Sept 2026). What this changed in the code:
- `gemini_model` defaults to `gemini-3.8-flash` (override with `CHATBOT_GEMINI_MODEL`).
- Thinking is a LEVEL, not a token budget: `CHATBOT_GEMINI_THINKING_LEVEL` = `low` (default,
  fastest allowed), `medium`, `high`, or `none` (send nothing; the model then uses medium).
  `minimal` is rejected by the API for 3.8 Flash, so the config refuses it with a clear message.
  The old numeric `thinkingBudget` is no longer sent (that mismatch was a likely cause of the
  original stall/empty replies).
- Temperature is not sent unless configured (`CHATBOT_GEMINI_TEMPERATURE`, 0 to 2): Gemini 3
  guidance is to leave it at the model default. Per-call `temperature=` is still honoured.
- The output limit is `CHATBOT_GEMINI_MAX_OUTPUT_TOKENS` (default 4096) because thinking tokens
  count against it; tiny limits can produce empty replies.
- Quota and pricing for this model were not confirmed from Google's own pages; check your limits
  in AI Studio and set `CHATBOT_GEMINI_RPM` accordingly.

## Steps to resume (in order, stop at the first failure and read the message)
1. `export GEMINI_API_KEY=...` (never commit it), then `python debug_gemini.py`.
   - 429: no quota for this model/account (check AI Studio limits) or a rate limit;
     `python check_setup.py` lists the models your key can use (`CHATBOT_GEMINI_MODEL`).
   - 400/403: key problem or key restrictions. 404: wrong model name.
   - Normal request works but streaming stalls: a proxy/antivirus is buffering; try
     another network.
   - Both work: the bug is in our client; go to step 2.
2. `CHATBOT_BACKEND=gemini CHATBOT_GEMINI_MAX_RETRIES=0 python check_setup.py`
   and read the first error (retries are off so nothing waits silently).
3. If the API shape differs from what the code expects, compare a raw streamed event
   from `debug_gemini.py` with the parsing in `GeminiClient._iter_stream`.
4. Free-tier limits are low (check yours in AI Studio and set `CHATBOT_GEMINI_RPM`).
   Do not generate per-file LLM summaries on a free key for large repositories.

## Rule while it is parked
Do not let new code depend on the cloud. Use `create_chat_client()` (never `OllamaClient`
or `GeminiClient` directly) so the backend stays swappable. Retrieval limits per backend
live in `Config.retrieval_profile`.
