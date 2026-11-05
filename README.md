# Chatbots project

Two assignment chatbots sharing one foundation:
1. a prompt-based code reviewer (planned),
2. a prompt + RAG + knowledge-graph codebase Q&A bot (in progress).

## Setup
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen3:8b && ollama pull nomic-embed-text   # local models
python check_setup.py                                   # verify everything works
```

## Web interface (both chatbots)
```
streamlit run app.py
```
Open the address it prints (normally http://localhost:8501). If Streamlit asks for an email on the
very first run, just press Enter. The sidebar switches between the two chatbots:

- **Project 1: Code reviewer.** Paste or upload code, optionally describe what is going wrong, and get
  issues grouped by severity with a minimal fix each. Then ask follow-up questions about the review.
- **Project 2: Codebase Q&A.** Index a GitHub URL or local folder, then chat with it. Each answer lists its
  sources, flags citations that were not in the context, and you can switch the knowledge graph on and
  off to compare. The "Explore the knowledge graph" box shows callers and callees of any function.

The models are set with environment variables; the default is local Ollama. For the cloud model start with
`CHATBOT_BACKEND=gemini GEMINI_API_KEY=... streamlit run app.py`. The sidebar shows which model is in use
and has a "Check connection" button.

Project 1 also works from the command line: `python -m code_review path/to/file.py --problem "crashes on empty list"`.

## Try the ingestion stage (works on a GitHub URL or a local folder)
```
python -m repo_qa.ingest https://github.com/owner/repo --show-chunks 3 --out data/debug/repo
python -m repo_qa.ingest path/to/folder
```
It prints what was kept, skipped and parsed. `--out` writes `chunks.jsonl`,
`parsed_python.json`, `files.json` and `summary.json` so you can inspect the chunks.
Limits (file count, size, timeouts) are environment variables; see `common/config.py`.

## Try the knowledge graph
```
python -m repo_qa.graph https://github.com/miguelgrinberg/microblog
python -m repo_qa.graph <repo> --callers verify_password          # who calls it?
python -m repo_qa.graph <repo> --callees "app/auth/routes.py::login"   # what does it call?
python -m repo_qa.graph <repo> --imports app/auth/routes.py       # imports / imported by
python -m repo_qa.graph <repo> --find routes.py
python -m repo_qa.graph <repo> --min-confidence likely            # hide guesses
python -m repo_qa.graph <repo> --out data/graphs/repo --labeling-csv data/calls.csv
```
Every edge is `certain`, `likely` or `guess` (see `repo_qa/graph/resolve.py`). `--labeling-csv`
writes every call decision to a CSV: fill the last column with y/n for a sample to measure
accuracy for your report.

## Build a searchable index and search it
```
python -m repo_qa.index build https://github.com/miguelgrinberg/microblog --keyword-only   # no Ollama needed
python -m repo_qa.index build https://github.com/miguelgrinberg/microblog                  # with embeddings (needs Ollama)
python -m repo_qa.index search data/index/miguelgrinberg__microblog "where is authentication handled?"
python -m repo_qa.index search data/index/miguelgrinberg__microblog "check_password" --mode keyword --show-text
```
`build` ingests, builds the knowledge graph, embeds every chunk with the local model (it prints
progress) and saves to `data/index/<repo>`. Re-running it only embeds chunks that changed.
Search combines meaning-based (vector) and exact-name (keyword) results.

## Chat with a repository
```
python -m repo_qa.rag data/index/miguelgrinberg__microblog                       # interactive chat
python -m repo_qa.rag data/index/miguelgrinberg__microblog --ask "Which API endpoints require a token?"
python -m repo_qa.rag data/index/miguelgrinberg__microblog --no-graph            # plain search, for comparison
python -m repo_qa.rag data/index/miguelgrinberg__microblog --show-context        # print what the model was given
```
Inside the chat: `/graph on|off`, `/mode hybrid|vector|keyword`, `/sources`, `/context`, `/clear`, `/quit`.
Every answer lists its sources, and every `path:lines` citation is checked against the context
the model actually saw (invalid ones are flagged). Cloud model: `CHATBOT_BACKEND=gemini` (see docs/CLOUD_TODO.md).

## Evaluate (for the report)
```
python -m repo_qa.eval validate  data/index/miguelgrinberg__microblog
python -m repo_qa.eval retrieval data/index/miguelgrinberg__microblog     # keyword / vector / hybrid / hybrid+graph
python -m repo_qa.eval answers   data/index/miguelgrinberg__microblog     # slow; --limit N, --resume
python -m repo_qa.eval report    data/index/miguelgrinberg__microblog/eval
```
25 questions with known answers live in `eval/microblog_questions.json`; `eval/microblog_call_sample.csv` is a ready
sample of call edges to label y/n. Full step-by-step guide: `docs/EVALUATION.md`.

## Tests (no Ollama, internet or GPU needed)
```
python -m unittest discover -s tests -t .
```

## Layout
```
common/          config, errors, Ollama + Gemini clients, backend factory, rate limits
repo_qa/ingest/  clone, filters, Python parser, chunker, pipeline, CLI
repo_qa/graph/   schema, graph container, import/call resolution, builder, CLI
repo_qa/index/   keyword index (BM25), numpy vector store, index builder, hybrid search, CLI
repo_qa/rag/     graph-aware retrieval, prompts, answering with citation checks, chat CLI
repo_qa/service.py   indexing and chat sessions shared by the web page and the command line
code_review/     Project 1: code reviewer (output schema, prompts, reviewer, renderer, CLI)
webapp/          Streamlit pages; app.py is the entry point
repo_qa/eval/    evaluation harness (questions, metrics, reports, graph accuracy); data in eval/
tests/           unit tests with fake servers and temporary repositories
docs/            CLOUD_TODO.md (how to resume the Gemini backend)
```
