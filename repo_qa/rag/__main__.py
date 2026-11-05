"""Chat with an indexed repository from the command line.

    python -m repo_qa.rag data/index/miguelgrinberg__microblog
    python -m repo_qa.rag data/index/miguelgrinberg__microblog --ask "Where is authentication handled?"
    python -m repo_qa.rag data/index/miguelgrinberg__microblog --no-graph      # plain search, for comparison

Build the index first with: python -m repo_qa.index build <repo>.

Inside the chat:  /graph on|off   /mode hybrid|vector|keyword   /sources   /context   /clear   /help   /quit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

from common.backends import ChatBackend, Embedder, create_chat_client, create_embedder
from common.config import Config
from common.errors import ChatbotError, IngestError
from repo_qa.index.index import RepoIndex
from repo_qa.index.search import MODES, HybridSearcher

from .answer import Answer, Answerer
from .prompts import format_context
from .retrieve import Retriever

HELP = """Commands:
  /graph on|off     use the knowledge graph to find related code (default on)
  /mode <m>         search mode: hybrid, vector or keyword
  /effort <e>       effort/depth level: low, medium or high (default medium)
  /sources          show the sources used for the last answer
  /context          show exactly what the model was given for the last answer
  /clear            forget the conversation so far
  /quit             leave"""
MAX_HISTORY = 6


def describe_sources(answer: Answer) -> list[str]:
    lines = []
    for item in answer.retrieval.items:
        c = item.chunk
        lines.append(f"  [{item.origin}] {c.citation}  {c.kind} {c.name}  - {item.reason}")
    return lines


def main(
    argv: list[str] | None = None,
    *,
    chat_factory: Callable[[Config], ChatBackend] = create_chat_client,
    embedder_factory: Callable[[Config], Embedder] = create_embedder,
    input_fn: Callable[[str], str] = input,
    write: Callable[[str], None] = sys.stdout.write,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m repo_qa.rag", description=__doc__.split("\n\n")[0])
    parser.add_argument("index_dir", type=Path, help="folder created by: python -m repo_qa.index build")
    parser.add_argument("--ask", metavar="QUESTION", help="answer one question and exit")
    parser.add_argument("--no-graph", action="store_true", help="plain search without graph expansion")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--show-context", action="store_true", help="print the facts and chunks given to the model")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    def say(text: str = "") -> None:
        write(text + "\n")

    embedder: Embedder | None = None
    chat: ChatBackend | None = None
    try:
        cfg = Config.from_env()
        index = RepoIndex.load(args.index_dir)
        if index.has_vectors:
            embedder = embedder_factory(cfg)
            embedder.health_check()
            assert index.store is not None
            if index.store.model != embedder.model_name:
                raise IngestError(
                    f"The index was built with embedding model {index.store.model!r} but {embedder.model_name!r} "
                    "is configured. Re-index the repository."
                )
        elif args.mode != "keyword":
            say("Note: this index has no vectors, so only keyword search is used.")
        chat = chat_factory(cfg)
        chat.health_check()
        answerer = Answerer(Retriever(index, HybridSearcher(index, embedder, cfg), cfg), chat, cfg)
        return _run(args, answerer, index, say, input_fn, write)
    except (ChatbotError, ValueError) as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    finally:
        for resource in (chat, embedder):
            if resource is not None:
                resource.close()


def _run(args, answerer: Answerer, index: RepoIndex, say, input_fn, write) -> int:
    use_graph = not args.no_graph
    mode = args.mode
    effort = getattr(args, "effort", "medium")
    history: list[tuple[str, str]] = []
    last: Answer | None = None

    def ask(question: str) -> Answer:
        write("\n")
        result = answerer.ask(question, history, use_graph=use_graph, mode=mode, effort=effort, on_token=write)
        say("\n")
        if result.standalone_question != result.question:
            say(f"(understood as: {result.standalone_question})")
        say("Sources used:")
        for line in describe_sources(result):
            say(line)
        if result.retrieval.dropped:
            say(f"  ({result.retrieval.dropped} more related chunks did not fit the context budget)")
        if result.invalid_citations:
            say("WARNING: the answer cites locations that were not in its context: "
                + ", ".join(result.invalid_citations) + "  (treat these as unverified)")
        if args.show_context:
            say("\n--- context given to the model ---\n" + format_context(result.retrieval))
        return result

    if args.ask:
        try:
            ask(args.ask)
        except ValueError as exc:
            print(f"ERROR (ValueError): {exc}", file=sys.stderr)
            return 1
        return 0

    say(f"Chatting with {index.repo_name} ({len(index.chunks)} chunks, graph {'on' if use_graph else 'off'}). /help for commands.")
    while True:
        try:
            line = input_fn("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            say()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            command, _, rest = line.partition(" ")
            rest = rest.strip().lower()
            if command in ("/quit", "/exit", "/q"):
                return 0
            elif command == "/help":
                say(HELP)
            elif command == "/graph" and rest in ("on", "off"):
                use_graph = rest == "on"
                say(f"Graph expansion {'on' if use_graph else 'off'}.")
            elif command == "/mode" and rest in MODES:
                mode = rest
                say(f"Search mode: {mode}.")
            elif command == "/effort" and rest in ("low", "medium", "high"):
                effort = rest
                say(f"Effort level: {effort}.")
            elif command == "/clear":
                history.clear()
                say("Conversation cleared.")
            elif command == "/sources":
                say("\n".join(describe_sources(last)) if last else "Nothing yet: ask a question first.")
            elif command == "/context":
                say(format_context(last.retrieval) if last else "Nothing yet: ask a question first.")
            else:
                say("Unknown or incomplete command. /help lists them.")
            continue
        try:
            last = ask(line)
            history.append((line, last.text))
            del history[:-MAX_HISTORY]
        except (ChatbotError, ValueError) as exc:  # a failed question must not end the session
            say(f"\nError ({type(exc).__name__}): {exc}")


if __name__ == "__main__":
    sys.exit(main())
