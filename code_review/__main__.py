"""Review a source file from the command line.

    python -m code_review path/to/file.py
    python -m code_review path/to/file.py --problem "crashes when the list is empty"
    python -m code_review path/to/file.py --json            # machine-readable output
    cat snippet.py | python -m code_review -               # read from standard input
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.backends import create_chat_client
from common.config import Config
from common.errors import ChatbotError

from .render import review_markdown
from .reviewer import MAX_CODE_CHARS, CodeReviewer


def main(argv: list[str] | None = None, *, chat_factory=create_chat_client, stdin=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m code_review", description=__doc__.split("\n\n")[0])
    parser.add_argument("path", help="file to review, or - for standard input")
    parser.add_argument("--problem", help="what is going wrong (error message or expected behaviour)")
    parser.add_argument("--language", help="python, javascript, ... (default: detect)")
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="medium", help="effort level / analysis depth")
    parser.add_argument("--json", action="store_true", help="print the review as JSON")
    args = parser.parse_args(argv)

    chat = None
    try:
        if args.path == "-":
            code, filename = (stdin or sys.stdin).read(MAX_CODE_CHARS * 4), None
        else:
            path = Path(args.path)
            if not path.is_file():
                print(f"ERROR: {path} is not a file.", file=sys.stderr)
                return 1
            if path.stat().st_size > MAX_CODE_CHARS * 4:
                print(f"ERROR: {path} is too large to review in one go.", file=sys.stderr)
                return 1
            code, filename = path.read_text(encoding="utf-8", errors="replace"), path.name
        cfg = Config.from_env()
        chat = chat_factory(cfg)
        chat.health_check()
        result = CodeReviewer(chat, cfg).review(
            code, language=args.language, problem=args.problem, filename=filename, effort=args.effort
        )
    except (ChatbotError, ValueError) as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    finally:
        if chat is not None:
            chat.close()

    if args.json:
        print(json.dumps(result.review.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(review_markdown(result, title=f"Review of {filename or 'standard input'}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
