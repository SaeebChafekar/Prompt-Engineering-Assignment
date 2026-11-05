"""Answer a question about the repository: retrieve, prompt, stream, verify.

    question (+ history) -> standalone question -> Retriever -> prompt -> model -> Answer

After the model replies, every `path:lines` citation in the answer is checked against the
context that was actually shown. Citations that point at files or lines the model never saw
are reported as invalid, so a made-up reference cannot pass as evidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

from common.backends import ChatBackend
from common.config import Config
from common.errors import InvalidOutputError

from .prompts import (
    NOT_FOUND_TEXT,
    REWRITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_rewrite_message,
    build_user_message,
    get_rag_system_prompt,
)
from .retrieve import Retrieval, Retriever, verify_citations

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000
MAX_REWRITE_CHARS = 400
Turn = tuple[str, str]  # (question, answer)


@dataclass
class Answer:
    question: str  # as the user typed it
    standalone_question: str  # after resolving "it", "that function" ... (same if no history)
    text: str
    retrieval: Retrieval
    cited: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    effort: str = "medium"

    @property
    def not_found(self) -> bool:
        return NOT_FOUND_TEXT.lower().rstrip(".") in self.text.lower()

    @property
    def citations_ok(self) -> bool:
        return not self.invalid_citations


class Answerer:
    def __init__(self, retriever: Retriever, chat: ChatBackend, cfg: Config) -> None:
        self.retriever = retriever
        self.chat = chat
        self.cfg = cfg

    def ask(
        self,
        question: str,
        history: Sequence[Turn] = (),
        *,
        use_graph: bool = True,
        mode: str = "hybrid",
        effort: str = "medium",
        on_token: Callable[[str], None] | None = None,
    ) -> Answer:
        """Answer one question. Raises ValueError for empty or oversized input, and
        ChatbotError subclasses for model, search or timeout failures."""
        text = question.strip()
        if not text:
            raise ValueError("The question is empty.")
        if len(text) > MAX_QUESTION_CHARS:
            raise ValueError(f"The question is too long ({len(text)} characters; the limit is {MAX_QUESTION_CHARS}).")

        standalone = self._standalone(text, list(history))
        retrieval = self.retriever.retrieve(standalone, use_graph=use_graph, mode=mode, effort=effort)
        user_message = build_user_message(self.retriever.index.repo_name, standalone, retrieval)
        system_prompt = get_rag_system_prompt(effort)
        max_tokens = 4096 if effort == "high" else None

        pieces: list[str] = []
        for piece in self.chat.chat_stream([{"role": "user", "content": user_message}], system=system_prompt, max_tokens=max_tokens):
            pieces.append(piece)
            if on_token:
                on_token(piece)
        answer_text = "".join(pieces).strip()
        if not answer_text:
            raise InvalidOutputError("The model returned an empty answer. Try asking again.")

        cited, invalid = verify_citations(answer_text, retrieval.items)
        return Answer(text, standalone, answer_text, retrieval, cited, invalid, effort=effort)

    def _standalone(self, question: str, history: list[Turn]) -> str:
        """Rewrite a follow-up so retrieval sees the real subject. Falls back to the original."""
        if not history:
            return question
        try:
            rewritten = self.chat.chat(
                [{"role": "user", "content": build_rewrite_message(history, question)}],
                system=REWRITE_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=512,
            )
        except Exception as exc:  # the rewrite is a nicety: never let it break the answer
            log.warning("Could not rewrite the follow-up question (%s); using it as typed", exc)
            return question
        rewritten = rewritten.strip().strip('"').strip()
        if not rewritten or "\n" in rewritten or len(rewritten) > MAX_REWRITE_CHARS:
            return question
        return rewritten
