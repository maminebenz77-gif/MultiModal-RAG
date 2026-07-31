"""Rewrites a possibly context-dependent follow-up question into a
standalone one, using the conversation history so far.

Retrieval only ever sees the CURRENT query string — it has no memory of
earlier turns. A follow-up like "How does that compare to the local
option?" retrieves nothing useful on its own, because "that" isn't in
the corpus. Rewriting it first to "How does the internal gateway's
average latency compare to the local option's average latency?" gives
retrieval something it can actually match against. This is a separate,
narrowly-scoped LLM call — it does not feed the conversation history
into the final answer prompt, keeping that prompt's "only answer from
the retrieved context" contract unchanged.
"""

from ..providers.factory import get_llm

_REWRITE_SYSTEM_PROMPT = """Given a conversation history and a latest user message, rewrite the \
latest message into a fully self-contained question that can be understood WITHOUT the \
conversation history — resolve pronouns and implicit references (e.g. "that", "it", "the other \
one") into what they actually refer to. Preserve the original meaning and intent exactly; do not \
add information that wasn't implied. If the latest message is already self-contained, return it \
unchanged. Output ONLY the rewritten question, nothing else — no preamble, no quotes."""


def build_rewrite_messages(
    history: list[tuple[str, str]], latest_query: str
) -> list[dict[str, str]]:
    transcript = "\n".join(f"User: {question}\nAssistant: {answer}" for question, answer in history)
    user_content = (
        f"Conversation so far:\n{transcript}\n\n"
        f"Latest user message: {latest_query}\n\n"
        "Rewritten standalone question:"
    )
    return [
        {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def rewrite_query(query: str, history: list[tuple[str, str]]) -> str:
    """Returns `query` unchanged if there's no history — a first turn is
    standalone by definition, so skip the extra LLM call entirely."""
    if not history:
        return query
    messages = build_rewrite_messages(history, query)
    return get_llm().generate(messages).strip()
