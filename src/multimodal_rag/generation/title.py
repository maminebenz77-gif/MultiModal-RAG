"""Generates a short title for a new conversation, from its first turn --
shown in the "previous conversations" picker instead of the raw first
question (see api/db.py's conversations.title column).
"""

from ..providers.factory import get_llm

_SYSTEM_PROMPT = (
    "Give this conversation a short, specific title -- 3 to 6 words, plain "
    "text, no quotation marks and no punctuation at the end. Output ONLY "
    "the title, nothing else."
)


def generate_title(question: str, answer: str) -> str | None:
    """Optional one-line title for a conversation. Returns None if no LLM
    provider is configured, the call fails, or the provider doesn't
    support it (e.g. InternalServerLLM's plain-generate stub) -- a title
    is a nice-to-have, never a reason to fail the actual answer it's
    generated alongside.
    """
    try:
        llm = get_llm()
    except Exception:
        return None
    try:
        title = llm.generate(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"User: {question}\nAssistant: {answer}"},
            ]
        )
    except Exception:
        return None
    title = title.strip()
    return title or None
