"""Shared vector similarity helpers.

Pure Python (no numpy) — the vectors involved here are small enough
(sentence-level, or a handful of chunks at a time) that a dependency
isn't worth it, and it keeps the mechanism transparent.
"""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
