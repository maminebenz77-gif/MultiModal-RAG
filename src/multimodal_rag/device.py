"""Resolves which compute device to run local models on.

Device is a hardware fact, not a preference — so "auto" is the default and
does real detection, while an explicit override (e.g. forcing "cpu" for a
reproducibility check) always wins.
"""

import torch


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
