"""Shared retry-with-backoff helper for batched network operations
(embedding API calls, vector store upserts, ...). The retry mechanics are
identical everywhere; what a caller does with a persistent failure
(re-raise with what succeeded so far, drop the batch, etc.) is domain-
specific and stays with the caller.
"""

import time
from collections.abc import Callable


def retry_with_backoff[T](
    operation: Callable[[], T], max_retries: int = 3, backoff_seconds: float = 1.0
) -> T:
    """Call `operation`, retrying up to max_retries times with exponential
    backoff between attempts. Re-raises the last exception if every
    attempt fails."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff_seconds * (2**attempt))
    assert last_exc is not None
    raise last_exc
