"""Shared image/chart description logic used by every format parser.

Multi-vector pattern: raw image bytes are kept on the Element for display,
while a text description (from the vision provider) is what actually gets
embedded and searched later. A query matches the description; the answer
shows the real image.
"""

from typing import Literal

from ..providers.base import VisionProvider
from ..providers.factory import get_vision


class ImageDescriber:
    """Wraps get_vision() once per document rather than once per image, so
    a missing/broken VisionProvider is detected a single time instead of
    raising repeatedly for every image in the document.
    """

    def __init__(self) -> None:
        self._provider: VisionProvider | None = None
        self._unavailable_reason: str | None = None
        try:
            self._provider = get_vision()
        except NotImplementedError as exc:
            self._unavailable_reason = str(exc)

    def describe(self, image_bytes: bytes) -> tuple[str, Literal["generated", "placeholder"]]:
        if self._provider is None:
            return (
                f"[vision description unavailable: {self._unavailable_reason}]",
                "placeholder",
            )
        try:
            return self._provider.describe(image_bytes), "generated"
        except Exception as exc:
            # A single bad image (corrupt bytes, transient provider failure)
            # shouldn't take down ingestion of the whole document.
            return f"[vision description failed: {exc}]", "placeholder"
