"""Shared image helpers -- used by the vision provider (downscaling
before sending to a vision API) and by chunking (downscaling before
persisting a thumbnail alongside a chunk, see chunking/schema.py's
ChunkElement). One implementation, two call sites with different
size budgets, rather than two copies of the same resize logic.
"""

import io

from PIL import Image


def downscale_image(image_bytes: bytes, max_dimension: int) -> bytes:
    """Cap the image's longest edge at max_dimension. Returns the
    original bytes unchanged if the image is already small enough, or
    if it can't be decoded (fails closed to "use as-is" rather than
    breaking the caller over a resize step that was only ever meant to
    save space/cost).
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception:
        return image_bytes

    if max(image.width, image.height) <= max_dimension:
        return image_bytes

    scale = max_dimension / max(image.width, image.height)
    new_size = (round(image.width * scale), round(image.height * scale))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    resized.save(buffer, format=image.format or "PNG")
    return buffer.getvalue()
