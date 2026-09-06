import io

import pytest
from PIL import Image

from multimodal_rag.image_utils import downscale_image


def _make_png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color="blue").save(buffer, format="PNG")
    return buffer.getvalue()


def test_small_image_is_returned_unchanged() -> None:
    small = _make_png(100, 100)
    assert downscale_image(small, max_dimension=1024) == small


def test_large_image_is_shrunk_to_max_dimension() -> None:
    large = _make_png(3000, 1500)
    result = downscale_image(large, max_dimension=1024)

    assert result != large
    resized = Image.open(io.BytesIO(result))
    assert max(resized.width, resized.height) == 1024


def test_aspect_ratio_is_preserved() -> None:
    large = _make_png(3000, 1500)  # 2:1
    result = downscale_image(large, max_dimension=1024)
    resized = Image.open(io.BytesIO(result))
    assert resized.width / resized.height == pytest.approx(2.0, rel=0.02)


def test_undecodable_bytes_are_returned_unchanged() -> None:
    garbage = b"not an image at all"
    assert downscale_image(garbage, max_dimension=1024) == garbage
