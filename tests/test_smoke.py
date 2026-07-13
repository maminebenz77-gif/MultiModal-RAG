"""Smoke test: confirms the package is importable and the toolchain is wired up."""

import multimodal_rag


def test_package_importable() -> None:
    assert multimodal_rag is not None
