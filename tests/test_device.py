from pytest import MonkeyPatch

from multimodal_rag.device import resolve_device


def test_explicit_override_passes_through_without_detection() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_auto_prefers_cuda_when_available(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_auto_falls_back_to_mps_when_no_cuda(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert resolve_device("auto") == "mps"


def test_auto_falls_back_to_cpu_when_nothing_available(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
