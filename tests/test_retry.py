import pytest

from multimodal_rag.retry import retry_with_backoff


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multimodal_rag.retry.time.sleep", lambda seconds: None)


def test_succeeds_on_first_try_without_retrying() -> None:
    calls = {"count": 0}

    def operation() -> str:
        calls["count"] += 1
        return "ok"

    assert retry_with_backoff(operation, max_retries=3) == "ok"
    assert calls["count"] == 1


def test_retries_and_eventually_succeeds() -> None:
    calls = {"count": 0}

    def operation() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("transient failure")
        return "ok"

    assert retry_with_backoff(operation, max_retries=5) == "ok"
    assert calls["count"] == 3


def test_raises_last_exception_after_exhausting_retries() -> None:
    calls = {"count": 0}

    def operation() -> str:
        calls["count"] += 1
        raise RuntimeError(f"failure {calls['count']}")

    with pytest.raises(RuntimeError, match="failure 3"):
        retry_with_backoff(operation, max_retries=3)
    assert calls["count"] == 3


def test_backoff_sleeps_between_attempts_not_after_the_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("multimodal_rag.retry.time.sleep", sleeps.append)

    def always_fails() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        retry_with_backoff(always_fails, max_retries=3, backoff_seconds=1.0)

    assert sleeps == [1.0, 2.0]  # 2 sleeps between 3 attempts, exponential
