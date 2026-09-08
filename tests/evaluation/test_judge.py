import pytest

from multimodal_rag.evaluation.judge import JudgeParseError, score_faithfulness, score_relevance
from multimodal_rag.providers.base import LLMProvider


class _FakeLLM(LLMProvider):
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] | None = None

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.last_messages = messages
        return self._response


def test_score_faithfulness_parses_a_well_formed_json_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = _FakeLLM('{"score": 0.8, "reasoning": "mostly supported"}')
    monkeypatch.setattr("multimodal_rag.evaluation.judge.get_llm", lambda: fake_llm)

    score = score_faithfulness("The answer.", "The context.")

    assert score == 0.8
    assert fake_llm.last_messages is not None
    assert "The answer." in fake_llm.last_messages[0]["content"]
    assert "The context." in fake_llm.last_messages[0]["content"]


def test_score_faithfulness_does_not_pass_the_question_to_the_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Faithfulness is about (answer, context) only -- the question isn't a
    # parameter of score_faithfulness at all, by design (see judge.py's
    # module docstring for why).
    fake_llm = _FakeLLM('{"score": 1.0, "reasoning": "fine"}')
    monkeypatch.setattr("multimodal_rag.evaluation.judge.get_llm", lambda: fake_llm)

    score_faithfulness("This mentions P99 latency.", "The context has no P99 data.")

    assert fake_llm.last_messages is not None
    content = fake_llm.last_messages[0]["content"]
    assert "## QUESTION" not in content


def test_score_relevance_parses_a_well_formed_json_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = _FakeLLM('{"score": 0.3, "reasoning": "off-topic"}')
    monkeypatch.setattr("multimodal_rag.evaluation.judge.get_llm", lambda: fake_llm)

    score = score_relevance("What was the P95 latency?", "The sky is blue.")

    assert score == 0.3


def test_score_extracts_json_even_when_wrapped_in_a_markdown_code_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = _FakeLLM('```json\n{"score": 0.6, "reasoning": "partial"}\n```')
    monkeypatch.setattr("multimodal_rag.evaluation.judge.get_llm", lambda: fake_llm)

    assert score_faithfulness("a", "b") == 0.6


def test_score_extracts_json_even_with_leading_and_trailing_prose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = _FakeLLM('Sure, here is my score: {"score": 0.9} -- hope that helps!')
    monkeypatch.setattr("multimodal_rag.evaluation.judge.get_llm", lambda: fake_llm)

    assert score_faithfulness("a", "b") == 0.9


def test_score_is_clamped_to_the_zero_to_one_range(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = _FakeLLM('{"score": 1.7}')
    monkeypatch.setattr("multimodal_rag.evaluation.judge.get_llm", lambda: fake_llm)

    assert score_faithfulness("a", "b") == 1.0


def test_raises_judge_parse_error_when_no_json_object_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = _FakeLLM("I refuse to answer in JSON.")
    monkeypatch.setattr("multimodal_rag.evaluation.judge.get_llm", lambda: fake_llm)

    with pytest.raises(JudgeParseError):
        score_faithfulness("a", "b")


def test_raises_judge_parse_error_for_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = _FakeLLM('{"score": 0.5,,,}')
    monkeypatch.setattr("multimodal_rag.evaluation.judge.get_llm", lambda: fake_llm)

    with pytest.raises(JudgeParseError):
        score_faithfulness("a", "b")


def test_raises_judge_parse_error_when_score_field_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = _FakeLLM('{"reasoning": "no score here"}')
    monkeypatch.setattr("multimodal_rag.evaluation.judge.get_llm", lambda: fake_llm)

    with pytest.raises(JudgeParseError):
        score_faithfulness("a", "b")


def test_raises_judge_parse_error_when_score_is_not_numeric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = _FakeLLM('{"score": "high"}')
    monkeypatch.setattr("multimodal_rag.evaluation.judge.get_llm", lambda: fake_llm)

    with pytest.raises(JudgeParseError):
        score_faithfulness("a", "b")


def test_provider_errors_propagate_instead_of_being_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingLLM(LLMProvider):
        def generate(self, messages: list[dict[str, str]]) -> str:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr("multimodal_rag.evaluation.judge.get_llm", lambda: _FailingLLM())

    with pytest.raises(RuntimeError):
        score_faithfulness("a", "b")
