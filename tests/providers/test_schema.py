import pytest

from multimodal_rag.providers.schema import EmbeddingVector, assert_single_model


def _vector(model_id: str) -> EmbeddingVector:
    return EmbeddingVector(vector=[0.1, 0.2], model_id=model_id, dimension=2)


def test_single_model_passes_silently() -> None:
    assert_single_model([_vector("model-a"), _vector("model-a"), _vector("model-a")])


def test_empty_list_passes_silently() -> None:
    assert_single_model([])


def test_mixed_models_raises() -> None:
    with pytest.raises(ValueError, match="model-a.*model-b|model-b.*model-a"):
        assert_single_model([_vector("model-a"), _vector("model-b")])
