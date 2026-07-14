import pytest

from multimodal_rag.providers.base import EmbeddingProvider, LLMProvider, Reranker, VisionProvider


@pytest.mark.parametrize("interface", [LLMProvider, EmbeddingProvider, VisionProvider, Reranker])
def test_interfaces_cannot_be_instantiated_directly(interface: type) -> None:
    with pytest.raises(TypeError):
        interface()
