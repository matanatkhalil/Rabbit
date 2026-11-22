import json

import httpx
import numpy as np
import pytest

import config
from embedder import get_embedder


@pytest.mark.parametrize("method", ["embed_documents", "embed_queries"])
def test_azure_request_and_response_order(monkeypatch, method):
    settings = config.load(env={"AZURE_EMBEDDING_API_KEY": "test-embedding-key", "AZURE_CHAT_API_KEY": "other-key"})
    emb = get_embedder(settings)
    texts = ["first phrase", "second phrase"]

    def respond(request):
        assert request.method == "POST"
        assert str(request.url) == "https://agro-llm.services.ai.azure.com/openai/v1/embeddings"
        assert request.headers["authorization"] == "Bearer test-embedding-key"
        assert request.headers["content-type"] == "application/json"
        assert json.loads(request.content) == {"input": texts, "model": "text-embedding-3-large"}
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [0.0, 1.0] + [0.0] * 3070},
            {"index": 0, "embedding": [1.0, 0.0] + [0.0] * 3070},
        ]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(httpx, "post", client.post)
        vectors = getattr(emb, method)(texts)
    assert vectors.shape == (2, 3072)
    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors[:, :2], np.eye(2))


def test_no_key_and_empty_batches_do_not_call_azure(monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("Azure should not be called")

    monkeypatch.setattr(httpx, "post", unexpected_call)
    emb = get_embedder(config.load(env={"AZURE_CHAT_API_KEY": "chat-only"}))
    assert emb.embed_documents([]).shape == (0, 3072)
    with pytest.raises(RuntimeError, match="AZURE_EMBEDDING_API_KEY"):
        emb.embed_queries(["question"])
