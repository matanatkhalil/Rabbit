import os
import re
from zlib import crc32

import numpy as np
import pytest

from embedder import get_tokenizer

TEST_DB_URL = os.environ.get(
    "RABBIT_TEST_DATABASE_URL", "postgresql://rabbit:rabbit@localhost:5432/orange_rabbit_test"
)


@pytest.fixture(scope="session")
def tok():
    return get_tokenizer()


@pytest.fixture(scope="module", autouse=True)
def fake_embeddings(request):
    """DB integration tests use deterministic word vectors instead of calling Azure."""
    if request.node.get_closest_marker("integration") is None:
        yield
        return

    from embedder.azure import AzureEmbedder

    def embed(self, texts):
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for v, text in zip(vectors, texts):
            for word in set(re.findall(r"\w+", text.lower())):
                v[crc32(word.encode()) % self.dim] = 1.0
            if not v.any():
                v[0] = 1.0
            v /= np.linalg.norm(v)
        return vectors

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(AzureEmbedder, "_embed", embed)
        yield


@pytest.fixture(scope="session")
def db_url():
    import psycopg

    import db

    try:
        db.ensure_database(TEST_DB_URL)
    except psycopg.OperationalError as e:
        pytest.skip(f"test database unavailable: {e}")
    return TEST_DB_URL


class RealLLMCall(BaseException):
    """BaseException so answer_question's error handling can't turn it into a normal failure."""


@pytest.fixture(autouse=True)
def no_real_llm_calls(monkeypatch):
    """Tests never reach the real LLM (its API has a tight request budget).
    A test that needs an answer replaces chat.answer._post_completion with a stub."""
    import chat.answer

    def refuse(messages, settings):
        raise RealLLMCall("a test tried to call the real LLM")

    monkeypatch.setattr(chat.answer, "_post_completion", refuse)
