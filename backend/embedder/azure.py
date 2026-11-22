import httpx
import numpy as np

import config


class AzureEmbedder:
    """Send raw texts to the Azure embeddings endpoint in input order."""

    model_name = config.EMBED_MODEL
    dim = config.EMBED_DIM

    def __init__(self, settings: config.Settings):
        self._settings = settings

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        settings = self._settings
        if not settings.embed_api_key:
            raise RuntimeError("Set AZURE_EMBEDDING_API_KEY in .env to enable embeddings.")
        response = httpx.post(
            settings.embed_url,
            headers={"Authorization": f"Bearer {settings.embed_api_key}"},
            json={"input": texts, "model": self.model_name},
            timeout=settings.embed_timeout_s,
        )
        response.raise_for_status()

        data = sorted(response.json()["data"], key=lambda item: item["index"])
        if [item["index"] for item in data] != list(range(len(texts))):
            raise ValueError("Azure did not return exactly one embedding per input.")
        vectors = np.asarray([item["embedding"] for item in data], dtype=np.float32)
        if vectors.shape != (len(texts), self.dim) or not np.isfinite(vectors).all():
            raise ValueError(f"Azure returned invalid embeddings; expected {self.dim} dimensions per input.")
        return vectors
