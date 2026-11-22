from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Tokenizer(Protocol):
    def count(self, text: str) -> int:
        """Number of model tokens in `text` (no special tokens)."""

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span of every token in `text`, in order."""


@runtime_checkable
class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Document embeddings as float32 vectors of shape (len(texts), dim)."""

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Query embeddings in the same vector space as the documents."""
