"""Embedding service boundary.

The rest of the backend uses the `Embedder` and `Tokenizer` interfaces and
the two factories below.
"""
from .base import Embedder, Tokenizer
from .factory import get_embedder, get_tokenizer

__all__ = ["Embedder", "Tokenizer", "get_embedder", "get_tokenizer"]
