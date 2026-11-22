from functools import cache

import config

from .azure import AzureEmbedder
from .base import Embedder, Tokenizer
from .tokenizer import OpenAITokenizer


def get_embedder(settings: config.Settings | None = None) -> Embedder:
    return AzureEmbedder(settings or config.SETTINGS)


@cache
def get_tokenizer() -> Tokenizer:
    return OpenAITokenizer()
