import tiktoken


class OpenAITokenizer:
    """Token counts and character offsets for chunking text-embedding-3-large inputs."""

    def __init__(self):
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self._encoding.encode_ordinary(text))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        _, starts = self._encoding.decode_with_offsets(self._encoding.encode_ordinary(text))
        # Several tokens may share a start when their bytes overlap a Unicode character.
        return [(start, max(start + 1, end)) for start, end in zip(starts, starts[1:] + [len(text)])]
