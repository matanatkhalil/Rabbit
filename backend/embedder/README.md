# Embeddings

`get_embedder()` sends book chunks and search questions to the Azure embeddings
endpoint configured under `embedder.url` in `config.yaml`. Requests use
`AZURE_EMBEDDING_API_KEY` as the bearer token and `text-embedding-3-large` as the
model. Results are 3,072-dimensional float32 vectors in input order.

`get_tokenizer()` supplies token counts and character offsets for chunking via
`tiktoken`'s `cl100k_base` encoding. It does not run an embedding model.
