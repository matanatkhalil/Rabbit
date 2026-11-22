# Orange Reader — backend (ingestion)

PDF → text → chunks → embeddings → Postgres/pgvector, an HTTP API for the frontend, and
questions answered from a book (top-k chunks + one LLM call). Each ingestion step is one
replaceable function in `ingest/`.

```
config.py            settings from ../config.yaml, overridden by RABBIT_* env vars; secrets from ../.env
db.py                connections, schema apply, test/stress DB creation
schema.sql           books + chunks tables (idempotent)
embedder/            Azure embedding client and tokenizer for chunking
ingest/
  extract.py         extract_pages(pdf_path)
  clean.py           clean_text(pages)            -> .needs_ocr flag
  chunk.py           chunk_pages(pages, size, overlap)
  embed.py           embed_chunks(chunks, batch_size)
  save.py            save_chunks(book_id, chunks, vectors, ...)
  pipeline.py        ingest_book(book_id, pdf_path, on_progress)
  library.py         add_book / reindex_book / delete_book / list_books
  __main__.py        CLI
api/
  app.py             FastAPI app: /api/* (frontend LibraryApi) + the built UI at /
  jobs.py            background (re)indexing worker, in-memory progress
  events.py          server-sent events: progress, library changes
  schemas.py         DB row -> frontend Book
  __main__.py        python -m api (host/port from config.yaml)
chat/
  retrieve.py        search_chunks: embed the question, ORDER BY cosine distance, LIMIT k
  answer.py          answer_question: shared system prompt + this question + its chunks, one LLM call
tests/               pytest (unit + integration)
stress/              synthetic PDF generator + stress runner
```

## Run

The whole app starts with one command from the project root (`Rabbit/Rabbit/`):

```bash
./start.sh        # database, schema, deps, UI build, then http://127.0.0.1:8000
```

In the project root, copy `.env.example` to `.env`. Set `AZURE_EMBEDDING_API_KEY` for book indexing and question search, and `AZURE_CHAT_API_KEY` for chat replies. Chat uses `DeepSeek-V4.1-Flash` at the Azure endpoint set in `.env.example`.

## Questions

Each question is answered on its own: the question is embedded (`embed_queries`), the
`chat.top_k` closest chunks of that book are taken by cosine distance, and one request
goes to the LLM with the shared system prompt, those chunks and the question. There is
no re-ranking, query rewriting or routing, and no history: nothing from earlier questions
is sent or stored.

## CLI (maintenance)

Works on the same database and data folder as the app (run from this directory):

```bash
../../.venv/bin/python -m ingest add path/to/book.pdf --title "Title" [--author "Name"]
../../.venv/bin/python -m ingest reindex <book_id>
../../.venv/bin/python -m ingest list
../../.venv/bin/python -m ingest delete <book_id>      # chunks cascade
```

## Embedder boundary

Book chunks and search questions are sent to the Azure `/openai/v1/embeddings` endpoint configured
in `config.yaml`. Both use `text-embedding-3-large` (3,072 dimensions). Chunking uses
`tiktoken`; no embedding model weights are downloaded or run locally.

On startup, books indexed with an earlier embedding model are marked for reindexing.
Use **Re-index** in the library or run `python -m ingest reindex <book_id>` from `backend/`
to replace the old vectors using the existing PDF.

## Tests

```bash
python -m pytest                     # all (integration uses DB orange_rabbit_test)
python -m pytest -m "not integration"
```

Tests use stub chat replies and embeddings instead of calling Azure.

## Stress tests

```bash
python -m stress.stress_ingest                         # all scenarios; uses the Azure embedding API
python -m stress.stress_ingest throughput --books 5 --pages 300
python -m stress.stress_ingest concurrent --workers 8
python -m stress.make_pdf book out.pdf --pages 250     # just make a synthetic PDF
```

Scenarios: `throughput`, `reindex` (idempotence), `concurrent` (parallel adds, parallel
reindex of one book, duplicate-upload race), `failures` (scanned/encrypted/corrupt/
duplicate/missing source), `large` (1200 pages), then `cascade` (delete everything).
Uses its own DB `orange_rabbit_stress` and `stress/out/`; JSON report in `stress/out/`.
