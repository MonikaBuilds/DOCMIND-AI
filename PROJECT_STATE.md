# DocMind AI Project State

## Completed Modules

- Phase 1 planning refined for AI/ML, GenAI, RAG, and interview preparation.
- Phase 2 architecture refined with provider abstractions, OCR fallback strategy, and RAG evaluation.
- Git hygiene defined for local-only learning docs, uploads, vector DB files, OCR outputs, logs, caches, and secrets.
- Phase 3 project structure created for backend, frontend validation UI, frontend production UI, private learning docs, and public project notes.
- Phase 4 configuration foundation created with `.env.example` and typed settings.
- Provider abstraction placeholders created for embeddings, LLMs, OCR, and vector storage.
- RAG evaluation placeholders created in the API, domain, and service layers.
- Phase 5 PDF upload pipeline implemented with validation, local storage, response schemas, and service tests.
- Phase 6 PDF parsing implemented with PyMuPDF, metadata extraction, page-level text, and OCR readiness signals.
- Phase 7 OCR decision flow implemented with provider abstraction, page rendering adapter, and fake-provider tests.
- Phase 8 document preprocessing implemented with deterministic cleanup and metadata preservation.
- Phase 9 page-aware chunking implemented with configurable word windows, overlap, stable chunk IDs, and citation metadata.
- Phase 10 metadata management implemented with vector metadata serialization, citation labels, heading propagation, and document summaries.
- Phase 11 embedding generation implemented with provider abstraction, BGE provider, query/passage prefixes, and fake-provider service tests.
- Phase 12 vector database integration implemented with ChromaDB adapter, vector service, and fake-store tests.
- Phase 13 semantic retrieval implemented with query embedding, vector search orchestration, top-k handling, and score filtering.
- Phase 14 hybrid retrieval implemented with BM25-style keyword scoring, weighted fusion, and duplicate merging.
- Phase 15 re-ranking implemented with provider interface and dependency-free heuristic re-ranker.
- Phase 16 conversation memory implemented with bounded per-session history and prompt-formatting support.
- Phase 17 prompt engineering implemented with grounded prompt builder, source formatting, citation instructions, and refusal rules.
- Phase 18 LLM integration implemented with provider abstraction, mock provider, Gemini skeleton, OpenAI skeleton, and LLM service tests.
- Phase 19 answer generation implemented with prompt/LLM orchestration and structured cited answers.
- Phase 20 citation generation implemented with structured citations, deduplication, labels, and chunk traceability.
- Phase 21 Streamlit validation UI implemented with professional ChatGPT-style layout, PDF processing, mock chat, citations, and pipeline inspection.
- Phase 22 FastAPI production API completed with upload registration, document processing, search, chat, runtime store, schemas, and API flow tests.
- Phase 23 React production frontend implemented with Vite, typed API client, ChatGPT-style layout, upload/process/search/chat flows, and citations.
- React production frontend corrected with a polished Claude-style chat workspace, professional empty state, launch-facing non-technical copy, viewport-locked single-screen layout, sources drawer, mobile document drawer, and improved overflow handling.
- Phase 24 real RAG pipeline activated with provider factories, OpenAI embedding support, vector indexing, semantic search API wiring, and RAG orchestration tests.
- Added Gemini embedding provider and dependency-free in-memory vector store fallback for local Windows/demo runs where Chroma native dependencies are unavailable.
- Added remote HTTP embedding and LLM providers for Kaggle/Cloudflare-tunnel GPU inference or future RunPod/Modal/Hugging Face endpoints.
- Added Kaggle-ready remote GPU model server notebook with FastAPI `/embed`, `/generate`, `/health`, token auth, model loading, local tests, and Cloudflare tunnel instructions.
- Added quota-free local fallback providers: `HashEmbeddingProvider` for dependency-free lexical embeddings and `ExtractiveLLMProvider` for grounded source-text answers without OpenAI/Gemini calls.
- Phase 25 RAG evaluation implemented with Recall@K, Precision@K, hit rate, citation accuracy, groundedness scaffolding, sample dataset, and tests.

## Current Module

- Phase 26: Deployment.

## Pending Modules

1. Deployment.
2. Monitoring and logging.
3. Future improvements.

## Known Issues

- The visible `.git` directory is not currently recognized by `git status`; Git initialization may need to be repaired or recreated.
- FastAPI is not installed in the base environment, so API runtime checks need the backend requirements installed.
- Upload, parsing, OCR decision, preprocessing, chunking, metadata, embedding, vector, semantic retrieval, hybrid retrieval, re-ranking, memory, prompt, LLM, answer, citation, Streamlit validation UI, and FastAPI API flow are implemented and tested where possible.
- API process/search/chat now use the configured RAG provider path instead of the earlier keyword/mock-only path.
- Conversation memory is process-local and not persistent.
- Concrete EasyOCR/PaddleOCR provider implementation is still pending.
- ChromaDB is required before using the real `ChromaVectorStore`.
- Local `.env` is configured to use `EMBEDDING_PROVIDER=gemini` and `VECTOR_STORE_PROVIDER=memory` for a smoother demo path without Chroma build tools.
- Local `.env` is currently configured to use `EMBEDDING_PROVIDER=hash`, `LLM_PROVIDER=extractive`, and `VECTOR_STORE_PROVIDER=memory` to avoid OpenAI quota and unavailable Gemini embedding-model issues.
- Remote provider support exists, but Kaggle/Cloudflare URLs are temporary and should be treated as demo infrastructure, not production hosting.
- Kaggle notebook lives in ignored `docs_learning/` because it contains demo instructions and may be customized with private tokens or tunnel URLs.
- `sentence-transformers` is required before using the real BGE provider.
- `google-generativeai` and `GEMINI_API_KEY` are required before using the real Gemini provider.
- `openai` and `OPENAI_API_KEY` are required before using the real OpenAI provider.
- `OPENAI_API_KEY` can also be used for OpenAI embeddings when `EMBEDDING_PROVIDER=openai`.
- RAG groundedness/faithfulness metrics currently use deterministic lexical support, not LLM-as-judge.
- Hybrid keyword scoring currently works over supplied in-memory chunks.
- Re-ranking uses a heuristic; cross-encoder re-ranking is pending.
- Citations are page-level, not bounding-box-level.
- Streamlit validation chat uses mock LLM output until a provider is configured.
- FastAPI chat currently uses mock LLM output until provider factories are switched to real OpenAI/Gemini.
- React frontend depends on the FastAPI backend running locally at `http://127.0.0.1:8000`.
- React frontend dev server verified over HTTP at `http://127.0.0.1:5173`; browser automation CLI is not available in this environment for screenshot verification.
- `__pycache__` files were generated by syntax verification and are ignored by Git.
- PyMuPDF tests require access to the user site-packages install; in this environment they pass with elevated test execution and `--basetemp .test_tmp`.

## Design Decisions

- Use FastAPI as the API-first backend.
- Use Streamlit before React to validate the AI pipeline quickly.
- Keep React as the production frontend after the API and RAG behavior stabilize.
- Use provider abstractions for embeddings and LLMs.
- Use `BAAI/bge-small-en-v1.5` as the default embedding model.
- Support Gemini and OpenAI behind the LLM provider interface.
- Use ChromaDB first but hide it behind a vector store interface.
- Preserve page-aware metadata for citations.
- Add a dedicated RAG evaluation phase before deployment.
- Keep upload separate from parsing so ingestion, validation, and document processing remain testable independently.
- Use PyMuPDF before OCR because direct text extraction is faster and more accurate for digital PDFs.
- Mark weakly extracted pages with `needs_ocr` instead of running OCR inside the parser.
- Keep OCR provider logic separate from page rendering and parsing.
- Keep preprocessing deterministic and conservative before chunking.
- Chunk within page boundaries to keep citations accurate.
- Use word-based chunks with overlap as a clear baseline before semantic chunking.
- Generate citations from metadata instead of asking the LLM to invent references.
- Keep vector metadata primitive and portable across ChromaDB, Qdrant, and Pinecone.
- Load model/provider SDKs lazily so unit tests and non-provider modules stay lightweight.
- Keep retrieval separate from generation so retrieval quality can be evaluated independently.
- Use hybrid retrieval to combine semantic recall with exact keyword matching.
- Keep re-ranking as a post-retrieval step with a provider interface for future cross-encoders.
- Keep conversation memory separate from retrieved document context.
- Use strict document-grounded prompts with explicit refusal behavior.
- Keep API keys in ignored `.env` files, not frontend code.
- Keep API routes thin and service-driven with request/response schemas.
- Use runtime store only for local development before database persistence.
- Use React as the production UI after Streamlit validation and API completion.
- Keep the production UI chat-first and demo-ready: no internal RAG/backend terminology in visible copy, focused center workspace, strong empty state, restrained document rail, sources drawer, and mobile drawer navigation.
- Keep all provider keys server-side in `backend/.env`.
- Use provider factories so `.env` controls whether the runtime uses OpenAI, Gemini, BGE, Chroma, or mock providers.
- Use memory vector storage for local demos when Chroma cannot install because of Windows C++ build requirements.
- Use remote HTTP providers for experimental open-source model inference while keeping retrieval and generation logic unchanged.
- Use hash embeddings plus extractive answers as the no-secrets/no-quota fallback path for demos when paid providers or remote GPU tunnels are unavailable.
- Evaluate retrieval and answer quality before deployment.

## Future Improvements

- Qdrant or Pinecone adapter.
- Anthropic and local LLM adapters.
- OpenAI embedding adapter.
- Advanced table and figure extraction.
- Knowledge graph extraction.
- RAG evaluation dashboard.
- Authentication and document collections.
- Cloud object storage adapter for uploaded PDFs.
- More advanced PDF quality scoring using layout, images, fonts, and scanned-page detection.
- OCR confidence scores and language configuration.
- Layout-aware preprocessing for tables, code, formulas, and figure captions.
- Semantic and heading-aware chunking.
- Rich citation objects with bounding boxes or paragraph IDs.
- Embedding cache keyed by chunk text hash.
- Replace local keyword scorer with `rank-bm25`, SQLite FTS, Elasticsearch, or OpenSearch.
- Cross-encoder re-ranker.
- Persistent conversation memory keyed by authenticated users and sessions.
- Token-budget-aware context packing.
- Streaming LLM responses.

## Next Implementation Target

Implement Phase 26: Deployment.

The next module should add:

- deployment configuration for backend and frontend
- environment variable checklist
- production storage/vector DB notes
- deployment docs
- private learning note for Phase 26
- `PROJECT_STATE.md` update after completion
