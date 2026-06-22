# DocMind AI

DocMind AI is a production-minded document intelligence system that lets users upload PDFs and ask questions through a conversational assistant. It parses documents, prepares page-aware sections, retrieves relevant evidence, and answers with source references.

The project is designed as an AI/ML and GenAI engineering showcase: it demonstrates retrieval systems, document processing, provider abstraction, clean FastAPI architecture, evaluation thinking, and a polished user workflow.
## 🎥 DocMind AI Demo

**[Watch the Demo Video](https://drive.google.com/file/d/1eU5VhdexjHEJ7Kp49gRj4pNSFLpV4XTd/view?usp=drive_link)**

## Project Vision

Most document chat demos stop at "upload PDF and ask a question." DocMind AI is built to go deeper. It focuses on the complete lifecycle needed for an explainable document assistant:

- reliable file ingestion
- page-level PDF parsing
- text cleaning and chunking
- metadata preservation
- embedding generation
- vector retrieval
- answer generation
- citations and page references
- retrieval evaluation
- provider flexibility for local, cloud, and remote GPU inference

The goal is not only to build a working app, but to make every architectural decision understandable in technical interviews and practical enough for hackathon or industrial demos.

## Core Features

- Upload one or more PDF documents
- Prepare documents into searchable page-aware sections
- Ask natural-language questions
- Retrieve relevant document sections
- Generate grounded answers from retrieved context
- Show document/page references
- Search inside documents
- Track document readiness, pages, and sections
- Use local fallback providers when paid APIs or GPU endpoints are unavailable
- Switch to OpenAI, Gemini, BGE, Chroma, or remote GPU providers through configuration
- Evaluate retrieval and answer quality with RAG-focused metrics

## Current Status

The project currently supports a full local demo path without paid API usage:

```env
EMBEDDING_PROVIDER=hash
LLM_PROVIDER=extractive
VECTOR_STORE_PROVIDER=memory
```

This fallback mode is intentionally included so the system can still run during demos when:

- OpenAI quota is exhausted
- Gemini embedding models are unavailable
- Chroma native dependencies are difficult to install on Windows
- a remote GPU tunnel is not ready

For stronger model quality, the same backend can be switched to OpenAI, Gemini, BGE, ChromaDB, or a remote GPU server without changing the RAG pipeline code.

## System Architecture

```text
React Frontend
  |
  v
FastAPI Backend
  |
  +-- Upload API
  |     -> validates PDFs
  |     -> stores original file
  |     -> registers document metadata
  |
  +-- Document Processing
  |     -> parses PDF with PyMuPDF
  |     -> detects low-quality pages
  |     -> cleans text
  |     -> chunks page-aware sections
  |     -> preserves document/page/source metadata
  |
  +-- Embedding Layer
  |     -> Hash fallback provider
  |     -> BGE provider
  |     -> OpenAI provider
  |     -> Gemini provider
  |     -> Remote HTTP provider
  |
  +-- Vector Store Layer
  |     -> In-memory demo store
  |     -> ChromaDB adapter
  |     -> ready for Qdrant/Pinecone adapters
  |
  +-- Retrieval Layer
  |     -> semantic retrieval
  |     -> hybrid retrieval foundations
  |     -> re-ranking interface
  |
  +-- Generation Layer
        -> prompt builder
        -> extractive fallback provider
        -> OpenAI/Gemini/remote LLM providers
        -> citation generation
```

## Request Lifecycle

```text
PDF upload
  -> file validation
  -> local storage
  -> document registration
  -> PDF parsing
  -> OCR decision point
  -> text preprocessing
  -> page-aware chunking
  -> metadata propagation
  -> embedding generation
  -> vector storage
  -> user question
  -> query embedding
  -> semantic retrieval
  -> re-ranking
  -> context construction
  -> prompt generation
  -> answer generation
  -> citation formatting
  -> final response
```

## Why This Project Is Strong For AI/ML And GenAI

DocMind AI covers the skills expected from an AI/ML or LLM engineer:

- document AI and PDF processing
- NLP preprocessing
- embeddings and vector search
- retrieval-augmented generation
- clean ML system design
- provider abstraction
- API-first backend engineering
- frontend integration
- RAG evaluation
- deployment-aware trade-offs
- production failure handling

It can be explained as:

> A modular document intelligence system that processes PDFs into page-aware searchable sections and uses a configurable RAG pipeline to answer user questions with source references.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| Frontend | React, Vite, TypeScript |
| Validation UI | Streamlit |
| PDF Parsing | PyMuPDF |
| Embeddings | Hash fallback, BGE, OpenAI, Gemini, remote HTTP |
| Vector Storage | In-memory fallback, ChromaDB adapter |
| LLM Providers | Extractive fallback, OpenAI, Gemini, remote HTTP |
| Testing | Pytest |
| Deployment Direction | Render/Railway for backend, Vercel for frontend, GPU endpoint optional |

## Provider Modes

### 1. Local Demo Mode

Best when you want the app to run without paid APIs:

```env
EMBEDDING_PROVIDER=hash
LLM_PROVIDER=extractive
VECTOR_STORE_PROVIDER=memory
```

This mode is deterministic, fast to start, and useful for demos. It is not as semantically strong as a real embedding model or hosted LLM.

### 2. OpenAI Mode

```env
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL_NAME=text-embedding-3-small
LLM_PROVIDER=openai
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=your_key_here
VECTOR_STORE_PROVIDER=memory
```

### 3. Gemini Mode

```env
LLM_PROVIDER=gemini
GEMINI_MODEL_NAME=gemini-1.5-flash
GEMINI_API_KEY=your_key_here
```

### 4. Remote GPU Mode

Use this when hosting open-source models on Kaggle, RunPod, Modal, or another GPU server:

```env
EMBEDDING_PROVIDER=remote
REMOTE_EMBEDDING_URL=https://your-model-server/embed

LLM_PROVIDER=remote
REMOTE_LLM_URL=https://your-model-server/generate

REMOTE_MODEL_API_KEY=your_private_token
REMOTE_REQUEST_TIMEOUT_SECONDS=180
VECTOR_STORE_PROVIDER=memory
```

The private Kaggle notebook for this is kept in `docs_learning/`, which is intentionally ignored by Git.

## Project Structure

```text
DocMind AI/
  backend/
    app/
      api/
        routes/
          upload.py
          documents.py
          chat.py
          search.py
          summary.py
          evaluation.py
      core/
        config.py
        exceptions.py
        providers.py
        runtime_store.py
      domain/
        documents.py
        chunks.py
        retrieval.py
        chat.py
        citations.py
        evaluation.py
      infrastructure/
        embeddings/
        llm/
        vector_store/
        file_storage/
        ocr/
      schemas/
      services/
      tests/
    requirements.txt
    .env.example

  frontend/
    react_app/
      src/
        App.tsx
        api.ts
        styles.css
    streamlit_app/

  PROJECT_STATE.md
  README.md
  .gitignore
```

## Backend Setup

```powershell
cd "C:\All Projects\DocMind AI\backend"
python -m pip install -r requirements.txt
```

Create a private environment file:

```powershell
copy .env.example .env
```

For a no-quota local demo, keep:

```env
EMBEDDING_PROVIDER=hash
LLM_PROVIDER=extractive
VECTOR_STORE_PROVIDER=memory
```

Run the backend:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## Frontend Setup

```powershell
cd "C:\All Projects\DocMind AI\frontend\react_app"
npm.cmd install
npm.cmd run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Main API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Backend health check |
| `POST` | `/api/v1/upload/pdfs` | Upload PDF files |
| `GET` | `/api/v1/documents` | List uploaded documents |
| `POST` | `/api/v1/documents/{document_id}/process` | Prepare a document for search/chat |
| `POST` | `/api/v1/search` | Search prepared documents |
| `POST` | `/api/v1/chat` | Ask document-grounded questions |
| `GET` | `/api/v1/evaluation/status` | Evaluation module status |
| `POST` | `/api/v1/evaluation/retrieval` | Run retrieval evaluation |

## Evaluation Strategy

DocMind AI includes a RAG evaluation layer because production RAG cannot be trusted by intuition alone.

Implemented evaluation concepts:

- Recall@K
- Precision@K
- retrieval hit rate
- citation accuracy
- groundedness scaffolding
- faithfulness scaffolding
- hallucination-risk baseline

This makes the project stronger than a simple chatbot because it shows how retrieval and answer quality can be measured and improved.

## Key Engineering Decisions

### Clean Architecture

Routes stay thin. Business logic lives in services. Provider-specific code lives in infrastructure. Domain objects are kept separate from API schemas.

### Provider Abstraction

Embeddings, vector storage, and LLM calls are behind interfaces. This allows switching between local fallback, OpenAI, Gemini, BGE, Chroma, and remote GPU services without rewriting retrieval or answer generation.

### Page-Aware Chunking

Chunks preserve:

- document ID
- filename
- page number
- source path
- heading
- chunk index

This enables citations and makes answers easier to audit.

### Local Fallback Mode

The local fallback mode is included for reliability during demos. It is not meant to replace stronger embedding and LLM providers, but it prevents the project from failing because of quota, billing, or native dependency issues.

### Private Learning Notes

The `docs_learning/` folder contains personal implementation notes, interview prep material, and local notebooks. It is ignored by Git and should not be committed.

## Git Hygiene And Security

Tracked:

- backend source code
- frontend source code
- tests
- `README.md`
- `PROJECT_STATE.md`
- `.env.example`
- dependency files

Ignored:

- `.env`
- `.env.*`
- uploaded PDFs
- vector databases
- OCR outputs
- parsed documents
- logs
- caches
- `node_modules`
- build outputs
- `docs_learning/`
- local agent/editor metadata

Never commit real API keys, uploaded user documents, local notebooks with tunnel URLs, or generated vector stores.

## Testing

Run focused backend tests:

```powershell
cd "C:\All Projects\DocMind AI"
python -m pytest backend\app\tests
```

Run frontend build:

```powershell
cd "C:\All Projects\DocMind AI\frontend\react_app"
npm.cmd run build
```

## Known Limitations

- In-memory storage resets when the backend restarts.
- Local fallback embeddings are lexical and not as strong as real semantic embeddings.
- Extractive fallback answers are grounded but less fluent than a hosted LLM.
- ChromaDB may require native build support on some Windows environments.
- Remote Kaggle/Cloudflare tunnels are temporary and not production hosting.
- Authentication and persistent user document collections are future work.

## Roadmap

- persistent database for document records
- persistent vector storage with Chroma/Qdrant/Pinecone
- authentication and user workspaces
- cloud object storage for uploaded PDFs
- production OCR provider
- streaming chat responses
- stronger re-ranking with cross-encoders
- improved answer evaluation with LLM-as-judge
- deployment on managed backend and frontend platforms
- GPU inference endpoint through RunPod, Modal, or Hugging Face
