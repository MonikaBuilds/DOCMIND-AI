# Deployment Guide

This project is designed as a split deployment:

- Backend: FastAPI on Render, Railway, or another Python web host
- Frontend: React/Vite on Vercel

## Backend On Render

The repository includes `render.yaml` at the project root.

Steps:

1. Push the repository to GitHub.
2. Open Render.
3. Choose **New Blueprint**.
4. Select the GitHub repository.
5. Render will detect `render.yaml`.
6. Deploy the `docmind-ai-backend` service.

Default production-safe environment:

```env
EMBEDDING_PROVIDER=hash
LLM_PROVIDER=extractive
VECTOR_STORE_PROVIDER=memory
```

This mode avoids API quota problems and native Chroma build issues. For stronger quality, switch providers later:

```env
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
```

Backend health check:

```text
https://your-backend-url.onrender.com/health
```

## Frontend On Vercel

Use `frontend/react_app` as the Vercel project root.

Vercel settings:

```text
Framework: Vite
Build command: npm run build
Output directory: dist
Install command: npm install
Root directory: frontend/react_app
```

Set this Vercel environment variable:

```env
VITE_API_BASE_URL=https://your-backend-url.onrender.com/api/v1
```

Then deploy.

## Important Production Notes

- Render free services may sleep after inactivity.
- In-memory storage resets when the backend restarts.
- Use a persistent database and durable vector store for real production.
- Do not deploy with real API keys in Git.
- Configure CORS before public production use if the frontend/backend domains differ.

## Future Production Upgrade

Recommended production path:

```text
Frontend: Vercel
Backend: Render/Railway/Fly.io
Uploads: S3/R2/Cloudinary
Metadata DB: PostgreSQL
Vector DB: Qdrant/Pinecone/Chroma Cloud
LLM: OpenAI/Gemini/remote GPU endpoint
Monitoring: structured logs + error tracking
```
