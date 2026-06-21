# React Production Frontend

Professional ChatGPT-style frontend for DocMind AI.

Run after starting the FastAPI backend:

```powershell
cd backend
python -m uvicorn app.main:app --reload
```

Then start React:

```powershell
cd frontend\react_app
npm.cmd install
npm.cmd run dev
```

The app expects the backend at `http://127.0.0.1:8000` through the Vite `/api` proxy. For a deployed backend, set:

```env
VITE_API_BASE_URL=https://your-api-domain.com/api/v1
```
