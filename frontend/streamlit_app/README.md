# Streamlit Validation UI

This is the Phase 1 frontend path for validating the DocMind AI pipeline before the production React app.

Run from the project root after installing backend requirements:

```powershell
python -m streamlit run frontend\streamlit_app\app.py
```

The UI provides:

- PDF upload and processing
- parsing, cleaning, and chunking inspection
- ChatGPT-style local chat layout
- mock answer generation
- citation display
- provider reminder for `.env` keys

This UI is for validation and learning. The production UI will be built in React later.
