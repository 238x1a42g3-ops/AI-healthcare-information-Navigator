# AI Healthcare Information Navigator

A local React chat interface for asking questions about a healthcare PDF. The FastAPI service indexes the uploaded document with the existing TF-IDF and ChromaDB pipeline and uses Groq to answer questions from retrieved passages.

## Setup

1. Create a `.env` file in the project root with `GROQ_API_KEY=your_api_key_here`.
2. Create a Python virtual environment and install the dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r .\data\requirements.txt
   ```

3. Install the frontend dependencies:

   ```powershell
   cd .\frontend
   npm install
   ```

## Run locally

Start the API from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

In another terminal, start the React app:

```powershell
cd .\frontend
npm run dev
```

Open the local URL printed by Vite. Upload a text-based PDF (up to 25 MB) and ask questions grounded in its contents. The API is available at `http://127.0.0.1:8000`; interactive API docs are at `/docs`.

This tool provides educational information only and is not a substitute for care from a qualified healthcare professional.
