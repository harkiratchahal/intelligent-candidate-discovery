# Intelligent Candidate Discovery - Backend

This is the backend service for the Intelligent Candidate Discovery platform. It provides a robust, AI-powered pipeline to search, filter, score, and explain candidate matches for specific job descriptions.

## 🚀 Tech Stack

- **Framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **Database:** SQLAlchemy (async), Alembic, PostgreSQL (asyncpg) / SQLite
- **AI & ML:** Google Gemini (Generative AI + Embeddings), Faiss (Vector Search), Rank-BM25 / BM25s (Keyword Search), LightGBM, Scikit-learn
- **Dependency Management:** [uv](https://github.com/astral-sh/uv)

## 🏗️ Project Structure

```text
backend/
├── app/
│   ├── api/         # FastAPI routes and endpoints (jobs, candidates, pipeline)
│   ├── db/          # Database connection, sessions, and migrations
│   ├── engine/      # Core candidate discovery pipeline (retrieval, ranking, explain)
│   ├── models/      # SQLAlchemy ORM models and Pydantic schemas
│   ├── services/    # Business logic and external service integrations
│   ├── config.py    # Environment variables and application settings
│   └── main.py      # FastAPI application entry point
├── pyproject.toml   # Project metadata and dependencies
└── README.md        # Project documentation
```

## 🧠 Discovery Pipeline (`app/engine`)

The candidate discovery process is broken down into a multi-stage pipeline:

1. **Retrieval (`retrieval.py`):** Combines semantic search (Faiss + Gemini Embeddings) with keyword search (BM25) using Reciprocal Rank Fusion (RRF) to retrieve a broad set of candidates.
2. **Filtering (`filters.py`):** Applies hard constraints (e.g., location, years of experience, specific skills) to narrow down the retrieved candidates.
3. **Feature Extraction (`features.py`):** Computes advanced features for the remaining candidates, such as semantic similarity to the job description and skill overlap.
4. **Ranking (`ranker.py`):** Uses a machine learning model (LightGBM) to score and rank candidates based on the extracted features.
5. **Reranking & Explanation (`rerank_explain.py`):** Leverages an LLM (Gemini) to provide human-readable explanations for the top candidates' scores and suitability.

## ⚙️ Setup & Installation

1. **Install `uv` (if not already installed):**
   Follow the [uv installation guide](https://github.com/astral-sh/uv#installation).

2. **Install dependencies:**
   Navigate to the `backend` directory and run:
   ```bash
   uv sync
   ```
   Or use standard pip installation if preferred.

3. **Environment Configuration:**
   Create a `.env` file in the root of the `backend` directory and configure the necessary environment variables. Refer to `app/config.py` for all available settings:
   ```env
   # .env example
   DATABASE_URL=sqlite+aiosqlite:///./test.db # or your postgres URL
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## 🏃‍♂️ Running the Server

Start the development server:

```bash
uv run uvicorn app.main:app --reload
```

Alternatively, if you've activated the virtual environment manually:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.
- Interactive API Documentation (Swagger UI): `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🛠️ Development Tools

- **Formatting & Linting:** The project uses `ruff` for fast linting and formatting.
- **Testing:** `pytest` and `pytest-asyncio` are configured for running tests.
