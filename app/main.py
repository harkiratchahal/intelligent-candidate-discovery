from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db.session import create_db_and_tables
from app.api.routes.jobs import router as jobs_router
from app.api.routes.candidates import router as candidates_router
from app.api.routes.pipeline import router as pipeline_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(
    title="Intelligent Candidate Discovery",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(jobs_router, prefix="/jobs", tags=["Jobs"])
app.include_router(candidates_router, prefix="/candidates", tags=["Candidates"])
app.include_router(pipeline_router, prefix="/pipeline", tags=["Pipeline"])

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}