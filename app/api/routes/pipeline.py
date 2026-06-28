import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from app.api.deps import get_session
from app.models.result import PipelineRunRead
from app.engine import pipeline
from pydantic import BaseModel

router = APIRouter()


class PipelineRunRequest(BaseModel):
    job_id: uuid.UUID


@router.post("/run", response_model=PipelineRunRead)
def run_pipeline(
    request: PipelineRunRequest,
    session: Session = Depends(get_session),
):
    try:
        result = pipeline.run(
            job_id=request.job_id,
            session=session,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}", response_model=PipelineRunRead)
def get_pipeline_run(
    run_id: uuid.UUID,
    session: Session = Depends(get_session),
):
    try:
        return pipeline.get_run(run_id, session)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))