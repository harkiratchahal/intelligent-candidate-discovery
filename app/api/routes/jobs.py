import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from app.api.deps import get_session
from app.db.models import Jobs
from app.models.job import JobCreate, JobRead

router = APIRouter()


@router.post("", response_model=JobRead)
def create_job(job: JobCreate, session: Session = Depends(get_session)):
    db_job = Jobs(
        title=job.title,
        description=job.description,
        required_skills=job.required_skills,
        min_experience=job.min_experience,
        required_certs=job.required_certs,
    )
    session.add(db_job)
    session.commit()
    session.refresh(db_job)
    return db_job


@router.get("/{job_id}", response_model=JobRead)
def get_job(job_id: uuid.UUID, session: Session = Depends(get_session)):
    job = session.get(Jobs, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job