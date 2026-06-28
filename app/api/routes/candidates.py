import uuid
import pandas as pd
from io import BytesIO
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlmodel import Session
from app.api.deps import get_session
from app.db.models import Candidates
from app.models.candidate import CandidateCreate, CandidateRead

router = APIRouter()


def parse_list_field(value) -> list[str]:
    """
    Parses a comma-separated string into a list of strings.
    Handles None, empty string, and already-list values.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v]
    if isinstance(value, str) and value.strip():
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def parse_date_field(value) -> datetime | None:
    """
    Parses a date string into a timezone-aware datetime.
    Returns None if value is missing or unparseable.
    """
    if value is None or (isinstance(value, float)):
        return None
    try:
        dt = pd.to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.to_pydatetime()
    except Exception:
        return None


def parse_float_field(value, default: float = 0.0) -> float:
    """
    Safely parses a float field, returns default if unparseable.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def row_to_candidate_create(row: dict) -> CandidateCreate:
    """
    Converts one CSV/Excel row (as dict) into a CandidateCreate object.
    Everything that isn't a known field goes into raw_data.
    """
    known_fields = {
        "name", "profile_text", "skills",
        "years_experience", "certifications",
        "last_active", "profile_complete",
    }

    raw_data = {
        k: str(v) for k, v in row.items()
        if k not in known_fields and v is not None
    }

    return CandidateCreate(
        name=str(row.get("name", "")).strip(),
        profile_text=str(row.get("profile_text", "")).strip(),
        skills=parse_list_field(row.get("skills")),
        years_experience=parse_float_field(row.get("years_experience"), default=0.0),
        certifications=parse_list_field(row.get("certifications")),
        last_active=parse_date_field(row.get("last_active")),
        profile_complete=parse_float_field(row.get("profile_complete"), default=0.0),
        raw_data=raw_data if raw_data else None,
    )


@router.post("/upload", response_model=list[CandidateRead])
async def upload_candidates(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    # ── Validate file type ───────────────────────────────────────────────────
    filename = file.filename or ""
    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        raise HTTPException(
            status_code=400,
            detail="Only .csv and .xlsx files are supported"
        )

    contents = await file.read()
    buffer = BytesIO(contents)

    # ── Parse file into DataFrame ────────────────────────────────────────────
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(buffer)
        else:
            df = pd.read_excel(buffer)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse file: {str(e)}"
        )

    # ── Validate required columns ────────────────────────────────────────────
    required_columns = {"name", "profile_text", "years_experience", "profile_complete"}
    missing = required_columns - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}"
        )

    # ── Parse rows and save to DB ────────────────────────────────────────────
    saved: list[CandidateRead] = []
    errors: list[str] = []

    for i, row in enumerate(df.to_dict(orient="records")):
        try:
            candidate_data = row_to_candidate_create(row)

            db_candidate = Candidates(
                name=candidate_data.name,
                profile_text=candidate_data.profile_text,
                skills=candidate_data.skills,
                years_experience=candidate_data.years_experience,
                certifications=candidate_data.certifications,
                last_active=candidate_data.last_active,
                profile_complete=candidate_data.profile_complete,
                raw_data=candidate_data.raw_data,
            )
            session.add(db_candidate)
            session.commit()
            session.refresh(db_candidate)
            saved.append(CandidateRead.model_validate(db_candidate))

        except Exception as e:
            errors.append(f"Row {i + 1}: {str(e)}")
            session.rollback()
            continue

    if not saved:
        raise HTTPException(
            status_code=400,
            detail=f"No candidates saved. Errors: {errors}"
        )

    return saved


@router.get("/{candidate_id}", response_model=CandidateRead)
def get_candidate(candidate_id: uuid.UUID, session: Session = Depends(get_session)):
    candidate = session.get(Candidates, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate