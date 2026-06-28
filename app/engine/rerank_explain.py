import pandas as pd
from app.db.models import Candidates, Jobs
from app.models.result import RankedCandidateResult
from app.services.llm import generate_explanation
from app.engine.ranker import get_top_k
from app.config import settings
import uuid


def run(
    ranked_df: pd.DataFrame,
    candidates: list[Candidates],
    job: Jobs,
    top_k: int = None,
) -> list[RankedCandidateResult]:
    """
    Takes the ranked DataFrame from ranker.py and:
    1. Slices to top_k candidates
    2. For each, calls Gemini to generate matched_skills, gaps, justification
    3. Returns a list of RankedCandidateResult (Pydantic models)

    Candidates outside top_k still get added to results
    but without LLM justification (justification=None)
    """
    if ranked_df.empty:
        return []

    top_k = top_k or settings.EXPLANATION_TOP_K

    # build a lookup map: candidate_id (str) → Candidates ORM object
    candidate_map: dict[str, Candidates] = {
        str(c.id): c for c in candidates
    }

    # slice to top_k for LLM explanation
    top_df = get_top_k(ranked_df, k=top_k)
    rest_df = ranked_df.iloc[top_k:].reset_index(drop=True)

    results: list[RankedCandidateResult] = []

    # ── Top k — with LLM explanation ────────────────────────────────────────
    for _, row in top_df.iterrows():
        candidate_id = row["candidate_id"]
        candidate = candidate_map.get(candidate_id)

        if candidate is None:
            continue

        try:
            explanation = generate_explanation(
                job_title=job.title,
                job_description=job.description or "",
                required_skills=job.required_skills or [],
                candidate_name=candidate.name,
                candidate_profile=candidate.profile_text,
                candidate_skills=candidate.skills or [],
                matched_skills=row["matched_skills"],
                gaps=row["gaps"],
            )
            matched_skills = explanation["matched_skills"]
            gaps = explanation["gaps"]
            justification = explanation["justification"]

        except Exception:
            # if Gemini call fails, fall back to features.py values
            matched_skills = row["matched_skills"]
            gaps = row["gaps"]
            justification = None

        results.append(
            RankedCandidateResult(
                candidate_id=uuid.UUID(candidate_id),
                name=row["name"],
                rank=int(row["rank"]),
                final_score=float(row["final_score"]),
                semantic_sim=float(row["semantic_sim"]),
                skill_overlap=float(row["skill_overlap"]),
                exp_fit=float(row["exp_fit"]),
                activity_score=float(row["activity_score"]),
                matched_skills=matched_skills,
                gaps=gaps,
                justification=justification,
            )
        )

    # ── Rest — without LLM explanation ──────────────────────────────────────
    for _, row in rest_df.iterrows():
        candidate_id = row["candidate_id"]
        candidate = candidate_map.get(candidate_id)

        if candidate is None:
            continue

        results.append(
            RankedCandidateResult(
                candidate_id=uuid.UUID(candidate_id),
                name=row["name"],
                rank=int(row["rank"]),
                final_score=float(row["final_score"]),
                semantic_sim=float(row["semantic_sim"]),
                skill_overlap=float(row["skill_overlap"]),
                exp_fit=float(row["exp_fit"]),
                activity_score=float(row["activity_score"]),
                matched_skills=row["matched_skills"],
                gaps=row["gaps"],
                justification=None,     # no LLM call for these
            )
        )

    return results