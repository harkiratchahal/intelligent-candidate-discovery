import pandas as pd
from datetime import datetime, timezone
from rapidfuzz import fuzz
from app.db.models import Candidates, Jobs


def compute_skill_overlap(
    candidate_skills: list[str],
    required_skills: list[str],
    threshold: int = 80,        # fuzzy match threshold 0-100
) -> tuple[float, list[str], list[str]]:
    """
    For each required skill, fuzzy match against candidate skills.
    Returns:
        overlap_ratio   → float 0.0–1.0
        matched_skills  → list of required skills that were matched
        gaps            → list of required skills that were NOT matched
    """
    if not required_skills:
        return 1.0, [], []

    candidate_skills = candidate_skills or []
    matched = []
    gaps = []

    for req_skill in required_skills:
        # check if any candidate skill fuzzy matches this required skill
        is_matched = any(
            fuzz.partial_ratio(req_skill.lower(), c_skill.lower()) >= threshold
            for c_skill in candidate_skills
        )
        if is_matched:
            matched.append(req_skill)
        else:
            gaps.append(req_skill)

    overlap_ratio = len(matched) / len(required_skills)
    return overlap_ratio, matched, gaps


def compute_exp_fit(
    candidate_experience: float,
    min_experience: float,
    overqualified_threshold: float = 10.0,  # years over min to be considered overqualified
) -> float:
    """
    Scores how well candidate experience matches job requirement.
        >= min_experience                           → 1.0
        within 1 year under min                    → 0.5
        more than 1 year under min                 → 0.0
        overqualified by more than threshold years  → 0.8 (slight penalty)
    """
    diff = candidate_experience - min_experience

    if diff >= overqualified_threshold:
        return 0.8      # overqualified — slight penalty
    elif diff >= 0:
        return 1.0      # meets or exceeds requirement
    elif diff >= -1.0:
        return 0.5      # within 1 year under — borderline
    else:
        return 0.0      # too underqualified


def compute_recency_score(last_active: datetime | None) -> float:
    """
    Scores how recently the candidate was active.
        active within 30 days   → 1.0
        active within 90 days   → 0.75
        active within 180 days  → 0.5
        active within 365 days  → 0.25
        older than 365 days     → 0.0
        no data                 → 0.5 (neutral — don't penalize missing data)
    """
    if last_active is None:
        return 0.5

    now = datetime.now(timezone.utc)

    # make last_active timezone aware if it isn't
    if last_active.tzinfo is None:
        last_active = last_active.replace(tzinfo=timezone.utc)

    days_since_active = (now - last_active).days

    if days_since_active <= 30:
        return 1.0
    elif days_since_active <= 90:
        return 0.75
    elif days_since_active <= 180:
        return 0.5
    elif days_since_active <= 365:
        return 0.25
    else:
        return 0.0


def compute_activity_score(
    last_active: datetime | None,
    profile_complete: float,
) -> float:
    """
    Combines recency + profile completeness into one activity score.
        activity_score = 0.5 * recency_score + 0.5 * profile_complete
    """
    recency = compute_recency_score(last_active)
    return round((0.5 * recency) + (0.5 * profile_complete), 4)


def extract(
    retrieved: list[dict],      # output of retrieval.py
    job: Jobs,
) -> pd.DataFrame:
    """
    Builds the feature table — one row per candidate, one column per feature.
    Input:  list of dicts from retrieval.py, each containing 'candidate' + 'semantic_sim'
    Output: pandas DataFrame with columns:
            candidate_id, name, semantic_sim, skill_overlap,
            exp_fit, activity_score, matched_skills, gaps
    """
    if not retrieved:
        return pd.DataFrame()

    rows = []

    for item in retrieved:
        candidate: Candidates = item["candidate"]
        semantic_sim: float = item["semantic_sim"]

        # ── Feature 2: skill overlap ─────────────────────────────────────────
        skill_overlap, matched_skills, gaps = compute_skill_overlap(
            candidate_skills=candidate.skills or [],
            required_skills=job.required_skills or [],
        )

        # ── Feature 3: experience fit ────────────────────────────────────────
        exp_fit = compute_exp_fit(
            candidate_experience=candidate.years_experience,
            min_experience=job.min_experience,
        )

        # ── Feature 4: activity score ────────────────────────────────────────
        activity_score = compute_activity_score(
            last_active=candidate.last_active,
            profile_complete=candidate.profile_complete,
        )

        rows.append({
            "candidate_id":   str(candidate.id),
            "name":           candidate.name,
            "semantic_sim":   round(semantic_sim, 4),
            "skill_overlap":  round(skill_overlap, 4),
            "exp_fit":        round(exp_fit, 4),
            "activity_score": round(activity_score, 4),
            "matched_skills": matched_skills,
            "gaps":           gaps,
        })

    df = pd.DataFrame(rows)
    return df