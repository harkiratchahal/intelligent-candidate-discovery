import uuid
from datetime import datetime, timezone
from sqlmodel import Session, select

from app.db.models import Candidates, Jobs, PipelineRun, RankedResult, RunStatus
from app.models.result import PipelineRunRead, RankedCandidateResult
from app.engine import filters, retrieval, features, ranker, rerank_explain


def run(job_id: uuid.UUID, session: Session) -> PipelineRunRead:
    """
    Main pipeline orchestrator.
    Runs all 5 stages in order and saves results to DB.

    Flow:
        Stage 0 — hard filters
        Stage 1 — BM25 + embedding retrieval + RRF
        Stage 2 — feature extraction
        Stage 3 — weighted sum scoring + ranking
        Stage 4/5 — LLM re-rank + explanation (top k only)
    """

    # ── Load job from DB ─────────────────────────────────────────────────────
    job = session.get(Jobs, job_id)
    if job is None:
        raise ValueError(f"Job {job_id} not found in database")

    # ── Create PipelineRun record ────────────────────────────────────────────
    pipeline_run = PipelineRun(
        job_id=job_id,
        status=RunStatus.pending,
    )
    session.add(pipeline_run)
    session.commit()
    session.refresh(pipeline_run)

    try:
        # ── Update status to running ─────────────────────────────────────────
        pipeline_run.status = RunStatus.running
        session.add(pipeline_run)
        session.commit()

        # ── Load all candidates from DB ──────────────────────────────────────
        all_candidates: list[Candidates] = list(
            session.exec(select(Candidates)).all()
        )

        if not all_candidates:
            raise ValueError("No candidates found in database")

        # ── Stage 0: Hard Filters ────────────────────────────────────────────
        eligible = filters.apply(all_candidates, job)

        if not eligible:
            raise ValueError("No candidates passed the hard filters")

        # ── Stage 1: Retrieval ───────────────────────────────────────────────
        retrieved = retrieval.retrieve(eligible, job)

        if not retrieved:
            raise ValueError("Retrieval returned no candidates")

        # ── Stage 2: Feature Extraction ──────────────────────────────────────
        feature_df = features.extract(retrieved, job)

        if feature_df.empty:
            raise ValueError("Feature extraction returned empty DataFrame")

        # ── Stage 3: Scoring + Ranking ───────────────────────────────────────
        ranked_df = ranker.score(feature_df)

        # ── Stage 4/5: LLM Re-rank + Explanation ────────────────────────────
        results: list[RankedCandidateResult] = rerank_explain.run(
            ranked_df=ranked_df,
            candidates=eligible,
            job=job,
        )

        # ── Save RankedResults to DB ─────────────────────────────────────────
        for result in results:
            ranked_result = RankedResult(
                run_id=pipeline_run.id,
                candidate_id=result.candidate_id,
                rank=result.rank,
                final_score=result.final_score,
                semantic_sim=result.semantic_sim,
                skill_overlap=result.skill_overlap,
                exp_fit=result.exp_fit,
                activity_score=result.activity_score,
                matched_skills=result.matched_skills,
                gaps=result.gaps,
                justification=result.justification,
            )
            session.add(ranked_result)

        # ── Mark run as completed ────────────────────────────────────────────
        pipeline_run.status = RunStatus.completed
        pipeline_run.completed_at = datetime.now(timezone.utc)
        session.add(pipeline_run)
        session.commit()

    except Exception as e:
        # ── Mark run as failed ───────────────────────────────────────────────
        pipeline_run.status = RunStatus.failed
        pipeline_run.error = str(e)
        pipeline_run.completed_at = datetime.now(timezone.utc)
        session.add(pipeline_run)
        session.commit()
        raise

    # ── Build and return PipelineRunRead ────────────────────────────────────
    return PipelineRunRead(
        run_id=pipeline_run.id,
        job_id=pipeline_run.job_id,
        status=pipeline_run.status,
        created_at=pipeline_run.created_at,
        completed_at=pipeline_run.completed_at,
        error=pipeline_run.error,
        results=results,
    )


def get_run(run_id: uuid.UUID, session: Session) -> PipelineRunRead:
    """
    Fetches a completed pipeline run from DB by run_id.
    Used by GET /pipeline/{run_id} route.
    """
    pipeline_run = session.get(PipelineRun, run_id)
    if pipeline_run is None:
        raise ValueError(f"Pipeline run {run_id} not found")

    # load ranked results for this run ordered by rank
    ranked_results = session.exec(
        select(RankedResult)
        .where(RankedResult.run_id == run_id)
        .order_by(RankedResult.rank)
    ).all()

    # load candidate names in one query
    candidate_ids = [r.candidate_id for r in ranked_results]
    candidates = session.exec(
        select(Candidates).where(Candidates.id.in_(candidate_ids))
    ).all()
    candidate_map = {c.id: c.name for c in candidates}

    results = [
        RankedCandidateResult(
            candidate_id=r.candidate_id,
            name=candidate_map.get(r.candidate_id, "Unknown"),
            rank=r.rank,
            final_score=r.final_score,
            semantic_sim=r.semantic_sim,
            skill_overlap=r.skill_overlap,
            exp_fit=r.exp_fit,
            activity_score=r.activity_score,
            matched_skills=r.matched_skills or [],
            gaps=r.gaps or [],
            justification=r.justification,
        )
        for r in ranked_results
    ]

    return PipelineRunRead(
        run_id=pipeline_run.id,
        job_id=pipeline_run.job_id,
        status=pipeline_run.status,
        created_at=pipeline_run.created_at,
        completed_at=pipeline_run.completed_at,
        error=pipeline_run.error,
        results=results,
    )