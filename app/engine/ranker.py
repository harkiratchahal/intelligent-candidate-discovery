import pandas as pd
from app.config import settings


# Weights for the weighted sum formula
# Must sum to 1.0
WEIGHTS = {
    "semantic_sim":   0.35,
    "skill_overlap":  0.30,
    "exp_fit":        0.20,
    "activity_score": 0.15,
}


def score(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the feature DataFrame from features.py and:
    1. Computes a final_score per candidate using weighted sum
    2. Sorts by final_score descending
    3. Assigns rank column (1 = best)

    Input:  DataFrame with columns:
            candidate_id, name, semantic_sim, skill_overlap,
            exp_fit, activity_score, matched_skills, gaps

    Output: Same DataFrame with two new columns added:
            final_score, rank
    """
    if feature_df.empty:
        return feature_df

    # ── Weighted sum ─────────────────────────────────────────────────────────
    feature_df["final_score"] = (
        WEIGHTS["semantic_sim"]   * feature_df["semantic_sim"]
        + WEIGHTS["skill_overlap"]  * feature_df["skill_overlap"]
        + WEIGHTS["exp_fit"]        * feature_df["exp_fit"]
        + WEIGHTS["activity_score"] * feature_df["activity_score"]
    ).round(4)

    # ── Sort descending by final_score ───────────────────────────────────────
    feature_df = feature_df.sort_values(
        by="final_score",
        ascending=False,
    ).reset_index(drop=True)

    # ── Assign rank (1-indexed) ──────────────────────────────────────────────
    feature_df["rank"] = feature_df.index + 1

    return feature_df


def get_top_k(ranked_df: pd.DataFrame, k: int = None) -> pd.DataFrame:
    """
    Slices the ranked DataFrame to top k rows.
    Used by rerank_explain.py to get only the candidates
    that go to the LLM explanation stage.
    """
    k = k or settings.EXPLANATION_TOP_K
    return ranked_df.head(k).reset_index(drop=True)