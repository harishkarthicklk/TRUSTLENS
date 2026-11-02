import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from feature_selector import extract_relevant_features


def run_phase3_scoring(
    clean_path: str,
    threshold: float = 50.0,
    classified_domain: str = "Unclassified Domain",
    recommended_tests: list = None,
) -> tuple[pd.DataFrame, dict]:
    if recommended_tests is None:
        recommended_tests = []

    df = (
        pd.read_csv(clean_path)
        if clean_path.endswith(".csv")
        else pd.read_excel(clean_path)
    )

    # Pure LLM feature selection based on the domain & tests identified in Phase 2
    selected_cols = extract_relevant_features(
        df=df,
        classified_domain=classified_domain,
        recommended_tests=recommended_tests,
        min_relevance_threshold=0.5,
    )

    if not selected_cols:
        selected_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if not c.startswith("FLAG_")
        ]

    # Safe neutral median imputation
    X = df[selected_cols].copy()
    for col in selected_cols:
        X[col] = X[col].replace(-99999, np.nan)
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0)

    # Fit Isolation Forest strictly on Ollama-selected features
    iso = IsolationForest(
        contamination=0.10, random_state=42, n_estimators=100
    )
    iso.fit(X)
    decision_scores = iso.decision_function(X)

    # Continuous 0-100 normalization
    min_s = decision_scores.min()
    max_s = decision_scores.max()
    normalized_scores = 100 * (
        1 - (decision_scores - min_s) / (max_s - min_s + 1e-5)
    )
    df["SCORE_1"] = np.round(normalized_scores, 2)
    df["PHASE3_FLAGGED"] = df["SCORE_1"] >= threshold

    score_1_global = float(np.round(df["SCORE_1"].mean(), 2))
    p3_flagged_count = int(df["PHASE3_FLAGGED"].sum())

    metrics = {
        "score_1": score_1_global,
        "p3_flagged_count": p3_flagged_count,
        "global_risk_score": score_1_global,
        "flagged_count": p3_flagged_count,
        "total_records": len(df),
        "selected_features": selected_cols,
    }

    return df, metrics