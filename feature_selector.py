import json
import numpy as np
import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"


def extract_relevant_features(
    df: pd.DataFrame,
    classified_domain: str,
    recommended_tests: list[dict],
    min_relevance_threshold: float = 0.5,
) -> list[str]:
    """Pure LLM Feature Selection Engine:

    - Passes raw dataset numerical columns, domain, and recommended tests to
    Ollama. - Ollama decides column relevance (0.0 to 1.0) based on domain context
    and selected tests. - No hardcoded keyword patterns or static schema rules.
    """
    # Select all numerical columns, excluding system flags
    all_numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if not c.startswith("FLAG_")
        and not c.startswith("PHASE")
        and not c.startswith("SCORE_")
    ]

    if not all_numeric_cols:
        return []

    column_samples = {
        col: df[col].dropna().head(3).tolist() for col in all_numeric_cols
    }

    test_names = [
        t.get("test_name", str(t)) if isinstance(t, dict) else str(t)
        for t in recommended_tests
    ]

    prompt = f"""You are an expert Forensic Data Architect.

Target Domain Defined by Phase 2: {classified_domain}
Selected Forensic Tests: {json.dumps(test_names)}

Here are all numerical columns in the dataset along with sample values:
{json.dumps(column_samples, indent=2)}

Your Task:
Evaluate each numerical column and determine its relevance to performing the selected forensic tests within the defined domain.

Rules:
- Assign a relevance score from 0.0 to 1.0 to each column.
- Columns that act as generic metadata, noise, or are non-relevant to the chosen tests and domain should receive low scores (< 0.5).
- Columns representing primary metrics, financial values, or behavioral signals needed for the selected tests should receive high scores (>= 0.5).

Return ONLY a valid JSON object matching this exact structure:
{{
  "relevance_scores": {{
    "column_name_1": 0.9,
    "column_name_2": 0.1
  }}
}}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()

        raw_res = response.json().get("response", "{}")
        scores = json.loads(raw_res).get("relevance_scores", {})

        # Retain features meeting or exceeding the relevance threshold
        selected_cols = [
            col
            for col in all_numeric_cols
            if float(scores.get(col, 1.0)) >= min_relevance_threshold
        ]

        # Guardrail: If Ollama filters out all columns, fallback to all numeric features
        return selected_cols if selected_cols else all_numeric_cols

    except Exception as e:
        print(f"LLM Feature Selection Warning: {e}")
        return all_numeric_cols