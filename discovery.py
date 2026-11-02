import json
import pandas as pd
import requests

FORENSIC_TEST_POOL = [
    "Outlier Detection",
    "Duplicate Detection",
    "Time-Series Analysis",
    "Correlation Analysis",
    "Benford's Law",
    "Measure of Central Tendency",
    "Measure of Dispersion",
    "Largest Subset Test",
    "Largest Growth Test",
]

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"


def run_phase2_discovery(file_path: str, filename: str) -> dict:
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, nrows=1000)
    else:
        df = pd.read_excel(file_path, nrows=1000)

    schema_info = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}

    sample_snippet = []
    for record in df.head(3).fillna("").to_dict(orient="records"):
        clean_record = {
            k: (str(v)[:30] + "..." if len(str(v)) > 30 else v)
            for k, v in record.items()
        }
        sample_snippet.append(clean_record)

    prompt = f"""
You are an expert Forensic Data Auditor analyzing an uploaded dataset to detect anomalies, risk factors, or fraud.

Dataset Schema:
{json.dumps(schema_info, indent=2)}

Data Sample (Top 3 Rows):
{json.dumps(sample_snippet, indent=2)}

Your task:
1. Dynamically define the industry/domain of this dataset purely based on its columns and sample values.
2. Select 2 to 4 forensic tests from the allowed pool that are most applicable to this domain and schema.

ALLOWED FORENSIC TEST POOL:
{json.dumps(FORENSIC_TEST_POOL, indent=2)}

CRITICAL RULES:
- Define the domain dynamically without assuming any hardcoded industry defaults.
- Every test_name in your output MUST match an item in the allowed pool exactly.

Respond ONLY with a valid JSON object matching this exact structure:
{{
  "domain_classification": "<defined business domain>",
  "recommended_tests": [
    {{
      "test_name": "<EXACT name from the allowed pool>",
      "reasoning": "<why this test applies to this domain and schema>"
    }}
  ]
}}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()

        raw_llm_response = response.json().get("response", "{}")
        llm_output = json.loads(raw_llm_response)

        raw_tests = llm_output.get("recommended_tests", [])

        valid_tests = [
            test
            for test in raw_tests
            if isinstance(test, dict)
            and test.get("test_name") in FORENSIC_TEST_POOL
        ]

        return {
            "filename": filename,
            "domain_classification": llm_output.get(
                "domain_classification", "Unclassified Domain"
            ),
            "recommended_tests": valid_tests,
            "available_pool": FORENSIC_TEST_POOL,
        }

    except Exception as e:
        raise RuntimeError(f"Ollama Phase 2 Discovery failed: {str(e)}")