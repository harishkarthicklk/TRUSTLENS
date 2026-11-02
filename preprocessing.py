import os
import numpy as np
import pandas as pd


def clean_and_preprocess_forensic(
    file_path: str, output_path: str
) -> tuple[pd.DataFrame, dict]:
    """Forensic-safe preprocessing engine.

    Sanitizes strings, flags duplicates/missing values with sentinels (-99999 /
    MISSING_VALUE), preserves audit trails, and saves to output_path.
    """
    filename = os.path.basename(file_path)

    # 1. Load Data
    if filename.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            "Unsupported file format. Please upload CSV or Excel."
        )

    audit_summary = {
        "original_rows": len(df),
        "original_cols": len(df.columns),
        "exact_duplicates_count": int(df.duplicated().sum()),
        "missing_values_by_col": df.isnull().sum().to_dict(),
    }

    # 2. FRAUD FLAG: Preserve Exact Duplicates (Do NOT drop them)
    df["FLAG_IS_DUPLICATE_ROW"] = df.duplicated(keep=False)

    # 3. String & Currency Sanitization
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        if col == "FLAG_IS_DUPLICATE_ROW":
            continue
        # Strip whitespaces
        df[col] = df[col].astype(str).str.strip()

        # Try converting currency/numeric strings safely
        cleaned_str = (
            df[col]
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        converted_num = pd.to_numeric(cleaned_str, errors="coerce")

        if converted_num.notnull().mean() > 0.8:
            df[col] = converted_num

    # 4. Safe Date Parsing
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed_dates = pd.to_datetime(df[col], errors="coerce")
                if parsed_dates.notnull().mean() > 0.5:
                    df[col] = parsed_dates
            except Exception:
                pass

    # 5. FRAUD SAFE Imputation (Flag missing values)
    for col in list(df.columns):
        if col.startswith("FLAG_"):
            continue
        if df[col].isnull().sum() > 0:
            df[f"FLAG_MISSING_{col}"] = df[col].isnull()
            if np.issubdtype(df[col].dtype, np.number):
                df[col] = df[col].fillna(-99999)  # Forensic numeric sentinel
            else:
                df[col] = df[col].fillna("MISSING_VALUE")

    # 6. Save directly to target output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith(".csv"):
        df.to_csv(output_path, index=False)
    else:
        df.to_excel(output_path, index=False)

    return df, audit_summary