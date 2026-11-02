import numpy as np
import pandas as pd

# =====================================================================
# 9 INDIVIDUAL FORENSIC TEST MODULES
# =====================================================================


def run_outlier_detection(df: pd.DataFrame) -> tuple[float, list[int]]:
    """1. Outlier Detection: Detects numerical values outside 1.5 * IQR bounds."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [
        c
        for c in numeric_cols
        if not c.startswith("FLAG_") and not c.endswith("_ID")
    ]

    if not numeric_cols:
        return 0.0, []

    outlier_indices = set()
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices.update(outliers)

    outlier_ratio = len(outlier_indices) / len(df) if len(df) > 0 else 0
    score = min(outlier_ratio * 100 * 5, 100.0)  # Normalized 0-100 risk score
    return round(score, 2), list(outlier_indices)


def run_duplicate_detection(df: pd.DataFrame) -> tuple[float, list[int]]:
    """2. Duplicate Detection: Detects exact duplicate transactions or business entity keys."""
    # Priority columns for business logic duplicates
    key_candidates = [
        c
        for c in [
            "Transaction_ID",
            "Customer_ID",
            "Amount",
            "Total_Amount",
            "Date",
        ]
        if c in df.columns
    ]

    if len(key_candidates) >= 2:
        duplicates = df[df.duplicated(subset=key_candidates, keep=False)]
    else:
        duplicates = df[df.duplicated(keep=False)]

    dup_indices = list(duplicates.index)
    dup_ratio = len(dup_indices) / len(df) if len(df) > 0 else 0
    score = min(dup_ratio * 100 * 10, 100.0)
    return round(score, 2), dup_indices


def run_time_series_analysis(df: pd.DataFrame) -> tuple[float, list[int]]:
    """3. Time-Series Analysis: Identifies temporal volume spikes (>2.5 StDev above daily mean)."""
    date_col = next(
        (
            c
            for c in df.columns
            if ("date" in c.lower() or "time" in c.lower())
            and not c.startswith("FLAG_")
        ),
        None,
    )

    if not date_col:
        return 0.0, []

    # Parse dates flexibly across mixed formats (e.g., 9/18/2023 vs 05-08-23)
    temp_dates = pd.to_datetime(df[date_col], format="mixed", errors="coerce")
    valid_dates = temp_dates.dropna()

    if valid_dates.empty:
        return 0.0, []

    daily_counts = valid_dates.dt.date.value_counts()
    mean_daily = daily_counts.mean()
    std_daily = daily_counts.std()

    if std_daily == 0 or pd.isna(std_daily):
        return 0.0, []

    # Use 2.5 StDev threshold to capture subtle daily spikes
    threshold = mean_daily + 2.5 * std_daily
    spike_days = daily_counts[daily_counts > threshold].index
    flagged_indices = df[temp_dates.dt.date.isin(spike_days)].index.tolist()

    score = min(
        (len(spike_days) / max(len(daily_counts), 1)) * 100 * 5, 100.0
    )
    return round(score, 2), flagged_indices

def run_correlation_analysis(df: pd.DataFrame) -> tuple[float, list[int]]:
    """4. Correlation Analysis: Flags rows breaking strong linear relationships (|r| > 0.8)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [
        c
        for c in numeric_cols
        if not c.startswith("FLAG_") and df[c].nunique() > 5
    ]

    if len(numeric_cols) < 2:
        return 0.0, []

    corr_matrix = df[numeric_cols].corr().abs()
    flagged_indices = set()
    high_corr_pairs = 0

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            r = corr_matrix.loc[col1, col2]

            if r >= 0.8:  # Strong expected linear relationship
                high_corr_pairs += 1
                # Standardized residuals to find breakdown rows
                ratio = df[col1] / (df[col2] + 1e-5)
                med_ratio = ratio.median()
                mad = (ratio - med_ratio).abs().median() or 1.0
                anomalies = df[(0.6745 * (ratio - med_ratio).abs() / mad) > 4.0]
                flagged_indices.update(anomalies.index)

    score = min((len(flagged_indices) / max(len(df), 1)) * 100 * 5, 100.0)
    return round(score, 2), list(flagged_indices)


def run_benfords_law(df: pd.DataFrame) -> tuple[float, list[int]]:
    """5. Benford's Law: Evaluates first-digit distribution against Benford's Logarithmic Law."""
    target_col = next(
        (c for c in ["Amount", "Total_Amount", "Price"] if c in df.columns),
        None,
    )

    if not target_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_col = numeric_cols[0] if len(numeric_cols) > 0 else None

    if not target_col:
        return 0.0, []

    # Extract first non-zero digit
    valid_vals = df[target_col].dropna().abs()
    valid_vals = valid_vals[valid_vals > 0]

    first_digits = (
        valid_vals.astype(str)
        .str.extract(r"([1-9])")[0]
        .dropna()
        .astype(int)
    )

    if first_digits.empty:
        return 0.0, []

    # Observed vs Benford expected frequencies
    observed = first_digits.value_counts(normalize=True).reindex(
        range(1, 10), fill_value=0
    )
    benford_expected = pd.Series(
        [np.log10(1 + 1 / d) for d in range(1, 10)], index=range(1, 10)
    )

    # Mean Absolute Deviation (MAD) for Benford compliance
    mad_stat = (observed - benford_expected).abs().mean()

    # Thresholds: MAD > 0.015 indicates non-conformity (potential fraud/manipulation)
    score = min((mad_stat / 0.03) * 100, 100.0)

    # Flag records starting with anomalous over-represented digits
    over_rep_digits = observed[observed > (benford_expected * 1.8)].index
    flagged_indices = df[
        df[target_col]
        .astype(str)
        .str.extract(r"([1-9])")[0]
        .astype(float)
        .isin(over_rep_digits)
    ].index.tolist()

    return round(score, 2), flagged_indices


def run_measure_of_central_tendency(
    df: pd.DataFrame,
) -> tuple[float, list[int]]:
    """6. Measure of Central Tendency: Detects extreme skew between Mean and Median."""
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if not c.startswith("FLAG_")
    ]
    if not numeric_cols:
        return 0.0, []

    flagged_indices = set()
    skew_scores = []

    for col in numeric_cols:
        mean_val = df[col].mean()
        median_val = df[col].median()
        std_val = df[col].std() or 1.0

        # Non-parametric skewness measure
        skewness = abs(mean_val - median_val) / std_val
        skew_scores.append(skewness)

        if skewness > 0.5:  # Significant skew
            extreme_rows = df[(df[col] - mean_val).abs() > 3 * std_val].index
            flagged_indices.update(extreme_rows)

    avg_skew = np.mean(skew_scores) if skew_scores else 0
    score = min(avg_skew * 100, 100.0)
    return round(score, 2), list(flagged_indices)


def run_measure_of_dispersion(df: pd.DataFrame) -> tuple[float, list[int]]:
    """7. Measure of Dispersion: Evaluates Variance, IQR, and Coefficient of Variation (CV)."""
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if not c.startswith("FLAG_")
    ]
    if not numeric_cols:
        return 0.0, []

    flagged_indices = set()
    cv_scores = []

    for col in numeric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()

        if mean_val != 0 and not pd.isna(std_val):
            cv = std_val / abs(mean_val)  # Coefficient of Variation
            cv_scores.append(cv)

            if cv > 1.5:  # High dispersion dataset
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                dispersion_outliers = df[
                    (df[col] < (q1 - 2.0 * iqr)) | (df[col] > (q3 + 2.0 * iqr))
                ].index
                flagged_indices.update(dispersion_outliers)

    avg_cv = np.mean(cv_scores) if cv_scores else 0
    score = min((avg_cv / 2.0) * 100, 100.0)
    return round(score, 2), list(flagged_indices)


def run_largest_subset_test(df: pd.DataFrame) -> tuple[float, list[int]]:
    """8. Largest Subset Test: Finds disproportionately large categorical groupings."""
    cat_cols = [
        c
        for c in df.select_dtypes(include=["object", "category"]).columns
        if not c.endswith("_date") and df[c].nunique() > 1
    ]

    if not cat_cols:
        return 0.0, []

    flagged_indices = set()
    concentration_ratios = []

    for col in cat_cols:
        top_group_size = df[col].value_counts().max()
        ratio = top_group_size / len(df)
        concentration_ratios.append(ratio)

        # Flag if a single category dominates > 40% of all transaction records
        if ratio > 0.40:
            top_cat = df[col].value_counts().idxmax()
            flagged_indices.update(df[df[col] == top_cat].index)

    max_ratio = max(concentration_ratios) if concentration_ratios else 0
    score = min(max_ratio * 100, 100.0)
    return round(score, 2), list(flagged_indices)


def run_largest_growth_test(df: pd.DataFrame) -> tuple[float, list[int]]:
    """9. Largest Growth Test: Detects sudden entity volume growth spikes over time."""
    date_col = next(
        (c for c in df.columns if "date" in c.lower() or "year" in c.lower()),
        None,
    )
    entity_col = next(
        (
            c
            for c in df.columns
            if "customer" in c.lower()
            or "vendor" in c.lower()
            or "store" in c.lower()
        ),
        None,
    )

    if not date_col or not entity_col:
        return 0.0, []

    df_temp = df.copy()
    df_temp["_dt"] = pd.to_datetime(df_temp[date_col], errors="coerce")
    df_temp = df_temp.dropna(subset=["_dt"])

    if df_temp.empty:
        return 0.0, []

    # Group entity count by month
    growth = (
        df_temp.groupby([entity_col, pd.Grouper(key="_dt", freq="ME")])
        .size()
        .unstack(fill_value=0)
    )

    if growth.shape[1] < 2:
        return 0.0, []

    # Calculate month-over-month growth rate
    mom_growth = growth.pct_change(axis=1).fillna(0)
    max_growth_entities = mom_growth.max(axis=1)

    # Flag entities experiencing > 300% sudden growth
    spiking_entities = max_growth_entities[max_growth_entities > 3.0].index
    flagged_indices = df[df[entity_col].isin(spiking_entities)].index.tolist()

    score = min((len(spiking_entities) / max(len(growth), 1)) * 100, 100.0)
    return round(score, 2), flagged_indices


# =====================================================================
# MASTER EXECUTOR MAP
# =====================================================================
TEST_EXECUTOR_POOL = {
    "Outlier Detection": run_outlier_detection,
    "Duplicate Detection": run_duplicate_detection,
    "Time-Series Analysis": run_time_series_analysis,
    "Correlation Analysis": run_correlation_analysis,
    "Benford's Law": run_benfords_law,
    "Measure of Central Tendency": run_measure_of_central_tendency,
    "Measure of Dispersion": run_measure_of_dispersion,
    "Largest Subset Test": run_largest_subset_test,
    "Largest Growth Test": run_largest_growth_test,
}


def execute_phase4_tests(df: pd.DataFrame, selected_tests: list[str]) -> dict:
    """Executes ONLY the user/LLM selected tests from the pool and calculates Score 2."""
    test_results = {}
    all_flagged_indices = set()
    test_scores = []

    for test_name in selected_tests:
        executor = TEST_EXECUTOR_POOL.get(test_name)
        if executor:
            score, flagged_idxs = executor(df)
            test_results[test_name] = {
                "score": score,
                "flagged_count": len(flagged_idxs),
            }
            test_scores.append(score)
            all_flagged_indices.update(flagged_idxs)
        else:
            test_results[test_name] = {
                "score": 0.0,
                "flagged_count": 0,
                "status": "Test module not recognized",
            }

    # Score 2 is the average risk score across chosen tests
    score_2 = round(float(np.mean(test_scores)), 2) if test_scores else 0.0

    # Cross-reference with Phase 3 Flagged Records
    p3_flagged = set(
        df[df.get("PHASE3_FLAGGED", False) == True].index  # noqa: E712
    )

    overlap_records = list(p3_flagged.intersection(all_flagged_indices))
    new_phase4_records = list(all_flagged_indices - p3_flagged)
    combined_union = list(p3_flagged.union(all_flagged_indices))

    return {
        "score_2": score_2,
        "individual_test_results": test_results,
        "total_phase4_flagged": len(all_flagged_indices),
        "overlapping_p3_p4_count": len(overlap_records),
        "newly_surfaced_count": len(new_phase4_records),
        "combined_flagged_indices": combined_union,
    }