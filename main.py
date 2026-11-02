import os
import shutil
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Internal Module Imports
from discovery import run_phase2_discovery
from forensics import execute_phase4_tests
from interpretation import run_phase_5_pipeline
from preprocessing import clean_and_preprocess_forensic
from scoring import run_phase3_scoring


# Pydantic Schemas
class Phase3Request(BaseModel):
    filename: str
    threshold: float = 50.0
    classified_domain: str = "General Ledger / Financial Operations"
    recommended_tests: list[dict] = []


class Phase4Request(BaseModel):
    filename: str
    selected_tests: list[str]


class Phase5Request(BaseModel):
    filename: str
    score_1: float
    score_2: float
    metrics: dict
    test_breakdown: dict
    sample_records: list[dict]


app = FastAPI(
    title="TrustLens API",
    description="Backend service for TrustLens AI-Powered Forensic Data Engine",
)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "uploads"
PREPROCESSED_DIR = "preprocessed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)


# ==========================================
# PHASE 1: INGESTION & PREPROCESSING
# ==========================================
@app.post("/api/v1/phase1/upload")
async def upload_and_preprocess(file: UploadFile = File(...)):
    """Ingests raw file, performs forensic preprocessing, and saves to /preprocessed."""
    allowed_extensions = (".csv", ".xlsx", ".xls")
    if not file.filename.endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only CSV and Excel files are supported.",
        )

    raw_filename = os.path.basename(file.filename)
    clean_filename = f"clean_{raw_filename}"

    raw_path = os.path.join(UPLOAD_DIR, raw_filename)
    clean_path = os.path.join(PREPROCESSED_DIR, clean_filename)

    try:
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        _, audit_summary = clean_and_preprocess_forensic(raw_path, clean_path)

        return {
            "status": "success",
            "message": "File successfully preprocessed and sanitized.",
            "raw_file": raw_filename,
            "preprocessed_file": clean_filename,
            "metrics": audit_summary,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Preprocessing failed: {str(e)}"
        )


# ==========================================
# PHASE 2: DOMAIN DISCOVERY & TEST SELECTION
# ==========================================
@app.post("/api/v1/phase2/discovery")
async def phase2_discovery(filename: str):
    """Loads preprocessed file, extracts schema, and queries qwen3:8b via Ollama."""
    clean_path = os.path.join(PREPROCESSED_DIR, filename)

    if not os.path.exists(clean_path):
        raise HTTPException(
            status_code=404,
            detail=f"Preprocessed file '{filename}' not found. Run Phase 1 first.",
        )

    try:
        discovery_results = run_phase2_discovery(clean_path, filename)
        return {"status": "success", "data": discovery_results}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Phase 2 Discovery failed: {str(e)}"
        )


# ==========================================
# PHASE 3: NORMALITY & BASELINE RISK SCORING (SCORE 1)
# ==========================================
@app.post("/api/v1/phase3/scoring")
async def phase3_scoring(request: Phase3Request):
    """Calculates Phase 3 Baseline Risk Score using LLM feature selection and safe median imputation."""
    clean_path = os.path.join(PREPROCESSED_DIR, request.filename)

    if not os.path.exists(clean_path):
        raise HTTPException(
            status_code=404, detail=f"File '{request.filename}' not found."
        )

    try:
        scored_df, metrics = run_phase3_scoring(
            clean_path=clean_path,
            threshold=request.threshold,
            classified_domain=request.classified_domain,
            recommended_tests=request.recommended_tests,
        )

        if request.filename.endswith(".csv"):
            scored_df.to_csv(clean_path, index=False)
        else:
            scored_df.to_excel(clean_path, index=False)

        flagged_preview = (
            scored_df[scored_df["PHASE3_FLAGGED"]]
            .sort_values(by="SCORE_1", ascending=False)
            .head(10)
            .fillna("")
            .to_dict(orient="records")
        )

        return {
            "status": "success",
            "filename": request.filename,
            "metrics": metrics,
            "flagged_preview": flagged_preview,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Phase 3 scoring failed: {str(e)}"
        )


# ==========================================
# PHASE 4: TARGETED FORENSIC EXECUTION (SCORE 2)
# ==========================================
@app.post("/api/v1/phase4/execution")
async def phase4_execution(request: Phase4Request):
    """Executes chosen forensic tests from the pool against the dataset and calculates Score 2."""
    clean_path = os.path.join(PREPROCESSED_DIR, request.filename)

    if not os.path.exists(clean_path):
        raise HTTPException(
            status_code=404, detail=f"File '{request.filename}' not found."
        )

    try:
        import pandas as pd

        df = (
            pd.read_csv(clean_path)
            if request.filename.endswith(".csv")
            else pd.read_excel(clean_path)
        )

        if "PHASE3_FLAGGED" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="Run Phase 3 scoring prior to executing Phase 4.",
            )

        execution_summary = execute_phase4_tests(df, request.selected_tests)
        df["SCORE_2"] = execution_summary["score_2"]

        if request.filename.endswith(".csv"):
            df.to_csv(clean_path, index=False)
        else:
            df.to_excel(clean_path, index=False)

        combined_indices = execution_summary["combined_flagged_indices"]
        high_risk_sample = (
            df.iloc[combined_indices].head(10).fillna("").to_dict(orient="records")
        )

        return {
            "status": "success",
            "filename": request.filename,
            "score_2": execution_summary["score_2"],
            "test_breakdown": execution_summary["individual_test_results"],
            "summary": {
                "p3_flagged_count": int(df["PHASE3_FLAGGED"].sum()),
                "p4_flagged_count": execution_summary["total_phase4_flagged"],
                "overlap_count": execution_summary["overlapping_p3_p4_count"],
                "newly_surfaced_count": execution_summary["newly_surfaced_count"],
            },
            "high_risk_context_sample": high_risk_sample,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Phase 4 execution failed: {str(e)}"
        )


# ==========================================
# PHASE 5: LLM INTERPRETATION & PDF GENERATION
# ==========================================
@app.post("/api/v1/phase5/interpretation")
async def phase5_interpretation(request: Phase5Request):
    """Executes Phase 5: LLM summary + PDF report generation."""
    try:
        summary_text, pdf_path = run_phase_5_pipeline(
            filename=request.filename,
            score_1=request.score_1,
            score_2=request.score_2,
            metrics=request.metrics,
            test_breakdown=request.test_breakdown,
            sample_records=request.sample_records,
        )
        return {
            "status": "success",
            "filename": request.filename,
            "pdf_report_path": pdf_path,
            "download_url": f"/api/v1/phase5/download?pdf_path={pdf_path}",
            "executive_summary": summary_text,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Phase 5 generation failed: {str(e)}"
        )


@app.get("/api/v1/phase5/download")
async def download_pdf(pdf_path: str):
    """Allows directly downloading the generated PDF report."""
    if os.path.exists(pdf_path):
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=os.path.basename(pdf_path),
        )
    raise HTTPException(status_code=404, detail="PDF report file not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)