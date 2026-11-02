# TrustLens: Autonomous Forensic Data Auditing & Risk Scoring

Developed as part of the course on **"Forensic Analytics"** at **Vellore Institute of Technology, Chennai**.

* **Authors:**
  * HARISH KARTHICK L.K (`23MIA1100`)
  * JAGAN NIVASAN M (`23MIA1137`)
* **Subject Code:** CSE4074
* **Semester:** Fall 2026–27
* **Course Professor:** Dr. S. Johanan Joysingh

## 📌 Project Overview

**TrustLens** is an autonomous, schema-agnostic forensic data auditing framework designed to automate anomaly detection, risk scoring, and audit report generation across enterprise datasets.

Traditional auditing workflows rely on rigid SQL/Excel rules that miss complex non-linear fraud patterns, or black-box machine learning models (e.g., Isolation Forest) that lack domain context. TrustLens solves this by pairing **Local LLM Inference** (powered by Ollama `qwen3:8b`) with a **9-Module Deterministic Forensic Engine**.

```
+-----------------------------------------------------------------------------------+
|                            PHASE 1: INGESTION & CLEANING                          |
|  • Payload Validation  • Regex Sanitization  • Missing Value Sentinel (-99999)    |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
|                            PHASE 2: AUTONOMOUS DISCOVERY                          |
|  • 1k Row Schema Sampling  • LLM (Qwen3:8b) Domain Inference & Test Selection     |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
|                     PHASE 3: FEATURE SELECTION & ML BASELINE                      |
|  • LLM Feature Filtering (Score ≥ 0.5)  • Median Imputation  • Isolation Forest    |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
|                        PHASE 4: TARGETED FORENSIC EXECUTION                       |
|  • Execution of Selected Tests (1 of 9)  • Overlap & Surfaced Anomaly Resolution  |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
|                     PHASE 5: SYNTHESIS & REPORT GENERATION                        |
|  • Executive Narrative Generation  • PDF Render (ReportLab)                       |
+-----------------------------------------------------------------------------------+
```

## 🚀 Key Architectural Features

1. **Phase 1: Ingestion & Sanitization (`preprocessing.py` & `main.py`)**
   * Strips whitespace, normalizes headers, and converts string monetary fields (e.g., `"$1,250.50"`, `"12%"`) to float metrics via regex.
   * Flags full-row duplicates (`FLAG_IS_DUPLICATE_ROW`) and preserves missingness metadata using a temporary `-99999` sentinel.

2. **Phase 2: Autonomous LLM Discovery (`discovery.py`)**
   * Samples 1,000 rows and sends structural metadata JSON to Ollama (`qwen3:8b`).
   * Classifies industry domain and selects 2–4 targeted forensic tests from `FORENSIC_TEST_POOL`.

3. **Phase 3: Dynamic Feature Selection & Unsupervised ML (`feature_selector.py` & `scoring.py`)**
   * Filters numeric columns based on domain relevance ($score \ge 0.5$).
   * Imputes `-99999` sentinels with column medians to protect distance bounds.
   * Fits an `IsolationForest` (100 trees, 0.10 contamination) to calculate continuous **Score 1** (0–100 scale).

4. **Phase 4: Targeted Forensic Rule Execution (`forensics.py`)**
   * Executes selected algorithmic test modules (Outlier, Duplicate, Time-Series, Correlation, Benford's Law, Central Tendency, Dispersion, Largest Subset, Largest Growth).
   * Computes aggregated **Score 2** (arithmetic mean of test scores).
   * Cross-references records: **Overlapping Records** ($S_{P3} \cap S_{P4}$) vs. **Newly Surfaced Risks** ($S_{P4} \setminus S_{P3}$).

5. **Phase 5: Audit Report Synthesis & PDF Compilation (`interpretation.py`)**
   * Prompts Ollama to generate executive audit narratives with dynamic rule-based fallbacks.
   * Compiles styled, publication-ready PDF audit reports using ReportLab.

## 🧮 Mathematical Foundations

### Phase 3: Isolation Forest & Baseline Scaling

* **BST Path Length Normalization Factor:**
  $$c(n) = 2(\ln(n - 1) + 0.5772156649) - \frac{2(n - 1)}{n}$$

* **Continuous Min-Max Score 1 Scaling:**
  $$\text{Score 1}_i = 100 \times \left( 1 - \frac{s_{\text{raw},i} - s_{\text{min}}}{s_{\text{max}} - s_{\text{min}} + 10^{-5}} \right)$$

### Phase 4: Core Module Formulas

* **Outlier Detection:**
  $$\text{Score}_{\text{Outlier}} = \min\left( \frac{\text{Count}(\text{Outliers})}{N} \times 100 \times 5, \, 100.0 \right)$$

* **Duplicate Detection:**
  $$\text{Score}_{\text{Duplicate}} = \min\left( \frac{\text{Count}(\text{Duplicates})}{N} \times 100 \times 10, \, 100.0 \right)$$

* **Benford's Law MAD Score:**
  $$\text{MAD} = \frac{1}{9} \sum_{d=1}^{9} \vert{}P_{\text{observed}}(d) - \log_{10}(1 + 1/d)\vert{}, \quad \text{Score} = \min\left( \frac{\text{MAD}}{0.03} \times 100, \, 100.0 \right)$$

* **Correlation Modified Z-Score:**
  $$\text{Modified Z}_i = \frac{0.6745 \times \vert{}R_i - \text{Median}(R)\vert{}}{\text{MAD}(R)} > 4.0$$

## ⚡ Quick Start & Run Commands

### 1. Start Local Ollama LLM Service
```bash
ollama run qwen3:8B
```

### 2. Backend Environment Setup & Launch (FastAPI)
Activate your Python virtual environment from the project root directory and run the Uvicorn server:

**Windows (cmd/powershell):**
```bash
venv\Scripts\activate
# OR if using .venv:
.venv\Scripts\activate
```

**Launch Uvicorn Backend Server:**
```bash
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup & Launch (React)
Open a new terminal window, navigate to the frontend directory, and start the app:
```bash
cd frontend
npm start
```

## 📁 Repository Structure

```
trustlens/
├── discovery.py           # Phase 2 Ollama domain inference & test selection
├── feature_selector.py    # Phase 3 feature relevance filtering
├── forensics.py           # Phase 4 9-module forensic test execution (Score 2)
├── interpretation.py      # Phase 5 report synthesis & ReportLab PDF engine
├── main.py                # FastAPI endpoints & main pipeline execution
├── preprocessing.py       # Phase 1 schema sanitization & sentinel tagging
├── scoring.py             # Phase 3 Isolation Forest baseline scoring (Score 1)
├── requirements.txt       # Python project dependencies
├── start.txt              # Execution commands reference
├── exports/               # Synthesized PDF Audit Reports
├── frontend/              # React dashboard UI
├── PPT's/                 # Project presentation decks & HTML slides
├── preprocessed/          # Cleaned CSV outputs
└── uploads/               # Temporary uploaded datasets
```

## 📜 License

Developed for academic assessment under the **CSE4074 Forensic Analytics** course at VIT Chennai.