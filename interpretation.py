import json
import os
import re
import requests
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b"
EXPORTS_DIR = "exports"


def query_llm_for_summary(
    score_1: float,
    score_2: float,
    metrics: dict,
    test_breakdown: dict,
    sample_records: list[dict],
) -> str:
    """Queries Ollama (qwen3:8b) with an extended 300s timeout and clean fallbacks."""
    # Trim sample records to top 3 rows and clean up dict representations
    trimmed_samples = []
    for rec in sample_records[:3]:
        clean_rec = {
            k: (str(v)[:30] + "..." if len(str(v)) > 30 else v)
            for k, v in rec.items()
            if not k.startswith("FLAG_")
        }
        trimmed_samples.append(clean_rec)

    prompt = f"""
You are an expert Forensic Data Auditor. Analyze the statistical findings below and generate a concise executive audit summary.

### STATISTICAL SUMMARY
- Phase 3 Baseline Risk Score (Score 1): {score_1} / 100
- Phase 4 Targeted Forensic Score (Score 2): {score_2} / 100
- Summary Metrics: {json.dumps(metrics)}
- Forensic Test Breakdown: {json.dumps(test_breakdown)}

### SAMPLE HIGH-RISK RECORDS DETECTED
{json.dumps(trimmed_samples, indent=2)}

### INSTRUCTIONS
Provide a clear executive summary with the following sections:
1. Executive Risk Assessment (Classify as High, Medium, or Low Risk based on Score 1 and Score 2)
2. Key Anomalies Identified (Summarize top flagged patterns)
3. Actionable Recommendations for Audit Team

Keep explanations concrete, professional, and clear. Do NOT use markdown code blocks in your text output.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    try:
        # Extended timeout to 300s (5 minutes)
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        if response.status_code == 200:
            res_text = response.json().get("response", "").strip()
            if res_text:
                return res_text
        return generate_rule_based_fallback_summary(
            score_1, score_2, metrics, test_breakdown
        )
    except Exception as e:
        print(f"Phase 5 LLM Generation Warning: {e}. Using fallback summary.")
        return generate_rule_based_fallback_summary(
            score_1, score_2, metrics, test_breakdown
        )


def generate_rule_based_fallback_summary(
    score_1: float, score_2: float, metrics: dict, test_breakdown: dict
) -> str:
    """Generates a dynamic executive summary if LLM service is offline or times out."""
    avg_score = (score_1 + score_2) / 2.0
    risk_level = (
        "HIGH RISK"
        if avg_score >= 50
        else ("MEDIUM RISK" if avg_score >= 25 else "LOW RISK")
    )

    high_scoring_tests = [
        t for t, d in test_breakdown.items() if d.get("score", 0) > 30
    ]
    test_str = (
        ", ".join(high_scoring_tests)
        if high_scoring_tests
        else "standard statistical variation checks"
    )

    summary = f"""<b>1. Executive Risk Assessment: {risk_level}</b><br/>
The dataset evaluated yielded a Phase 3 Baseline Risk Score of <b>{score_1:.2f}/100</b> and a Phase 4 Targeted Forensic Score of <b>{score_2:.2f}/100</b>. Overall risk profile is classified as <b>{risk_level}</b> based on statistical anomaly distribution across evaluated records.<br/><br/>

<b>2. Key Anomalies Identified:</b><br/>
• Total flagged records during baseline evaluation: <b>{metrics.get('p3_flagged_count', 0)}</b>.<br/>
• Deep forensic execution identified elevated anomaly metrics within: <b>{test_str}</b>.<br/>
• Newly surfaced high-risk records requiring review: <b>{metrics.get('newly_surfaced_count', 0)}</b>.<br/><br/>

<b>3. Actionable Recommendations for Audit Team:</b><br/>
• Prioritize manual verification on records exhibiting high Phase 3 anomaly indices.<br/>
• Conduct cross-reference verification on flagged test subsets, specifically focusing on key variance and distribution deviations.<br/>
• Verify source data integrity for records exhibiting multi-test overlap.
"""
    return summary


def build_pdf_report(
    filename: str,
    score_1: float,
    score_2: float,
    metrics: dict,
    test_breakdown: dict,
    executive_text: str,
) -> str:
    """Generates a PDF audit document and saves it locally."""
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    clean_name = os.path.basename(filename).replace(".csv", "").replace(".xlsx", "")
    pdf_filename = f"Audit_Report_{clean_name}.pdf"
    pdf_path = os.path.join(EXPORTS_DIR, pdf_filename)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
    )
    tbl_header_style = ParagraphStyle(
        "TblHeader",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=10,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1A202C"),
        alignment=1,
    )
    tbl_header_dark_style = ParagraphStyle(
        "TblHeaderDark",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=10,
        fontName="Helvetica-Bold",
        textColor=colors.white,
        alignment=0,
    )

    story = []

    # Document Header
    story.append(Paragraph("FORENSIC AUDIT EXECUTIVE REPORT", title_style))
    story.append(
        Paragraph(
            f"Dataset Evaluated: <b>{filename}</b> | Status: Complete",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 10))

    # Metric Table
    metrics_data = [
        [
            Paragraph("Phase 3 Baseline Score", tbl_header_style),
            Paragraph("Phase 4 Forensic Score", tbl_header_style),
            Paragraph("P3 Flagged", tbl_header_style),
            Paragraph("P4 Flagged", tbl_header_style),
            Paragraph("Newly Surfaced", tbl_header_style),
        ],
        [
            f"{score_1:.2f} / 100",
            f"{score_2:.2f} / 100",
            str(metrics.get("p3_flagged_count", 0)),
            str(metrics.get("p4_flagged_count", 0)),
            str(metrics.get("newly_surfaced_count", 0)),
        ],
    ]
    t_metrics = Table(metrics_data, colWidths=[110, 110, 100, 100, 120])
    t_metrics.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 1), (-1, 1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, 1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ])
    )
    story.append(t_metrics)
    story.append(Spacer(1, 10))

    # Forensic Test Breakdown Table
    story.append(Paragraph("Forensic Test Breakdown", heading_style))
    test_rows = [[
        Paragraph("Test Executed", tbl_header_dark_style),
        Paragraph("Calculated Risk Score", tbl_header_dark_style),
        Paragraph("Records Flagged", tbl_header_dark_style),
    ]]
    for test_name, res in test_breakdown.items():
        test_rows.append([
            test_name,
            f"{res.get('score', 0):.2f}",
            str(res.get("flagged_count", 0)),
        ])

    t_breakdown = Table(test_rows, colWidths=[220, 160, 160])
    t_breakdown.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ])
    )
    story.append(t_breakdown)
    story.append(Spacer(1, 10))

    # Executive Findings & Recommendations Section
    story.append(
        Paragraph("Executive Findings & Recommendations", heading_style)
    )
    formatted_llm_text = re.sub(
        r"\*\*(.*?)\*\*", r"<b>\1</b>", executive_text
    ).replace("\n", "<br/>")
    story.append(Paragraph(formatted_llm_text, body_style))

    doc.build(story)
    return pdf_path


def run_phase_5_pipeline(
    filename: str,
    score_1: float,
    score_2: float,
    metrics: dict,
    test_breakdown: dict,
    sample_records: list[dict],
) -> tuple[str, str]:
    executive_text = query_llm_for_summary(
        score_1, score_2, metrics, test_breakdown, sample_records
    )
    pdf_path = build_pdf_report(
        filename, score_1, score_2, metrics, test_breakdown, executive_text
    )
    return executive_text, pdf_path