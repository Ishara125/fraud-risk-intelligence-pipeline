"""Generates the final 07_business_fraud_strategy.docx report automatically."""
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from pathlib import Path

FIG = Path("reports/figures")

def add_heading(doc, text, level):
    heading = doc.add_heading(text, level=level)
    heading.style.font.color.rgb = RGBColor(0, 51, 102) # Dark Blue
    return heading

def add_image(doc, filename, caption, width=5.5):
    path = FIG / filename
    if path.exists():
        doc.add_picture(str(path), width=Inches(width))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p = doc.add_paragraph(caption)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.style.font.italic = True
        p.style.font.size = Pt(9)
    else:
        doc.add_paragraph(f"[Missing Image: {filename}]")

def main():
    doc = Document()

    # --- COVER PAGE ---
    title = doc.add_heading("Fraud-Risk Intelligence Pipeline", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph("\n\n")
    p = doc.add_paragraph("Business Strategy & Model Governance Report")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.style.font.size = Pt(16)
    
    doc.add_paragraph("\n\n\n\n")
    
    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run("Prepared for: ").bold = True
    info.add_run("Gamage Recruiters\n")
    info.add_run("Prepared by: ").bold = True
    info.add_run("Wijerathna D G K I\n")
    info.add_run("Date: ").bold = True
    info.add_run("October 2023\n")
    
    doc.add_page_break()

    # --- 1. EXECUTIVE SUMMARY ---
    add_heading(doc, "1. Executive Summary", 1)
    doc.add_paragraph(
        "This report outlines the development of a production-oriented fraud-risk intelligence pipeline. "
        "The objective was to detect suspicious transactions while strictly controlling false positives. "
        "Given a dataset with a 12.1% fraud rate and weak individual feature signal, we built a calibrated "
        "Logistic Regression model. Because the unconstrained mathematical optimum demanded a 98% manual "
        "review rate (operationally impossible), we designed a constrained decision policy (Approve/Block). "
        "Under hypothetical costs (FN=$500, Review=$15, Block=$50), our optimized policy costs $26,200 on the "
        "test set, saving $9,900 versus the naive 0.5 threshold."
    )

    # --- 2. DATA FORENSICS ---
    add_heading(doc, "2. Data Forensics & Quality Engineering", 1)
    doc.add_paragraph(
        "The raw dataset contained 2,999 rows. One malformed row (TXN5002485) was removed during ingestion, "
        "leaving 2,998 valid transactions. No duplicates were found. All impossible-value checks (e.g., negative amounts) passed. "
        "Extreme values (e.g., 63 outliers in transaction_amount) were retained to ensure the model learns to handle them, in strict adherence to the brief's rule: 'Do not remove difficult observations merely to improve metrics'."
    )
    
    add_heading(doc, "2.1 Missingness as Predictive Signal", 2)
    doc.add_paragraph(
        "A critical finding was that missingness is Not Missing At Random (NMAR). "
        "Transactions missing 'merchant_risk_score' had a fraud rate of 3.3% (vs 12.3% when present). "
        "Conversely, missing 'authentication_level' correlated with a 19.0% fraud rate. "
        "We captured this using explicit 'was_missing_*' indicator features inside the pipeline."
    )

    # --- 3. ADVANCED EDA ---
    add_heading(doc, "3. Advanced Exploratory Analysis", 1)
    doc.add_paragraph(
        "We produced over 15 visualizations. Two key insights drove our feature engineering: "
        "Fraud peaks in the early morning hours (4 AM - 6 AM) when legitimate users are asleep. "
        "Additionally, channel risk is highly stratified, with E-commerce showing the highest fraud rate."
    )
    
    add_image(doc, "06_fraud_by_hour.png", "Figure 1: Fraud rate peaks significantly in the early morning hours.")
    add_image(doc, "01_fraud_by_channel.png", "Figure 2: E-commerce has the highest fraud rate among channels.")
    add_image(doc, "12_interaction_amount_x_merchant_risk.png", "Figure 3: Interaction - High amount + High merchant risk yields a 20.3% fraud rate.")
    add_image(doc, "15_correlation_heatmap.png", "Figure 4: Correlation matrix confirms weak linear signal, motivating nonlinear interactions.")

    doc.add_page_break()

    # --- 4. FEATURE ENGINEERING & MODELING ---
    add_heading(doc, "4. Feature Engineering & Advanced Models", 1)
    doc.add_paragraph(
        "We engineered 12 defensible features, including velocity ratios (transactions_last_1h / transactions_last_24h), "
        "behavioral deviations (amount_per_account_day), and interaction terms (amount_x_merchant_risk). "
        "All 12 features are strictly available at prediction time (no future leakage)."
    )
    doc.add_paragraph(
        "We compared Logistic Regression, Random Forest, and HistGradientBoosting using 5-fold Stratified Cross-Validation. "
        "Surprisingly, the simplest model (Logistic Regression) outperformed the complex tree models (ROC-AUC 0.692 vs 0.591). "
        "HistGradientBoosting memorized noise and failed to generalize. Engineered features provided a marginal PR-AUC boost (0.2519 to 0.2574)."
    )

    # --- 5. CALIBRATION & IMBALANCE ---
    add_heading(doc, "5. Probability Calibration & Imbalanced Learning", 1)
    doc.add_paragraph(
        "Because we used class weighting (which beat random undersampling), the model artificially inflated probabilities. "
        "We applied Platt Scaling (Sigmoid), which halved the Brier Score from 0.2148 to 0.1027. This ensures that when the "
        "model predicts a 60% chance of fraud, it is actually 60%, making financial cost calculations reliable."
    )
    add_image(doc, "16_calibration_curves.png", "Figure 5: Platt Scaling aligns predicted probabilities with actual fraud rates.")

    # --- 6. COST-SENSITIVE OPERATIONS ---
    add_heading(doc, "6. Fraud Operations Simulation & Cost Sensitivity", 1)
    doc.add_paragraph(
        "Using the calibrated probabilities, we simulated operational policies under our base cost matrix (FN=$500, Review=$15, Block=$50)."
    )
    
    # Add Table (Kept tight to avoid page breaks)
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Light Shading Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Policy'
    hdr_cells[1].text = 'Total Cost'
    hdr_cells[2].text = 'Volumes'
    hdr_cells[3].text = 'Fraud Capture'
    
    rows = [
        ("Naive (0.5)", "$36,100", "3 Blocked, 597 Approved", "1 / 73"),
        ("Math Optimal", "$8,820", "1 Blocked, 588 Reviewed", "73 / 73 (98% review rate)"),
        ("Operational (Capped)", "$26,200", "146 Blocked, 454 Approved", "32 / 73 (Max 15% review)")
    ]
    for row in rows:
        cells = table.add_row().cells
        cells[0].text = row[0]
        cells[1].text = row[1]
        cells[2].text = row[2]
        cells[3].text = row[3]

    doc.add_paragraph("\nConclusion: The unconstrained math optimum is impossible. The operational policy (Block >= 0.15) is the only deployable solution, saving $9,900 over the naive baseline.")
    
    add_heading(doc, "6.1 Cost Sensitivity Analysis", 2)
    doc.add_paragraph(
        "Business costs are assumptions. We performed a sensitivity analysis: if the cost of a missed fraud (FN) doubles from $500 to $1,000, the operational policy would aggressively lower the block threshold to 0.10 to capture more fraud. "
        "Conversely, if the false-positive friction cost (Block) rises to $150, the model would be forced to raise the block threshold to 0.25 to protect customer experience. "
        "This demonstrates that the chosen thresholds are highly sensitive to external business realities and must be reviewed quarterly."
    )

    doc.add_page_break()

    # --- 7. EXPLAINABILITY ---
    add_heading(doc, "7. Explainable AI (SHAP)", 1)
    doc.add_paragraph(
        "We used SHAP (SHapley Additive exPlanations) to interpret the model. The global summary reveals that "
        "'transactions_last_1h' and 'is_night_txn' push risk up. However, we identified a counterintuitive finding: "
        "the 'velocity_ratio_1h_24h' pushes risk DOWN when high, because a ratio near 1.0 indicates steady, normal activity rather than a sudden burst."
    )
    add_image(doc, "17_shap_global_summary.png", "Figure 6: SHAP Global Summary Plot.", width=5.0)
    add_image(doc, "18_shap_local_1_txn.png", "Figure 7: Local Waterfall Plot explaining the highest-risk transaction.", width=5.0)

    # --- 8. ROBUSTNESS ---
    add_heading(doc, "8. Robustness & Stress Testing", 1)
    doc.add_paragraph(
        "We simulated distribution shifts to evaluate model degradation. "
        "An economic inflation shift (+50% amounts) and a velocity attack (2x transactions) actually decreased total cost because the model aggressively blocked more transactions. "
        "An authentication service outage (20% missing) barely affected ROC-AUC (0.703) thanks to our missingness indicator features. "
        "A channel shift (E-commerce to POS) caused the largest degradation (Cost: $26,900), proving the model is sensitive to channel mix."
    )

    # --- 9. GOVERNANCE ---
    add_heading(doc, "9. Model Governance & Monitoring", 1)
    doc.add_paragraph("To ensure safe production deployment, the following 5 metrics must be monitored weekly:")
    metrics = [
        "PR-AUC Drift: Retrain if it drops >10% below 0.2574.",
        "Population Stability Index (PSI): Monitor transaction_amount and transactions_last_1h. Alert if PSI > 0.2.",
        "Review Backlog Volume: If review volume exceeds 15%, adjust thresholds.",
        "False Positive Burden: If legitimate customer complaints exceed 5%, temporarily disable the model.",
        "Brier Score Drift: If it rises >0.15, recalibrate Platt scaling."
    ]
    for m in metrics:
        doc.add_paragraph(m, style='List Bullet')

    doc.save("07_business_fraud_strategy.docx")
    print("Successfully generated 07_business_fraud_strategy.docx!")

if __name__ == "__main__":
    main()