# Fraud-Risk Intelligence Pipeline

## Environment
- Python 3.9+
- Run `pip install -r requirements.txt` to install dependencies.
- `requirements.txt` should contain: pandas, numpy, scikit-learn, imbalanced-learn, shap, matplotlib, seaborn, joblib.

## Execution Order
1. `python src/clean_data.py`
2. `python src/eda.py` (Generates 15 plots in `reports/figures/`)
3. `python src/feature_engineering.py`
4. `python src/modeling.py` (5-fold CV)
5. `python src/calibration.py`
6. `python src/cost_optimization.py`
7. `python src/robustness.py`
8. `python src/shap_explain.py`

## Assumptions
- Data has no transaction timestamp. Random stratified split used.
- Base file was missing TXN5002485 (dropped during ingestion).
- Costs: FN=$500, Review=$15, Block=$50.

## Random Seed
- `random_state=42` used across all splits and models.