# Final Capstone: Day-Ahead Hourly Demand Forecasting for a City Bike-Share System

Task 6 of the AI/ML Research Specialist virtual internship (Yuva Intern, Thiruvananthapuram).
End-to-end project on real data: problem definition, collection, cleaning, feature engineering, EDA, statistical and ML modelling, evaluation, interpretation, uncertainty, visualisation and recommendations.

**Data:** UCI Bike Sharing Dataset (Capital Bikeshare, Washington DC, 2011-2012; 17,379 hourly records; CC BY 4.0; Fanaee-T and Gama 2013, doi:10.24432/C5W894). `src/download_data.py` verifies two independent copies are identical.

## Objectives -> where covered
| # | Objective | Where |
|---|-----------|-------|
| 1 | Define the problem | Notebook 1 section 1; report section 2 |
| 2 | Collect and prepare data | Notebook 1 sections 2-5; `src/bikeshare/data.py`, `features.py` |
| 3 | Apply data-science techniques | Notebook 2 sections 6-7, 9-10 (baselines, NB GLM, random forest, LightGBM, quantile models) |
| 4 | Evaluate and interpret | Notebook 2 sections 7-8 (WAPE, Diebold-Mariano, SHAP, GLM rate ratios, ablation) |
| 5 | Visualisations and recommendations | `outputs/figures/` (17 figures); report sections 9-10 |
| 6 | Final report and presentation | `docs/Task6_Capstone_Report.docx`; presentation (10 slides) |

## Layout
```
src/bikeshare/  data.py, features.py, metrics.py, modeling.py      tests/  22 pytest tests (incl. leakage tests)
notebooks/      01_Data_Preparation_EDA.ipynb, 02_Modeling_Evaluation.ipynb (executed)
data/           hour.csv (raw), SOURCE.md, processed/hourly_clean.csv
outputs/        figures/, tables/, models/, eda_metrics.json, model_metrics.json
docs/           Task6_Capstone_Report.docx
```

## Run it
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
python src/download_data.py && pytest -q               # 22 passed
jupyter notebook notebooks/                            # run both notebooks top to bottom
```

## Headline results (chronological test quarter, Oct-Dec 2012, used once)
| Model | WAPE |
|---|---|
| LightGBM (recommended) | 17.9% |
| Random forest (pre-declared champion) | 22.3% |
| Negative-binomial GLM | 23.2% |
| Best naive baseline (profile x recent level) | 28.8% |

- LightGBM is 37.8% better than the best baseline (Diebold-Mariano p < 0.001 against all other models).
- An hour-ahead model reaches 12.4% WAPE (31% lower error).
- Raw 80% intervals covered 68.8% of hours; conformal recalibration (validation-calibrated) gives 81.4%.

## Honest notes
- My pre-declared champion rule (lowest validation WAPE) picked the random forest in a statistical tie with LightGBM and had no tie-break. LightGBM was analysed afterwards because it was clearly better on test; that choice used test information, so its numbers are mildly optimistic. The random forest missed success criterion S1 (22.7% vs 25% required).
- Weather is observed, not forecast (stress-tested with synthetic noise). Hurricane Sandy, Christmas week and the day after Thanksgiving are known failure modes.
