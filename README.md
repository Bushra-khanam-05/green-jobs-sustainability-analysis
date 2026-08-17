# Green Jobs & Sustainability Analysis

Code accompanying the **Junior Data Analyst — Green Jobs & Sustainability Internship**, covering exploratory data analysis, visualization, and statistical/predictive modeling of renewable energy capacity, green employment, and sustainability performance trends.

## About the data

No proprietary databases were available for this internship, so the datasets used here are **simulated** — built to follow the real, publicly reported growth patterns and rankings from [IRENASTAT](https://www.irena.org/Data), [ILOSTAT](https://ilostat.ilo.org/), and the [NITI Aayog SDG India Index](https://sdgindiaindex.niti.gov.in/) (for example, China and the U.S. leading in absolute renewable capacity, Vietnam and South Africa showing the fastest recent growth off a smaller base, and Gujarat/Rajasthan/Tamil Nadu leading Indian states in installed capacity). The specific numeric values are illustrative, not the literal source figures — this is noted throughout the accompanying report.

The data collection and cleaning **methodology** designed for the real source data (source selection criteria, extraction methods, a seven-step cleaning process) is documented separately in the Week 2 report, since actual API/bulk-download access to IRENASTAT, ILOSTAT, and NITI Aayog was outside the scope of this internship's tooling.

## Repository structure

```
├── data/
│   ├── global_simulated.csv      # 12 countries x 10 years (2015-2024): renewable capacity & green employment
│   └── state_simulated.csv       # 12 Indian states: capacity, SDG India Index score, green-job share
├── scripts/
│   ├── week3_eda_visualization.py    # Simulates data, runs EDA, generates 6 charts
│   └── week4_statistical_modeling.py # OLS trend forecast + multiple linear regression
├── outputs/
│   ├── week3_charts/              # EDA visualizations (trend, scatter, bar, pie, heatmap)
│   ├── week4_charts/              # Forecast & regression diagnostic charts
│   └── results/                   # CSV outputs: descriptive stats, correlations, model coefficients, forecasts
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
python scripts/week3_eda_visualization.py    # generates data/ and outputs/week3_charts/
python scripts/week4_statistical_modeling.py # reads data/, generates outputs/week4_charts/
```

Both scripts are idempotent and can be re-run from a clean checkout — `week3_eda_visualization.py` regenerates the simulated CSVs in `data/` before `week4_statistical_modeling.py` reads them.

## Methods summary

**Week 3 — EDA & Visualization**
- Descriptive statistics (mean, median, std. dev.) across all core KPIs
- Correlation analysis between renewable capacity, green employment, SDG Index score, and green-job share
- Six visualizations: multi-country trend line, capacity-vs-employment scatter with regression fit, growth-rate bar chart, state capacity bar chart, sector-wise pie chart, correlation heatmap

**Week 4 — Statistical Modeling**
- **Model 1:** OLS linear trend regression forecasting India's renewable capacity for 2025–2027, with an 80% prediction interval (R² = 0.967)
- **Model 2:** Multiple linear regression predicting each state's SDG India Index score from renewable capacity and green-job workforce share (R² = 0.967), with residual analysis to flag states the model over/under-predicts

Full narrative interpretation, limitations, and policy implications are in the corresponding Word report deliverables (not included in this code repository).

## Key finding

Renewable capacity correlates strongly with green-job workforce share (r = 0.85) but only moderately with overall SDG Index performance (r = 0.30) — energy infrastructure alone does not fully explain a state's sustainability outcomes. In the multiple regression, capacity's effect on the SDG score is largely mediated through green-job creation rather than acting as an independent driver.

## Tech stack

Python · pandas · NumPy · statsmodels · scikit-learn · matplotlib · seaborn

---
*Bushra Khanam — Junior Data Analyst Internship, Green Jobs & Sustainability track*
