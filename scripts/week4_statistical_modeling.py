"""
Week 4 - Statistical Analysis & Predictive Modeling for Environmental Impact
Junior Data Analyst - Green Jobs & Sustainability Internship

Two models, both fit on the simulated data produced by
week3_eda_visualization.py:

  Model 1: OLS trend regression forecasting India's renewable capacity
           2025-2027 from its 2015-2024 history.
  Model 2: Multiple linear regression predicting each Indian state's
           SDG India Index score from renewable capacity + green-job
           workforce share.

Run from the repository root, after week3_eda_visualization.py has been
run at least once (this script reads data/global_simulated.csv and
data/state_simulated.csv):

    python scripts/week4_statistical_modeling.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error

sns.set_theme(style="whitegrid", font_scale=1.05)
ACCENT = "#1F6B8E"
DARK = "#0A1628"
ORANGE = "#C9762B"

DATA_DIR = "data"
CHARTS_DIR = "outputs/week4_charts"
RESULTS_DIR = "outputs/results"
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

global_df = pd.read_csv(f"{DATA_DIR}/global_simulated.csv")
state_df = pd.read_csv(f"{DATA_DIR}/state_simulated.csv")

# =================================================================
# MODEL 1: Time-series forecast - India renewable capacity, 2025-2027
# =================================================================
india = global_df[global_df.country == "India"].copy()
X = sm.add_constant(india["year"])
y = india["renewable_capacity_gw"]
lin_model = sm.OLS(y, X).fit()
print(lin_model.summary())

future_years = np.array([2025, 2026, 2027])
Xf = sm.add_constant(future_years, has_constant="add")
pred = lin_model.get_prediction(Xf)
pred_summary = pred.summary_frame(alpha=0.20)  # 80% interval
pred_summary["year"] = future_years
pred_summary.to_csv(f"{RESULTS_DIR}/india_forecast.csv", index=False)
print(pred_summary)

y_hat = lin_model.predict(X)
r2_lin = r2_score(y, y_hat)
rmse_lin = np.sqrt(mean_squared_error(y, y_hat))
print("Linear trend R2:", round(r2_lin, 3), "RMSE:", round(rmse_lin, 2))

plt.figure(figsize=(8, 5.5))
plt.plot(india["year"], india["renewable_capacity_gw"], "o-", color=ACCENT, label="Historical (simulated)", linewidth=2)
plt.plot(pred_summary["year"], pred_summary["mean"], "o--", color=ORANGE, label="Forecast (linear trend)", linewidth=2)
plt.fill_between(pred_summary["year"], pred_summary["mean_ci_lower"], pred_summary["mean_ci_upper"],
                  color=ORANGE, alpha=0.2, label="80% prediction interval")
plt.title("India: Renewable Capacity Forecast, 2025\u20132027", fontsize=13, color=DARK, weight="bold")
plt.xlabel("Year"); plt.ylabel("Installed Renewable Capacity (GW)")
plt.legend(frameon=False, fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/01_india_forecast.png", dpi=150, bbox_inches="tight")
plt.close()

# =================================================================
# MODEL 2: Multiple linear regression - predicting SDG India Index
# score from renewable capacity + green job share (state-level)
# =================================================================
Xs = state_df[["renewable_capacity_mw", "green_job_share_pct"]]
Xs = sm.add_constant(Xs)
ys = state_df["sdg_index_score"]

mlr_model = sm.OLS(ys, Xs).fit()
print(mlr_model.summary())

state_df["predicted_sdg"] = mlr_model.predict(Xs)
state_df["residual"] = state_df["sdg_index_score"] - state_df["predicted_sdg"]
state_df.to_csv(f"{RESULTS_DIR}/state_model_results.csv", index=False)

r2_mlr = mlr_model.rsquared
adj_r2_mlr = mlr_model.rsquared_adj
rmse_mlr = np.sqrt(mean_squared_error(ys, state_df["predicted_sdg"]))
print("MLR R2:", round(r2_mlr, 3), "Adj R2:", round(adj_r2_mlr, 3), "RMSE:", round(rmse_mlr, 2))

plt.figure(figsize=(6.5, 6))
plt.scatter(state_df["sdg_index_score"], state_df["predicted_sdg"], color=ACCENT, s=70, alpha=0.85)
lims = [50, 82]
plt.plot(lims, lims, "--", color=ORANGE, linewidth=1.5)
for _, row in state_df.iterrows():
    plt.annotate(row["state"], (row["sdg_index_score"], row["predicted_sdg"]), fontsize=7.5,
                 xytext=(4, 3), textcoords="offset points", color="#333333")
plt.xlabel("Actual SDG India Index Score"); plt.ylabel("Predicted SDG India Index Score")
plt.title(f"Actual vs. Predicted SDG Index Score (R\u00b2 = {r2_mlr:.2f})", fontsize=12.5, color=DARK, weight="bold")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/02_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7.5, 4.5))
order = state_df.sort_values("residual")
plt.barh(order["state"], order["residual"], color=[ACCENT if v >= 0 else ORANGE for v in order["residual"]])
plt.axvline(0, color="#333333", linewidth=0.8)
plt.title("Model Residuals by State (Actual \u2212 Predicted SDG Score)", fontsize=12.5, color=DARK, weight="bold")
plt.xlabel("Residual")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/03_residuals.png", dpi=150, bbox_inches="tight")
plt.close()

coef_table = pd.DataFrame({
    "variable": mlr_model.params.index,
    "coefficient": mlr_model.params.values,
    "std_err": mlr_model.bse.values,
    "p_value": mlr_model.pvalues.values,
})
coef_table.to_csv(f"{RESULTS_DIR}/mlr_coefficients.csv", index=False)
print(coef_table)

print("\nDone. Charts saved to", CHARTS_DIR, "| Results saved to", RESULTS_DIR)
