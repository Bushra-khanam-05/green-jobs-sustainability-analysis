"""
Week 3 - Exploratory Data Analysis & Visualization for Sustainability Trends
Junior Data Analyst - Green Jobs & Sustainability Internship

Simulates two datasets shaped to match real, publicly reported patterns from
IRENA, ILOSTAT, and the NITI Aayog SDG India Index (see README for details on
why simulated data was used), then runs descriptive statistics, correlation
analysis, and generates six visualizations.

Run from the repository root:
    python scripts/week3_eda_visualization.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

rng = np.random.default_rng(42)
sns.set_theme(style="whitegrid", font_scale=1.05)
ACCENT = "#1F6B8E"
DARK = "#0A1628"
PALETTE = ["#1F6B8E", "#3E92B5", "#7FB8D6", "#0A1628", "#C9762B", "#8AA29E"]

DATA_DIR = "data"
CHARTS_DIR = "outputs/week3_charts"
RESULTS_DIR = "outputs/results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. SIMULATED GLOBAL DATASET: renewable capacity vs green employment
# Illustrative country-level figures built to follow realistic aggregate
# growth patterns reported by IRENA/ILOSTAT -- NOT their actual figures.
# ---------------------------------------------------------------
countries = ["India", "China", "United States", "Germany", "Brazil",
             "Japan", "United Kingdom", "Vietnam", "South Africa", "Spain",
             "Australia", "Mexico"]
base_capacity_2015 = {
    "India": 36, "China": 480, "United States": 190, "Germany": 92,
    "Brazil": 122, "Japan": 88, "United Kingdom": 34, "Vietnam": 5,
    "South Africa": 4, "Spain": 47, "Australia": 15, "Mexico": 22
}
growth_rate = {
    "India": 0.145, "China": 0.15, "United States": 0.09, "Germany": 0.06,
    "Brazil": 0.08, "Japan": 0.05, "United Kingdom": 0.08, "Vietnam": 0.32,
    "South Africa": 0.18, "Spain": 0.09, "Australia": 0.14, "Mexico": 0.10
}

years = list(range(2015, 2025))
rows = []
for c in countries:
    cap = base_capacity_2015[c]
    g = growth_rate[c]
    for y in years:
        capacity = cap * ((1 + g) ** (y - 2015)) * (1 + rng.normal(0, 0.02))
        jobs_per_mw = rng.uniform(0.55, 0.95)
        employment_k = capacity * jobs_per_mw * (1 + rng.normal(0, 0.05))
        rows.append([c, y, round(capacity, 1), round(employment_k, 1)])

global_df = pd.DataFrame(rows, columns=["country", "year", "renewable_capacity_gw", "green_employment_thousands"])
global_df.to_csv(f"{DATA_DIR}/global_simulated.csv", index=False)

# ---------------------------------------------------------------
# 2. SIMULATED INDIA STATE-LEVEL DATASET (2024 snapshot)
# Shaped to reflect known real-world rankings (MNRE capacity leaders,
# NITI Aayog SDG India Index leaders) -- values are illustrative.
# ---------------------------------------------------------------
states_data = [
    ("Gujarat", 25100, 72, 8.4), ("Rajasthan", 24800, 68, 7.9),
    ("Tamil Nadu", 21300, 75, 8.1), ("Karnataka", 13800, 74, 7.2),
    ("Maharashtra", 12200, 70, 6.8), ("Andhra Pradesh", 10400, 69, 6.1),
    ("Madhya Pradesh", 8300, 63, 5.2), ("Kerala", 3200, 78, 6.5),
    ("Himachal Pradesh", 2600, 77, 5.9), ("Uttar Pradesh", 5100, 60, 4.3),
    ("Punjab", 2200, 66, 4.6), ("Bihar", 900, 55, 2.8),
]
state_df = pd.DataFrame(states_data, columns=["state", "renewable_capacity_mw", "sdg_index_score", "green_job_share_pct"])
state_df.to_csv(f"{DATA_DIR}/state_simulated.csv", index=False)

# ---------------------------------------------------------------
# 3. SIMULATED SECTOR-WISE GREEN JOB DISTRIBUTION (India, 2024)
# ---------------------------------------------------------------
sector_data = [
    ("Renewable Energy (Solar/Wind)", 34), ("Sustainable Agriculture", 21),
    ("Waste Management & Recycling", 16), ("Green Construction", 15),
    ("EV & Clean Transport", 9), ("Water & Resource Management", 5),
]
sector_df = pd.DataFrame(sector_data, columns=["sector", "share_pct"])

# =================================================================
# EDA: descriptive statistics
# =================================================================
desc_global = global_df.groupby("year")[["renewable_capacity_gw", "green_employment_thousands"]].sum().round(1)
desc_global.to_csv(f"{RESULTS_DIR}/desc_global_by_year.csv")

corr = global_df[["renewable_capacity_gw", "green_employment_thousands"]].corr().iloc[0, 1]
print("Correlation (capacity vs employment):", round(corr, 3))

state_corr = state_df[["renewable_capacity_mw", "sdg_index_score", "green_job_share_pct"]].corr()
state_corr.to_csv(f"{RESULTS_DIR}/state_correlation_matrix.csv")
print(state_corr)

summary_stats = global_df.groupby("country").agg(
    capacity_2024=("renewable_capacity_gw", lambda s: s[global_df.loc[s.index, "year"] == 2024].values[0]),
    capacity_2015=("renewable_capacity_gw", lambda s: s[global_df.loc[s.index, "year"] == 2015].values[0]),
).reset_index()
summary_stats["growth_pct"] = ((summary_stats["capacity_2024"] / summary_stats["capacity_2015"]) - 1) * 100
summary_stats = summary_stats.sort_values("growth_pct", ascending=False)
summary_stats.to_csv(f"{RESULTS_DIR}/growth_summary.csv", index=False)
print(summary_stats)

# =================================================================
# CHARTS
# =================================================================

top5 = summary_stats.nlargest(5, "capacity_2024")["country"].tolist()
plt.figure(figsize=(8, 5))
for i, c in enumerate(top5):
    sub = global_df[global_df.country == c]
    plt.plot(sub.year, sub.renewable_capacity_gw, marker="o", label=c, color=PALETTE[i % len(PALETTE)], linewidth=2)
plt.title("Renewable Capacity Growth, 2015\u20132024 (Top 5 Countries)", fontsize=13, color=DARK, weight="bold")
plt.xlabel("Year"); plt.ylabel("Installed Renewable Capacity (GW)")
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/01_capacity_trend.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 5.5))
sns.regplot(data=global_df, x="renewable_capacity_gw", y="green_employment_thousands",
            scatter_kws={"alpha": 0.5, "color": ACCENT}, line_kws={"color": "#C9762B"})
plt.title(f"Renewable Capacity vs. Green Employment (r = {corr:.2f})", fontsize=13, color=DARK, weight="bold")
plt.xlabel("Installed Renewable Capacity (GW)"); plt.ylabel("Green Employment (thousands)")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/02_capacity_vs_employment.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5.5))
colors = [ACCENT if v >= 0 else "#C9762B" for v in summary_stats["growth_pct"]]
plt.barh(summary_stats["country"], summary_stats["growth_pct"], color=colors)
plt.gca().invert_yaxis()
plt.title("Renewable Capacity Growth, 2015\u20132024 (%)", fontsize=13, color=DARK, weight="bold")
plt.xlabel("Growth (%)")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/03_growth_by_country.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5.5))
state_sorted = state_df.sort_values("renewable_capacity_mw", ascending=True)
plt.barh(state_sorted["state"], state_sorted["renewable_capacity_mw"], color=ACCENT)
plt.title("Installed Renewable Capacity by Indian State (2024, Simulated)", fontsize=12.5, color=DARK, weight="bold")
plt.xlabel("Capacity (MW)")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/04_state_capacity.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 7))
plt.pie(sector_df["share_pct"], labels=sector_df["sector"], autopct="%1.0f%%",
        colors=PALETTE, startangle=90, textprops={"fontsize": 9}, radius=0.85)
plt.title("Simulated Sector-wise Distribution of Green Jobs, India (2024)", fontsize=12, color=DARK, weight="bold")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/05_sector_pie.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(5.5, 4.5))
sns.heatmap(state_corr, annot=True, cmap="Blues", vmin=-1, vmax=1, fmt=".2f", cbar_kws={"label": "Correlation"})
plt.title("Correlation Between State-Level KPIs", fontsize=12, color=DARK, weight="bold")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/06_state_corr_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nAll charts saved to", CHARTS_DIR)
print("Global capacity-employment correlation:", round(corr, 3))
print("\nState correlation matrix:\n", state_corr.round(2))
