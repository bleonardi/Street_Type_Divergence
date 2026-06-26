"""
Year-by-year spatial mimetic spillover in cul-de-sac adoption.

For each construction year Y (1940–2020), restrict to block groups whose
median_year_built == Y (±2 year window), build a KNN spatial weight matrix,
compute neighbor_culdesac_lag, and run:

  culdesac_share ~ neighbor_lag + state_FE   (OLS, robust SE)

Report β_neighbor by year. This is the continuous-year version of the
era-bucketed analysis in spatial_lag_mimesis.png.

Output: data/processed/mimetic_beta_by_year_continuous.csv
        figures/mimetic_beta_continuous.png
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.neighbors import BallTree
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

PROC = Path(__file__).parent.parent / "data" / "processed"
FIG  = Path(__file__).parent.parent / "figures"

# ── Load block-group data ─────────────────────────────────────────────────────
bg  = pd.read_parquet(PROC / "bg_suffix_composition.parquet")
era = pd.read_parquet(PROC / "block_group_era.parquet",
                      columns=["GEOID","median_year_built"])

# Merge continuous year onto suffix data
# bg already has centroid_lat/lon; use era's year
df = bg.merge(era[["GEOID","median_year_built"]], on="GEOID", how="inner")
df = df.dropna(subset=["culdesac_share","median_year_built","centroid_lat","centroid_lon"])
df["year"] = df["median_year_built"].round().astype(int)
df = df[(df["year"] >= 1940) & (df["year"] <= 2020)].copy()

print(f"Block groups: {len(df):,}  year range: {df['year'].min()}–{df['year'].max()}")

# ── Build global KNN spatial weight (k=8) ────────────────────────────────────
# Use BallTree on all block groups; precompute neighbors once
print("Building global KNN (k=8)...")
coords_rad = np.radians(df[["centroid_lat","centroid_lon"]].values)
tree = BallTree(coords_rad, metric="haversine")
K = 8
dists, idxs = tree.query(coords_rad, k=K+1)   # +1 because self is included

# For each BG, avg culdesac_share of k nearest neighbors (excluding self)
culdesac = df["culdesac_share"].values
neighbor_lag = np.array([
    culdesac[idxs[i, 1:]].mean()   # skip index 0 (self)
    for i in range(len(df))
])
df["neighbor_lag"] = neighbor_lag

# State dummies for FE
states = pd.get_dummies(df["state"], drop_first=True, dtype=float)

# ── Rolling-window OLS by year ────────────────────────────────────────────────
WINDOW = 3   # ±window years (so 7-year window total)
MIN_N  = 200

records = []
years = sorted(df["year"].unique())

print(f"Running OLS for each year (±{WINDOW}-year window)...")
for y in years:
    sub = df[(df["year"] >= y - WINDOW) & (df["year"] <= y + WINDOW)].copy()
    if len(sub) < MIN_N:
        continue

    X_base = sub[["neighbor_lag"]].copy()
    X_base = pd.concat([X_base, states.reindex(sub.index).fillna(0)], axis=1)
    X = sm.add_constant(X_base)
    y_vec = sub["culdesac_share"]

    try:
        res = sm.OLS(y_vec, X).fit(cov_type="HC3")
        records.append({
            "year":          y,
            "beta_neighbor": res.params["neighbor_lag"],
            "se":            res.HC3_se["neighbor_lag"],
            "p":             res.pvalues["neighbor_lag"],
            "r2":            res.rsquared,
            "n":             len(sub),
        })
    except Exception:
        pass

out = pd.DataFrame(records)
out.to_csv(PROC / "mimetic_beta_by_year_continuous.csv", index=False)
print(f"\nEstimated β for {len(out)} years  (window ±{WINDOW})")
print(out[["year","beta_neighbor","se","p","n"]].to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
ERA_BANDS = [
    (1940, 1959, "#d0e8ff", "postwar boom"),
    (1960, 1964, "#c8f0c8", "early 1960s"),
    (1965, 1979, "#ffe8b0", "cul-de-sac\npeak"),
    (1980, 1999, "#ffd0d0", "late suburban"),
    (2000, 2020, "#e8e0ff", "new urbanism\nbacklash"),
]

fig, ax = plt.subplots(figsize=(11, 5))

for x0, x1, color, label in ERA_BANDS:
    ax.axvspan(x0, x1, color=color, alpha=0.55, zorder=0)
    ax.text((x0+x1)/2, 0.03, label, ha="center", va="bottom",
            fontsize=7.5, color="#555", zorder=1)

ax.fill_between(
    out["year"],
    out["beta_neighbor"] - 1.96*out["se"],
    out["beta_neighbor"] + 1.96*out["se"],
    alpha=0.18, color="#1565c0", zorder=2,
)
ax.plot(out["year"], out["beta_neighbor"], color="#1565c0", lw=2.2, zorder=3)

# Annotate specific years
for yr, label in [(1950, None), (1975, None), (2010, None)]:
    row = out[out["year"] == yr]
    if not row.empty:
        b = row.iloc[0]["beta_neighbor"]
        ax.annotate(f"β={b:.2f}", xy=(yr, b), xytext=(yr, b+0.05),
                    ha="center", fontsize=9, color="#1a237e",
                    arrowprops=dict(arrowstyle="-", color="#1a237e", lw=0.8))

ax.set_xlabel("Block group median construction year", fontsize=11)
ax.set_ylabel("β  (mimetic spillover from neighbors)", fontsize=11)
ax.set_title(
    "Mimetic spillover in cul-de-sac adoption — year by year\n"
    f"OLS with state FE + HC3 robust SE  |  ±{WINDOW}-year rolling window  "
    f"(n per year ≈ {int(out['n'].median()):,})",
    fontsize=11,
)
ax.set_xlim(out["year"].min()-1, out["year"].max()+1)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.25)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
out_path = FIG / "mimetic_beta_continuous.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out_path}")
