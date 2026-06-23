"""
Core analysis: for each suffix family, compute built-form attribute distributions
and a divergence score vs. a reference "archetype" profile.

Divergence score: how far the median/modal built-form of a suffix is from its
expected archetype (defined from pre-1950 urbanist/traffic-engineering norms).
"""

import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

DATA_PROC = Path(__file__).parent.parent / "data" / "processed"
FIG_DIR   = DATA_PROC.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Archetype profiles — expected values based on historic typological norms.
# highway_rank: lower = more local/residential, higher = more arterial/throughput
# Reference: ITE functional classification + traditional urban design vocabulary
ARCHETYPES = {
    # hw=highway rank, lanes, speed mph, sidewalk, intx_density (intersections/km)
    # intx_density: high = grid-like/urban, low = cul-de-sac/rural
    "Court":      dict(hw=1,    lanes=1.5, speed=20,  sidewalk=True,  intx=2),
    "Place":      dict(hw=1,    lanes=1.5, speed=20,  sidewalk=True,  intx=2),
    "Alley":      dict(hw=1,    lanes=1,   speed=15,  sidewalk=False, intx=8),
    "Lane":       dict(hw=1,    lanes=1.5, speed=25,  sidewalk=True,  intx=3),
    "Row":        dict(hw=1,    lanes=1,   speed=15,  sidewalk=False, intx=4),
    "Walk":       dict(hw=1,    lanes=1,   speed=10,  sidewalk=True,  intx=6),
    "Path":       dict(hw=1,    lanes=1,   speed=10,  sidewalk=True,  intx=4),
    "Terrace":    dict(hw=2,    lanes=2,   speed=25,  sidewalk=True,  intx=5),
    "Street":     dict(hw=2,    lanes=2,   speed=30,  sidewalk=True,  intx=8),
    "Avenue":     dict(hw=2,    lanes=2,   speed=30,  sidewalk=True,  intx=8),
    "Drive":      dict(hw=2,    lanes=2,   speed=30,  sidewalk=True,  intx=4),
    "Circle":     dict(hw=2,    lanes=2,   speed=25,  sidewalk=True,  intx=3),
    "Way":        dict(hw=2,    lanes=2,   speed=25,  sidewalk=True,  intx=4),
    "Run":        dict(hw=2,    lanes=2,   speed=30,  sidewalk=False, intx=3),
    "Road":       dict(hw=3,    lanes=2,   speed=35,  sidewalk=False, intx=3),
    "Pike":       dict(hw=3,    lanes=2,   speed=40,  sidewalk=False, intx=2),
    "Turnpike":   dict(hw=3,    lanes=2,   speed=45,  sidewalk=False, intx=2),
    "Trail":      dict(hw=2,    lanes=1,   speed=25,  sidewalk=False, intx=2),
    "Boulevard":  dict(hw=4,    lanes=4,   speed=35,  sidewalk=True,  intx=6),
    "Parkway":    dict(hw=4,    lanes=4,   speed=40,  sidewalk=True,  intx=3),
    "Highway":    dict(hw=5,    lanes=4,   speed=55,  sidewalk=False, intx=1),
}

HIGHWAY_RANK = {
    "service": 1, "living_street": 1, "pedestrian": 1, "track": 1,
    "residential": 2,
    "unclassified": 2,
    "tertiary": 3, "tertiary_link": 3,
    "secondary": 4, "secondary_link": 4,
    "primary": 5, "primary_link": 5,
    "trunk": 6, "trunk_link": 6,
    "motorway": 7, "motorway_link": 7,
}


def load_data() -> pd.DataFrame:
    parquet = DATA_PROC / "streets.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Run 02_extract_streets.py first — {parquet} not found")
    df = pd.read_parquet(parquet)
    df = df[df["suffix"].notna()].copy()
    df["hw_rank"] = df["highway"].map(HIGHWAY_RANK)
    df["has_sidewalk"] = df["sidewalk"].isin(["both", "left", "right", "yes", "separate"])
    return df


def suffix_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Median built-form attributes per suffix family."""
    return (
        df.groupby("suffix").agg(
            n                   = ("osm_id", "count"),
            hw_rank             = ("hw_rank", "median"),
            lanes               = ("lanes", "median"),
            maxspeed            = ("maxspeed", "median"),
            sidewalk_pct        = ("has_sidewalk", "mean"),
            intx_density        = ("intersection_density", "median"),
            way_length_m        = ("way_length_m", "median"),
        )
        .reset_index()
    )


def divergence_score(profile: pd.Series) -> float:
    """
    Z-score style divergence vs archetype on the attributes we have.
    Returns a 0-1 normalized score: 0 = archetype, 1 = maximum divergence.
    """
    suffix = profile["suffix"]
    arch = ARCHETYPES.get(suffix)
    if arch is None:
        return np.nan

    deltas = []
    if not np.isnan(profile["hw_rank"]):
        deltas.append(abs(profile["hw_rank"] - arch["hw"]) / 6)
    if not np.isnan(profile["lanes"]):
        deltas.append(abs(profile["lanes"] - arch["lanes"]) / 5)
    if not np.isnan(profile["maxspeed"]):
        deltas.append(abs(profile["maxspeed"] - arch["speed"]) / 50)
    if not np.isnan(profile["sidewalk_pct"]):
        expected_sw = 1.0 if arch["sidewalk"] else 0.0
        deltas.append(abs(profile["sidewalk_pct"] - expected_sw))
    if not np.isnan(profile["intx_density"]):
        deltas.append(abs(profile["intx_density"] - arch["intx"]) / 15)  # max range ~15

    return np.mean(deltas) if deltas else np.nan


def plot_divergence(scores: pd.DataFrame):
    scores = scores.dropna(subset=["divergence"]).sort_values("divergence", ascending=True)
    scores = scores[scores["n"] >= 50]  # minimum sample size

    fig, ax = plt.subplots(figsize=(9, max(5, len(scores) * 0.38)))
    colors = ["#d73027" if d > 0.25 else "#4575b4" if d < 0.12 else "#fee090"
              for d in scores["divergence"]]
    bars = ax.barh(scores["suffix"], scores["divergence"], color=colors, height=0.65)
    ax.axvline(0.25, color="#d73027", lw=1, ls="--", alpha=0.5, label="high divergence threshold")
    ax.axvline(0.12, color="#4575b4", lw=1, ls="--", alpha=0.5, label="low divergence threshold")
    ax.set_xlabel("Divergence score (0 = matches archetype, 1 = max divergence)")
    ax.set_title("Street type suffix vs. built-form archetype divergence", fontsize=13, pad=10)
    ax.legend(fontsize=9)
    for bar, (_, row) in zip(bars, scores.iterrows()):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"n={row['n']:,}", va="center", fontsize=8, color="#555")
    plt.tight_layout()
    out = FIG_DIR / "divergence_scores.png"
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")
    plt.close()


def plot_attribute_distributions(df: pd.DataFrame, top_n: int = 10):
    """Speed and lane count violin plots for the top N suffixes by count."""
    top = df["suffix"].value_counts().head(top_n).index.tolist()
    sub = df[df["suffix"].isin(top) & df["maxspeed"].notna()].copy()

    order = (sub.groupby("suffix")["maxspeed"].median()
               .reindex(top).sort_values().index.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, label in zip(axes,
                               ["maxspeed", "lanes"],
                               ["Max speed (mph)", "Lanes"]):
        data = [sub.loc[sub["suffix"] == s, col].dropna().values for s in order]
        ax.violinplot(data, showmedians=True, showextrema=False)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by suffix")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Built-form attribute distributions by street type suffix", y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "attribute_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} ways with known suffix\n")

    print("Computing suffix profiles...")
    profile = suffix_profile(df)
    profile["divergence"] = profile.apply(divergence_score, axis=1)
    profile = profile.sort_values("divergence", ascending=False)

    out_csv = DATA_PROC / "suffix_profiles.csv"
    profile.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}\n")

    print(profile[["suffix", "n", "hw_rank", "lanes", "maxspeed", "sidewalk_pct", "intx_density", "way_length_m", "divergence"]]
          .to_string(index=False))

    print("\nGenerating figures...")
    plot_divergence(profile)
    plot_attribute_distributions(df)
    print("\nDone. Check data/figures/")
