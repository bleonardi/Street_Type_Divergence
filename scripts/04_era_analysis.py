"""
Era analysis: for each suffix × era combination, compute built-form profiles.
Key question: does "Road" in a pre-1920 block group look different from
"Road" in a 1970s subdivision? How much has the meaning of a suffix drifted
across eras of urban development?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

DATA_PROC = Path(__file__).parent.parent / "data" / "processed"
FIG_DIR   = DATA_PROC.parent / "data" / "figures"
FIG_DIR   = Path(__file__).parent.parent / "data" / "figures"
FIG_DIR.mkdir(exist_ok=True)

ERA_ORDER = ["pre-1920", "1920s-40s", "postwar boom", "1965-1979", "1980-1999", "2000+"]
ERA_COLORS = {
    "pre-1920":      "#1a237e",
    "1920s-40s":     "#1565c0",
    "postwar boom":  "#0288d1",
    "1965-1979":     "#f9a825",
    "1980-1999":     "#e65100",
    "2000+":         "#b71c1c",
}

HIGHWAY_RANK = {
    "service": 1, "living_street": 1, "pedestrian": 1, "track": 1,
    "residential": 2, "unclassified": 2,
    "tertiary": 3, "tertiary_link": 3,
    "secondary": 4, "secondary_link": 4,
    "primary": 5, "primary_link": 5,
    "trunk": 6, "trunk_link": 6,
    "motorway": 7, "motorway_link": 7,
}

MIN_N = 50  # minimum ways per suffix × era cell to include


def load() -> pd.DataFrame:
    path = DATA_PROC / "streets_with_era.parquet"
    if not path.exists():
        raise FileNotFoundError("Run 02b_join_era.py first")
    df = pd.read_parquet(path)
    df = df[df["suffix"].notna() & df["era"].notna() & (df["era"] != "unknown")].copy()
    df["hw_rank"] = df["highway"].map(HIGHWAY_RANK)
    df["has_sidewalk"] = df["sidewalk"].isin(["both","left","right","yes","separate"])
    df["era"] = pd.Categorical(df["era"], categories=ERA_ORDER, ordered=True)
    return df


def era_profiles(df: pd.DataFrame) -> pd.DataFrame:
    profile = (
        df.groupby(["suffix", "era"], observed=True).agg(
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
    profile = profile[profile["n"] >= MIN_N]
    return profile


def plot_suffix_era_heatmap(profile: pd.DataFrame, attr: str, label: str, fname: str):
    """Heatmap: rows = suffixes, columns = eras, color = attribute value."""
    pivot = profile.pivot(index="suffix", columns="era", values=attr)
    pivot = pivot.reindex(columns=[e for e in ERA_ORDER if e in pivot.columns])

    # Order suffixes by their pre-1920 value (or overall median) for readability
    row_order = pivot.mean(axis=1).sort_values().index
    pivot = pivot.loc[row_order]

    fig, ax = plt.subplots(figsize=(11, max(5, len(pivot) * 0.45)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlBu_r")
    plt.colorbar(im, ax=ax, label=label, shrink=0.7)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{label} by suffix × building era", fontsize=13, pad=10)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(val) > pivot.values[~np.isnan(pivot.values)].mean() * 0.8 else "black")

    plt.tight_layout()
    out = FIG_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close()


def plot_suffix_era_lines(profile: pd.DataFrame, attr: str, label: str,
                          fname: str, top_suffixes: list[str]):
    """Line chart: x = era, y = attribute, one line per suffix."""
    sub = profile[profile["suffix"].isin(top_suffixes)].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    era_positions = {e: i for i, e in enumerate(ERA_ORDER)}

    for suffix, grp in sub.groupby("suffix"):
        grp = grp.sort_values("era")
        x = [era_positions[e] for e in grp["era"] if e in era_positions]
        y = grp.set_index("era").reindex(ERA_ORDER)[attr].dropna().values
        eras_present = [e for e in ERA_ORDER if e in grp["era"].values and not pd.isna(grp.set_index("era").loc[e, attr] if e in grp["era"].values else np.nan)]
        x = [era_positions[e] for e in eras_present]
        ax.plot(x, y, marker="o", label=suffix, linewidth=2)

    ax.set_xticks(range(len(ERA_ORDER)))
    ax.set_xticklabels(ERA_ORDER, rotation=20, ha="right")
    ax.set_ylabel(label)
    ax.set_title(f"{label} across building eras, by suffix", fontsize=12)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close()


def plot_convergence(profile: pd.DataFrame):
    """
    Show how spread across suffixes NARROWS over eras — the convergence story.
    For each era, compute IQR of median speeds across suffixes.
    Narrowing IQR = suffixes becoming more homogeneous (less meaningful).
    """
    rows = []
    for era in ERA_ORDER:
        sub = profile[profile["era"] == era]
        if len(sub) < 3:
            continue
        for attr, label in [("maxspeed", "Speed (mph)"), ("intx_density", "Intx/km"), ("way_length_m", "Way length (m)")]:
            vals = sub[attr].dropna()
            if len(vals) < 3:
                continue
            rows.append({
                "era": era,
                "attr": label,
                "iqr": vals.quantile(0.75) - vals.quantile(0.25),
                "cv": vals.std() / vals.mean() if vals.mean() > 0 else np.nan,
            })
    conv = pd.DataFrame(rows)

    attrs = conv["attr"].unique()
    fig, axes = plt.subplots(1, len(attrs), figsize=(5 * len(attrs), 4), sharey=False)
    if len(attrs) == 1:
        axes = [axes]
    for ax, attr in zip(axes, attrs):
        sub = conv[conv["attr"] == attr].set_index("era").reindex(ERA_ORDER).dropna()
        ax.plot(range(len(sub)), sub["cv"].values, marker="o", color="#d73027", linewidth=2)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub.index, rotation=25, ha="right")
        ax.set_ylabel("Coeff. of variation (spread across suffixes)")
        ax.set_title(attr)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Suffix convergence over eras: narrowing spread = suffixes losing meaning",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "suffix_convergence_over_eras.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    print("Loading data...")
    df = load()
    print(f"  {len(df):,} ways with suffix + era\n")

    print("Computing era profiles...")
    profile = era_profiles(df)
    profile.to_csv(DATA_PROC / "era_profiles.csv", index=False)
    print(f"  {len(profile):,} suffix × era cells (n >= {MIN_N})\n")

    top = df["suffix"].value_counts().head(10).index.tolist()

    print("Generating figures...")
    plot_suffix_era_heatmap(profile, "maxspeed",      "Median speed (mph)",     "era_heatmap_speed.png")
    plot_suffix_era_heatmap(profile, "intx_density",  "Median intx/km",         "era_heatmap_intx.png")
    plot_suffix_era_heatmap(profile, "way_length_m",  "Median way length (m)",  "era_heatmap_length.png")
    plot_suffix_era_lines(profile,   "maxspeed",      "Median speed (mph)",     "era_lines_speed.png",  top)
    plot_suffix_era_lines(profile,   "intx_density",  "Median intx/km",         "era_lines_intx.png",   top)
    plot_convergence(profile)

    print("\nTop findings — speed by suffix × era:")
    pivot = profile.pivot(index="suffix", columns="era", values="maxspeed")
    pivot = pivot.reindex(columns=[e for e in ERA_ORDER if e in pivot.columns])
    print(pivot.round(1).to_string())
