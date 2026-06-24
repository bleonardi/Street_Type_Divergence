"""
Street name semantic analysis — three lenses on placelessness:
  1. Geographic diffusion: how many states does each base name appear in?
  2. Prestige/marketing vocabulary: developer branding words
  3. Name token count: word-count inflation as marketing proxy

Reads:  data/processed/streets_with_era.parquet
Writes: data/processed/name_analysis.parquet
        figures/name_diffusion_by_era.png
        figures/name_prestige_by_era.png
        figures/name_tokens_by_era.png
        figures/name_placelessness_combined.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import re

BASE = Path(__file__).parent.parent
DATA = BASE / "data" / "processed"
FIGS = BASE / "figures"
FIGS.mkdir(exist_ok=True)

ERA_ORDER = ["pre-1920", "1920s-40s", "postwar boom", "1965-1979", "1980-1999", "2000+"]

# ── Prestige / marketing vocabulary ──────────────────────────────────────────
# Words that signal developer branding rather than local reference
PRESTIGE = {
    # Status
    "ESTATES", "ESTATE", "MANOR", "MANORS", "POINTE", "GRANDE", "GRAND",
    "ROYAL", "ROYALE", "IMPERIAL", "SOVEREIGN", "PREMIER", "ELITE",
    "VILLAS", "VILLA", "CHATEAU", "CHATEAUX", "PALMS", "PALACE",
    # Enclosure / community
    "RESERVE", "PRESERVE", "ENCLAVE", "COMMONS", "VILLAGE", "CROSSING",
    "LANDING", "HARBOUR", "HARBOR", "HAVEN", "RETREAT", "SANCTUARY",
    "GATEWAY", "OVERLOOK", "VISTA", "RIDGE", "SUMMIT", "PINNACLE",
    # Nature (generic pastoral, not locally referential)
    "GLEN", "GLEN", "CHASE", "RUN", "HOLLOW", "VALE", "VALE",
    "CREST", "BLUFF", "MEADOW", "MEADOWS", "GLEN", "BROOK", "BROOKS",
    "CREEK", "SPRINGS", "SPRING", "FALLS", "LAKES", "LAKE", "POND",
    "PINES", "PINE", "OAK", "OAKS", "ELM", "MAPLE", "CEDAR", "WILLOW",
    "BIRCH", "ASH", "HICKORY", "WALNUT", "CHESTNUT", "SYCAMORE",
    "MAGNOLIA", "DOGWOOD", "LAUREL", "IVY", "FERN", "HEATHER",
    "CLOVER", "THISTLE", "BRIAR", "BRAMBLE",
    # Directional grandeur
    "POINTE", "POINTE", "PLACE",   # 'Place' as aspirational suffix used in base
    # Activity / lifestyle
    "HUNT", "HUNTERS", "RIDING", "FOXFIRE", "FOXHUNT",
}

# Single nature words that are stock suburban vocabulary
NATURE_WORDS = {
    "OAK", "OAKS", "PINE", "PINES", "MAPLE", "ELM", "CEDAR", "WILLOW",
    "BIRCH", "ASH", "HICKORY", "WALNUT", "CHESTNUT", "SYCAMORE", "MAGNOLIA",
    "DOGWOOD", "LAUREL", "IVY", "FERN", "HEATHER", "CLOVER", "THISTLE",
    "BRIAR", "HOLLY", "CHERRY", "PEACH", "APPLE", "LINDEN", "POPLAR",
    "SPRUCE", "FIR", "HEMLOCK", "BEECH", "COTTONWOOD", "LOCUST",
    "CREEK", "BROOK", "RIVER", "LAKE", "POND", "SPRING", "SPRINGS",
    "FALLS", "RIDGE", "HILL", "HILLS", "VALLEY", "GLEN", "HOLLOW",
    "MEADOW", "MEADOWS", "FIELD", "FIELDS", "FOREST", "WOODS", "WOOD",
    "GROVE", "GLADE", "BLUFF", "CLIFF", "STONE", "ROCK", "BOULDER",
    "MILL", "MILLS", "FARM", "FARMS", "ORCHARD", "GARDEN", "GARDENS",
    "PRAIRIE", "PLAIN", "PLAINS", "MOOR", "MARSH", "FEN", "MEAD",
    "CREST", "SUMMIT", "PEAK", "KNOLL", "DALE", "DELL", "RUN",
    "CHASE", "HUNT", "HAVEN", "HARBOR", "COVE", "BAY", "SHORE",
    "VISTA", "VIEW", "OVERLOOK",
}


def strip_suffix(name: str, suffix: str | None) -> str:
    """Remove the suffix token from the full street name, return upper-cased base."""
    name = name.strip()
    if suffix:
        # Remove suffix from end, case-insensitive
        pattern = r'\s+' + re.escape(suffix) + r'\s*$'
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    return name.upper().strip()


def token_count(base: str) -> int:
    return len(base.split())


def prestige_score(base: str) -> float:
    """Fraction of tokens that are prestige/nature words."""
    tokens = set(base.split())
    if not tokens:
        return 0.0
    hits = tokens & PRESTIGE
    return len(hits) / len(tokens)


def has_prestige(base: str) -> bool:
    return bool(set(base.split()) & PRESTIGE)


def nature_compound(base: str) -> bool:
    """True if base contains 2+ distinct nature words (the 'Willowbrook' pattern)."""
    tokens = base.split()
    # Also check fused compounds by splitting on case boundaries or known words
    hits = [t for t in tokens if t in NATURE_WORDS]
    return len(hits) >= 2


def vectorized_strip_suffix(names: pd.Series, suffixes: pd.Series) -> pd.Series:
    """Strip suffix from name — vectorized, no row-wise apply."""
    result = names.str.upper().str.strip()
    mask = suffixes.notna()
    # Build per-row patterns only for rows that have a suffix
    for sfx in suffixes[mask].str.upper().unique():
        m = mask & (suffixes.str.upper() == sfx)
        result[m] = result[m].str.replace(
            r'\s+' + re.escape(sfx) + r'\s*$', '', regex=True
        ).str.strip()
    return result


def load_data() -> pd.DataFrame:
    cols = ["osm_id", "name", "suffix", "era", "state"]
    df = pd.read_parquet(DATA / "streets_with_era.parquet", columns=cols)
    df = df[df["name"].notna() & df["era"].isin(ERA_ORDER)].copy()
    df["base"] = vectorized_strip_suffix(df["name"], df["suffix"])
    df = df[df["base"].str.len() > 0]
    return df


def compute_diffusion(df: pd.DataFrame) -> pd.Series:
    """For each unique base name, count how many states it appears in."""
    print("Computing geographic diffusion...")
    return df.groupby("base")["state"].nunique().rename("n_states")


def classify_names(bases: pd.Index) -> pd.DataFrame:
    """Compute all features on unique base names — join back by index."""
    s = pd.Series(bases, index=bases)
    tokens = s.str.split()
    token_sets = tokens.apply(set)

    prestige_set = PRESTIGE
    nature_set = NATURE_WORDS

    has_p = token_sets.apply(lambda t: bool(t & prestige_set))
    nature_hits = token_sets.apply(lambda t: len(t & nature_set))
    tok_count = tokens.apply(len)

    return pd.DataFrame({
        "token_count": tok_count,
        "has_prestige": has_p,
        "is_nature_compound": nature_hits >= 2,
    })


def main():
    print("Loading data (named ways only)...")
    df = load_data()
    print(f"  {len(df):,} named ways")

    # ── Feature engineering on unique names, then join ────────────────────────
    print("Classifying unique base names...")
    unique_bases = df["base"].unique()
    feat = classify_names(unique_bases)
    df = df.join(feat, on="base")

    diffusion = compute_diffusion(df)
    df = df.join(diffusion, on="base")

    out = DATA / "name_analysis.parquet"
    df[["osm_id", "name", "base", "suffix", "era", "state",
        "token_count", "has_prestige", "is_nature_compound", "n_states"]
    ].to_parquet(out, index=False)
    print(f"Saved -> {out}")

    # ── Summary tables ────────────────────────────────────────────────────────
    era_stats = df.groupby("era").agg(
        n=("base", "count"),
        median_diffusion=("n_states", "median"),
        mean_diffusion=("n_states", "mean"),
        prestige_pct=("has_prestige", "mean"),
        nature_compound_pct=("is_nature_compound", "mean"),
        mean_tokens=("token_count", "mean"),
    )
    era_stats = era_stats.reindex([e for e in ERA_ORDER if e in era_stats.index])

    print("\n=== ERA SUMMARY ===")
    print(era_stats.round(3).to_string())

    # ── Plotting ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 13))
    fig.suptitle("Street Name Placelessness by Development Era", fontsize=14, y=0.98)
    present_eras = [e for e in ERA_ORDER if e in era_stats.index]
    era_stats = era_stats.loc[present_eras]
    era_labels = [e.replace("postwar boom", "postwar\nboom") for e in present_eras]

    x = np.arange(len(present_eras))
    w = 0.6

    # 1. Geographic diffusion
    ax = axes[0]
    vals = era_stats["median_diffusion"].values
    bars = ax.bar(x, vals, width=w, color=plt.cm.YlOrRd(np.linspace(0.2, 0.85, len(x))))
    ax.set_xticks(x); ax.set_xticklabels(era_labels)
    ax.set_ylabel("Median states\nname appears in")
    ax.set_title("Geographic diffusion — how generic is the name nationally?", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)

    # 2. Prestige / marketing vocabulary share
    ax = axes[1]
    vals2 = era_stats["prestige_pct"].values * 100
    bars2 = ax.bar(x, vals2, width=w, color=plt.cm.YlOrRd(np.linspace(0.2, 0.85, len(x))))
    ax.set_xticks(x); ax.set_xticklabels(era_labels)
    ax.set_ylabel("% streets with\nmarketing vocabulary")
    ax.set_title("Prestige / marketing vocabulary — Estates, Manor, Pointe, nature words…", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    for bar, v in zip(bars2, vals2):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9)

    # 3. Mean token count
    ax = axes[2]
    vals3 = era_stats["mean_tokens"].values
    bars3 = ax.bar(x, vals3, width=w, color=plt.cm.YlOrRd(np.linspace(0.2, 0.85, len(x))))
    ax.set_xticks(x); ax.set_xticklabels(era_labels)
    ax.set_ylabel("Mean words in\nbase name")
    ax.set_title("Name length — word-count inflation as marketing proxy", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    for bar, v in zip(bars3, vals3):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.001, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGS / "name_placelessness_combined.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> figures/name_placelessness_combined.png")

    # ── Top names per era ─────────────────────────────────────────────────────
    print("\n=== TOP 10 BASE NAMES PER ERA (by count) ===")
    for era in present_eras:
        top = df[df["era"] == era]["base"].value_counts().head(10)
        print(f"\n{era}:")
        for name, ct in top.items():
            diff = diffusion.get(name, 0)
            flag = "★" if has_prestige(name) else " "
            print(f"  {flag} {name:<30} n={ct:>7,}  states={diff:>2}")

    # ── Most locally unique names (low diffusion, high count) ─────────────────
    print("\n=== MOST GEOGRAPHICALLY UNIQUE NAMES (appear in ≤2 states, n≥100) ===")
    unique_names = (
        df.groupby("base")
        .agg(n=("base","count"), n_states=("n_states","first"), era=("era", lambda x: x.mode()[0]))
        .query("n_states <= 2 and n >= 100")
        .sort_values("n", ascending=False)
        .head(20)
    )
    print(unique_names.to_string())


if __name__ == "__main__":
    main()
