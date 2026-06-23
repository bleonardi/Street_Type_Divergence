"""
Normalize KYTC HIS shapefiles into a common schema for joining to OSM streets.

Available:
  SS — state system centerlines (geometry + route ID + name)
  LN — through lanes per route segment (LANES, LANESCRD, LANESNC)
  SL — speed limits [FILE MISSING from KYTC server as of 2026-06]

Output: data/processed/kytc_roads.parquet
  columns: rt_unique, name, suffix, lanes, speed_mph (NaN), geometry (WGS84)

Join strategy (for 02_extract_streets.py):
  Spatial nearest-neighbor join to OSM ways — attach lanes from KYTC to OSM
  segments within 20m, preferring KYTC lanes over OSM lanes tag.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path

DATA_RAW  = Path(__file__).parent.parent / "data" / "raw"
DATA_PROC = Path(__file__).parent.parent / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

SUFFIX_FAMILIES = {
    "RD": "Road", "ROAD": "Road",
    "ST": "Street", "STR": "Street", "STREET": "Street",
    "AVE": "Avenue", "AV": "Avenue", "AVENUE": "Avenue",
    "BLVD": "Boulevard", "BOULEVARD": "Boulevard",
    "DR": "Drive", "DRV": "Drive", "DRIVE": "Drive",
    "LN": "Lane", "LANE": "Lane",
    "CT": "Court", "CRT": "Court", "COURT": "Court",
    "PL": "Place", "PLACE": "Place",
    "CIR": "Circle", "CIRCLE": "Circle",
    "WAY": "Way",
    "TER": "Terrace", "TERR": "Terrace", "TERRACE": "Terrace",
    "PKWY": "Parkway", "PKY": "Parkway", "PARKWAY": "Parkway",
    "HWY": "Highway", "HIGHWAY": "Highway",
    "TRL": "Trail", "TRAIL": "Trail",
    "PIKE": "Pike",
    "TPKE": "Turnpike", "TURNPIKE": "Turnpike",
    "ALY": "Alley", "ALLEY": "Alley",
}


def extract_suffix(name: str) -> str | None:
    if not name:
        return None
    token = str(name).strip().upper().split()[-1]
    return SUFFIX_FAMILIES.get(token)


def load_kytc() -> gpd.GeoDataFrame:
    # ── Centerlines (SS) ────────────────────────────────────────────────────
    ss_shp = list((DATA_RAW / "kytc_ss").glob("*.shp"))
    if not ss_shp:
        raise FileNotFoundError("kytc_ss/*.shp not found — run 01c_fetch_dot_roads.py")
    ss = gpd.read_file(ss_shp[0])
    ss = ss[["RT_UNIQUE", "RT_DESCR", "geometry"]].copy()
    ss = ss.to_crs("EPSG:4326")
    ss["name"]   = ss["RT_DESCR"].str.strip()
    ss["suffix"] = ss["name"].apply(extract_suffix)
    print(f"  SS: {len(ss):,} centerline segments")

    # ── Lanes (LN) — join by RT_UNIQUE ──────────────────────────────────────
    ln_shp = list((DATA_RAW / "kytc_ln").glob("*.shp"))
    if ln_shp:
        ln = gpd.read_file(ln_shp[0])
        # LANES = total, LANESCRD = cardinal direction, LANESNC = non-cardinal
        ln_agg = (
            ln.groupby("RT_UNIQUE")["LANES"]
            .median()
            .reset_index()
            .rename(columns={"LANES": "lanes_kytc"})
        )
        ss = ss.merge(ln_agg, on="RT_UNIQUE", how="left")
        pct = ss["lanes_kytc"].notna().mean()
        print(f"  LN joined: {pct:.1%} of segments have lane count")
    else:
        ss["lanes_kytc"] = None
        print("  LN: not found, skipping lanes")

    # ── Speed (SL) — file missing from KYTC server ──────────────────────────
    ss["speed_kytc"] = float("nan")
    print("  SL: file missing from KYTC server — speed will remain NaN")

    ss = ss.rename(columns={"RT_UNIQUE": "rt_unique"})
    return ss[["rt_unique", "name", "suffix", "lanes_kytc", "speed_kytc", "geometry"]]


if __name__ == "__main__":
    print("Processing KYTC data...")
    kytc = load_kytc()

    out = DATA_PROC / "kytc_roads.parquet"
    kytc.to_parquet(out, index=False)
    print(f"\nSaved {len(kytc):,} segments -> {out}")
    print("\nSuffix distribution:")
    print(kytc["suffix"].value_counts().head(15))
    print(f"\nLane count sample:\n{kytc['lanes_kytc'].describe()}")
