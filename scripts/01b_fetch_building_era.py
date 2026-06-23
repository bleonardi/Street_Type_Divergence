"""
Fetch Census ACS block-group-level housing age data + block group geometries.
Table B25036: Year Structure Built (bins) at block group level.
Outputs a GeoParquet with block group geometry + median year built + era label.

State FIPS: RI=44, CT=09, OH=39, KY=21
"""

import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path

DATA_PROC = Path(__file__).parent.parent / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ACS 5-year 2022 — B25036: year structure built (universe: housing units in structure)
# Bins (E = estimate):
#  _002 Built 2020 or later
#  _003 2010-2019, _004 2000-2009, _005 1990-1999, _006 1980-1989
#  _007 1970-1979, _008 1960-1969, _009 1950-1959, _010 1940-1949
#  _011 1939 or earlier
BIN_VARS = {
    "B25036_002E": 2020,
    "B25036_003E": 2015,  # midpoint of 2010-2019
    "B25036_004E": 2005,
    "B25036_005E": 1995,
    "B25036_006E": 1985,
    "B25036_007E": 1975,
    "B25036_008E": 1965,
    "B25036_009E": 1955,
    "B25036_010E": 1945,
    "B25036_011E": 1935,  # midpoint of pre-1940
}

STATES = {"44": "rhode-island", "09": "connecticut", "39": "ohio", "21": "kentucky"}

ERA_BREAKS = [
    (0,    1919, "pre-1920"),
    (1920, 1944, "1920s-40s"),
    (1945, 1964, "postwar boom"),
    (1965, 1979, "1965-1979"),
    (1980, 1999, "1980-1999"),
    (2000, 9999, "2000+"),
]

def year_to_era(y: float) -> str:
    if pd.isna(y):
        return "unknown"
    for lo, hi, label in ERA_BREAKS:
        if lo <= y <= hi:
            return label
    return "unknown"


def weighted_median_year(row: pd.Series) -> float:
    """Weighted median year built from ACS bin counts."""
    pairs = [(mid, row.get(var, 0) or 0) for var, mid in BIN_VARS.items()]
    pairs = [(mid, w) for mid, w in pairs if w > 0]
    if not pairs:
        return float("nan")
    pairs.sort(key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    cumul = 0
    for mid, w in pairs:
        cumul += w
        if cumul >= total / 2:
            return float(mid)
    return float(pairs[-1][0])


def fetch_census(fips: str) -> pd.DataFrame:
    vars_str = ",".join(BIN_VARS.keys())
    url = (
        f"https://api.census.gov/data/2022/acs/acs5"
        f"?get={vars_str}"
        f"&for=block%20group:*"
        f"&in=state:{fips}%20county:*%20tract:*"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    cols = data[0]
    df = pd.DataFrame(data[1:], columns=cols)
    for v in BIN_VARS:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block group"]
    df["median_year_built"] = df.apply(weighted_median_year, axis=1)
    df["era"] = df["median_year_built"].apply(year_to_era)
    return df[["GEOID", "median_year_built", "era"]]


def fetch_bg_geometries(fips: str, state_name: str) -> gpd.GeoDataFrame:
    import pygris
    gdf = pygris.block_groups(state=fips, year=2022, cache=True)
    gdf = gdf[["GEOID", "geometry"]].copy()
    gdf = gdf.to_crs("EPSG:4326")
    return gdf


if __name__ == "__main__":
    all_gdfs = []
    for fips, name in STATES.items():
        print(f"Fetching {name} ({fips})...")
        try:
            census_df = fetch_census(fips)
            geom_gdf  = fetch_bg_geometries(fips, name)
            merged = geom_gdf.merge(census_df, on="GEOID", how="left")
            merged["state"] = name
            all_gdfs.append(merged)
            print(f"  {len(merged):,} block groups, median year range: "
                  f"{merged['median_year_built'].min():.0f}–{merged['median_year_built'].max():.0f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    combined = pd.concat(all_gdfs, ignore_index=True)
    out = DATA_PROC / "block_group_era.parquet"
    combined.to_parquet(out, index=False)
    print(f"\nSaved {len(combined):,} block groups -> {out}")
    print(combined["era"].value_counts())
