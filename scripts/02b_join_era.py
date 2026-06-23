"""
Spatial join: attach Census block-group era (median year structure built)
to each street way using its centroid.
Reads:  data/processed/streets.parquet
        data/processed/block_group_era.parquet
Writes: data/processed/streets_with_era.parquet
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point

DATA_PROC = Path(__file__).parent.parent / "data" / "processed"

def join_state(state_streets: pd.DataFrame, bg: gpd.GeoDataFrame) -> pd.DataFrame:
    """Spatial join for a single state's streets — keeps memory bounded."""
    from shapely.geometry import Point
    geometry = [Point(r.centroid_lon, r.centroid_lat) for r in state_streets.itertuples()]
    gdf = gpd.GeoDataFrame(state_streets, geometry=geometry, crs="EPSG:4326")
    joined = gpd.sjoin(
        gdf,
        bg[["GEOID", "median_year_built", "era", "geometry"]],
        how="left",
        predicate="within",
    )
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))


def main():
    print("Loading streets...")
    streets = pd.read_parquet(DATA_PROC / "streets.parquet")
    streets = streets[streets["centroid_lat"].notna() & streets["centroid_lon"].notna()].copy()

    print("Loading block group eras...")
    bg_raw = pd.read_parquet(DATA_PROC / "block_group_era.parquet")
    from shapely import wkt as shapely_wkt
    bg_raw["geometry"] = bg_raw["geometry_wkt"].apply(
        lambda x: shapely_wkt.loads(x) if pd.notna(x) else None
    )
    bg = gpd.GeoDataFrame(bg_raw, geometry="geometry", crs="EPSG:4326")
    bg = bg[~bg.geometry.is_empty & bg.geometry.notna()].copy()

    print(f"  {len(streets):,} streets, {len(bg):,} block groups")

    # Process state by state to keep memory bounded at national scale
    states = streets["state"].unique()
    print(f"  Joining {len(states)} states...")
    results = []
    for i, state in enumerate(sorted(states), 1):
        sub = streets[streets["state"] == state]
        # Filter block groups to state bounding box for speed
        minx, miny = sub["centroid_lon"].min() - 0.5, sub["centroid_lat"].min() - 0.5
        maxx, maxy = sub["centroid_lon"].max() + 0.5, sub["centroid_lat"].max() + 0.5
        bg_state = bg.cx[minx:maxx, miny:maxy]
        joined = join_state(sub, bg_state)
        results.append(joined)
        print(f"  [{i}/{len(states)}] {state}: {len(sub):,} ways, "
              f"{joined['era'].notna().mean():.0%} matched")

    result = pd.concat(results, ignore_index=True)
    out = DATA_PROC / "streets_with_era.parquet"
    result.to_parquet(out, index=False)
    print(f"\nSaved {len(result):,} rows -> {out}")
    print("\nEra distribution:")
    print(result["era"].value_counts())
    print(f"\nOverall match rate: {result['era'].notna().mean():.1%}")

if __name__ == "__main__":
    main()
