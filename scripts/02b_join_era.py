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

def main():
    print("Loading streets...")
    streets = pd.read_parquet(DATA_PROC / "streets.parquet")
    streets = streets[streets["centroid_lat"].notna() & streets["centroid_lon"].notna()].copy()

    print("Loading block group eras...")
    bg_raw = pd.read_parquet(DATA_PROC / "block_group_era.parquet")
    # R writes geometry as WKT column
    from shapely import wkt as shapely_wkt
    bg_raw["geometry"] = bg_raw["geometry_wkt"].apply(
        lambda x: shapely_wkt.loads(x) if pd.notna(x) else None
    )
    bg = gpd.GeoDataFrame(bg_raw, geometry="geometry", crs="EPSG:4326")
    bg = bg[bg.geometry.notna()].copy()

    print(f"  {len(streets):,} streets, {len(bg):,} block groups")

    # Build GeoDataFrame from street centroids
    geometry = [Point(row.centroid_lon, row.centroid_lat) for row in streets.itertuples()]
    streets_gdf = gpd.GeoDataFrame(streets, geometry=geometry, crs="EPSG:4326")

    # Spatial join — match each centroid to the block group it falls in
    print("Spatial joining (this takes a minute)...")
    joined = gpd.sjoin(
        streets_gdf,
        bg[["GEOID", "median_year_built", "era", "geometry"]],
        how="left",
        predicate="within",
    )

    # Drop geometry columns, keep flat DataFrame
    result = pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))

    out = DATA_PROC / "streets_with_era.parquet"
    result.to_parquet(out, index=False)
    print(f"\nSaved {len(result):,} rows -> {out}")
    print("\nEra distribution:")
    print(result["era"].value_counts())
    print(f"\nMatch rate: {result['era'].notna().mean():.1%} of streets matched to a block group")

if __name__ == "__main__":
    main()
