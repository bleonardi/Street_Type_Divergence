"""
Parse road-filtered OSM PBF files and write a flat parquet with one row per way.
Attributes captured:
  name, suffix, highway, lanes, maxspeed, sidewalk, surface, lit, oneway, state
  way_length_m, node_count, intersection_count, intersection_density (per km)
"""

import re
import osmium
import osmium.geom
import pandas as pd
from collections import Counter
from pathlib import Path

DATA_RAW  = Path(__file__).parent.parent / "data" / "raw"
DATA_PROC = Path(__file__).parent.parent / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

SUFFIX_FAMILIES = {
    "RD": "Road", "ROAD": "Road",
    "ST": "Street", "STR": "Street", "STREET": "Street",
    "AVE": "Avenue", "AV": "Avenue", "AVENUE": "Avenue",
    "BLVD": "Boulevard", "BOULEVARD": "Boulevard", "BL": "Boulevard",
    "DR": "Drive", "DRV": "Drive", "DRIVE": "Drive",
    "LN": "Lane", "LANE": "Lane",
    "CT": "Court", "CRT": "Court", "COURT": "Court",
    "PL": "Place", "PLACE": "Place",
    "CIR": "Circle", "CIRCLE": "Circle",
    "WAY": "Way", "WY": "Way",
    "TER": "Terrace", "TERR": "Terrace", "TERRACE": "Terrace",
    "PKWY": "Parkway", "PKY": "Parkway", "PARKWAY": "Parkway",
    "HWY": "Highway", "HIGHWAY": "Highway",
    "TRL": "Trail", "TRAIL": "Trail",
    "PATH": "Path", "WALK": "Walk",
    "PIKE": "Pike", "TPKE": "Turnpike", "TURNPIKE": "Turnpike",
    "ALY": "Alley", "ALLEY": "Alley",
    "ROW": "Row",
    "RUN": "Run",
}

# Haversine in pure Python — avoids heavy dep for single-segment distances
import math

def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def extract_suffix(name: str) -> str | None:
    if not name:
        return None
    token = name.strip().upper().split()[-1]
    return SUFFIX_FAMILIES.get(token)


def parse_lanes(val: str) -> float | None:
    try:
        return float(val.split(";")[0].strip())
    except Exception:
        return None


def parse_speed(val: str) -> float | None:
    if not val:
        return None
    val = val.lower().strip()
    try:
        if "mph" in val:
            return float(val.replace("mph", "").strip())
        if "kmh" in val or "km/h" in val:
            return float(re.sub(r"[^\d.]", "", val)) * 0.621371
        return float(re.sub(r"[^\d.]", "", val))
    except Exception:
        return None


# ── Pass 1: collect node membership counts ───────────────────────────────────

class NodeCountHandler(osmium.SimpleHandler):
    """Count how many highway ways each node belongs to."""
    def __init__(self):
        super().__init__()
        self.node_way_count: Counter = Counter()

    def way(self, w):
        if not w.tags.get("highway"):
            return
        for n in w.nodes:
            self.node_way_count[n.ref] += 1


# ── Pass 2: extract per-way attributes ───────────────────────────────────────

class RoadHandler(osmium.SimpleHandler):
    def __init__(self, node_way_count: Counter):
        super().__init__()
        self.node_way_count = node_way_count
        self.rows = []

    def way(self, w):
        tags = w.tags
        highway = tags.get("highway", "")
        if not highway:
            return

        name = tags.get("name", "")
        suffix = extract_suffix(name)

        # Geometry — requires locations=True in apply_file
        nodes = list(w.nodes)
        node_count = len(nodes)
        way_length_m = 0.0
        valid_geom = True
        try:
            for i in range(len(nodes) - 1):
                a, b = nodes[i], nodes[i+1]
                way_length_m += _haversine_m(a.lat, a.lon, b.lat, b.lon)
        except Exception:
            valid_geom = False
            way_length_m = None

        # Intersection count: nodes shared with at least one other highway way
        intersection_count = sum(
            1 for n in nodes if self.node_way_count[n.ref] >= 2
        )

        # Density: intersections per km of road
        if valid_geom and way_length_m and way_length_m > 0:
            intersection_density = intersection_count / (way_length_m / 1000)
        else:
            intersection_density = None

        # Centroid for spatial join with block group era data
        if valid_geom and nodes:
            lats = [n.lat for n in nodes if n.lat != 0]
            lons = [n.lon for n in nodes if n.lon != 0]
            centroid_lat = sum(lats) / len(lats) if lats else None
            centroid_lon = sum(lons) / len(lons) if lons else None
        else:
            centroid_lat = centroid_lon = None

        self.rows.append({
            "osm_id":               w.id,
            "name":                 name,
            "suffix":               suffix,
            "highway":              highway,
            "lanes":                parse_lanes(tags.get("lanes", "")),
            "maxspeed":             parse_speed(tags.get("maxspeed", "")),
            "sidewalk":             tags.get("sidewalk", None),
            "surface":              tags.get("surface", None),
            "lit":                  tags.get("lit", None),
            "oneway":               tags.get("oneway", None),
            "node_count":           node_count,
            "way_length_m":         way_length_m if valid_geom else None,
            "intersection_count":   intersection_count,
            "intersection_density": intersection_density,
            "centroid_lat":         centroid_lat,
            "centroid_lon":         centroid_lon,
        })


def process_pbf(pbf_path: Path, state: str) -> pd.DataFrame:
    print(f"  Pass 1 (node counts)...")
    nc_handler = NodeCountHandler()
    nc_handler.apply_file(str(pbf_path))

    print(f"  Pass 2 (way attributes + geometry)...")
    road_handler = RoadHandler(nc_handler.node_way_count)
    road_handler.apply_file(str(pbf_path), locations=True)

    df = pd.DataFrame(road_handler.rows)
    df["state"] = state
    print(f"  {state}: {len(df):,} ways, {df['suffix'].notna().sum():,} with known suffix")
    return df


if __name__ == "__main__":
    # Write per-state parquets to keep peak memory bounded;
    # downstream scripts read via DuckDB wildcard glob.
    PER_STATE = DATA_PROC / "streets_by_state"
    PER_STATE.mkdir(exist_ok=True)

    pbfs = sorted(DATA_RAW.glob("*_roads.osm.pbf"))
    if not pbfs:
        print("No *_roads.osm.pbf files found. Run 01_fetch_data.py first.")
    else:
        for pbf in pbfs:
            state = pbf.name.replace("_roads.osm.pbf", "").replace(".osm", "")
            out = PER_STATE / f"{state}.parquet"
            if out.exists():
                print(f"[skip] {state} already extracted")
                continue
            print(f"Processing {state}...")
            df = process_pbf(pbf, state)
            df.to_parquet(out, index=False)
            print(f"  Saved {len(df):,} rows -> {out.name}")

        # Write a combined view via DuckDB for compatibility with downstream scripts
        import duckdb
        combined_path = DATA_PROC / "streets.parquet"
        print(f"\nMerging per-state parquets -> {combined_path} ...")
        con = duckdb.connect()
        con.execute(f"""
            COPY (SELECT * FROM read_parquet('{PER_STATE}/*.parquet'))
            TO '{combined_path}' (FORMAT PARQUET)
        """)
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{combined_path}')").fetchone()[0]
        print(f"Saved {n:,} rows -> {combined_path}")
        con.close()
