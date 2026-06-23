"""
Download a state OSM PBF extract and filter to road features only.
Geofabrik US state extracts: https://download.geofabrik.de/north-america/us/
Start small (Rhode Island ~10MB) to validate the pipeline before running national.
"""

import os
import urllib.request
import subprocess
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

# Small state for dev/validation — swap for larger or loop all states
BASE = "https://download.geofabrik.de/north-america/us"
STATES = {
    # Already downloaded
    "rhode-island": f"{BASE}/rhode-island-latest.osm.pbf",
    "connecticut":  f"{BASE}/connecticut-latest.osm.pbf",
    "ohio":         f"{BASE}/ohio-latest.osm.pbf",
    "kentucky":     f"{BASE}/kentucky-latest.osm.pbf",
    # Remaining 46 states
    "alabama":        f"{BASE}/alabama-latest.osm.pbf",
    "alaska":         f"{BASE}/alaska-latest.osm.pbf",
    "arizona":        f"{BASE}/arizona-latest.osm.pbf",
    "arkansas":       f"{BASE}/arkansas-latest.osm.pbf",
    "california":     f"{BASE}/california-latest.osm.pbf",
    "colorado":       f"{BASE}/colorado-latest.osm.pbf",
    "delaware":       f"{BASE}/delaware-latest.osm.pbf",
    "florida":        f"{BASE}/florida-latest.osm.pbf",
    "georgia":        f"{BASE}/georgia-latest.osm.pbf",
    "hawaii":         f"{BASE}/hawaii-latest.osm.pbf",
    "idaho":          f"{BASE}/idaho-latest.osm.pbf",
    "illinois":       f"{BASE}/illinois-latest.osm.pbf",
    "indiana":        f"{BASE}/indiana-latest.osm.pbf",
    "iowa":           f"{BASE}/iowa-latest.osm.pbf",
    "kansas":         f"{BASE}/kansas-latest.osm.pbf",
    "louisiana":      f"{BASE}/louisiana-latest.osm.pbf",
    "maine":          f"{BASE}/maine-latest.osm.pbf",
    "maryland":       f"{BASE}/maryland-latest.osm.pbf",
    "massachusetts":  f"{BASE}/massachusetts-latest.osm.pbf",
    "michigan":       f"{BASE}/michigan-latest.osm.pbf",
    "minnesota":      f"{BASE}/minnesota-latest.osm.pbf",
    "mississippi":    f"{BASE}/mississippi-latest.osm.pbf",
    "missouri":       f"{BASE}/missouri-latest.osm.pbf",
    "montana":        f"{BASE}/montana-latest.osm.pbf",
    "nebraska":       f"{BASE}/nebraska-latest.osm.pbf",
    "nevada":         f"{BASE}/nevada-latest.osm.pbf",
    "new-hampshire":  f"{BASE}/new-hampshire-latest.osm.pbf",
    "new-jersey":     f"{BASE}/new-jersey-latest.osm.pbf",
    "new-mexico":     f"{BASE}/new-mexico-latest.osm.pbf",
    "new-york":       f"{BASE}/new-york-latest.osm.pbf",
    "north-carolina": f"{BASE}/north-carolina-latest.osm.pbf",
    "north-dakota":   f"{BASE}/north-dakota-latest.osm.pbf",
    "oregon":         f"{BASE}/oregon-latest.osm.pbf",
    "pennsylvania":   f"{BASE}/pennsylvania-latest.osm.pbf",
    "south-carolina": f"{BASE}/south-carolina-latest.osm.pbf",
    "south-dakota":   f"{BASE}/south-dakota-latest.osm.pbf",
    "tennessee":      f"{BASE}/tennessee-latest.osm.pbf",
    "texas":          f"{BASE}/texas-latest.osm.pbf",
    "utah":           f"{BASE}/utah-latest.osm.pbf",
    "vermont":        f"{BASE}/vermont-latest.osm.pbf",
    "virginia":       f"{BASE}/virginia-latest.osm.pbf",
    "washington":     f"{BASE}/washington-latest.osm.pbf",
    "west-virginia":  f"{BASE}/west-virginia-latest.osm.pbf",
    "wisconsin":      f"{BASE}/wisconsin-latest.osm.pbf",
    "wyoming":        f"{BASE}/wyoming-latest.osm.pbf",
    "district-of-columbia": f"{BASE}/district-of-columbia-latest.osm.pbf",
}

def download(state: str, url: str) -> Path:
    dest = DATA_RAW / f"{state}.osm.pbf"
    roads = DATA_RAW / f"{state}.osm_roads.osm.pbf"
    if roads.exists():
        print(f"[skip] {state} already filtered")
        return dest   # original may not exist; filter_roads will skip too
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return dest
    print(f"[download] {state} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"[ok] {dest}")
    return dest


def filter_roads(pbf_path: Path) -> Path:
    """Use osmium to extract only highway ways (roads) — drops buildings, POIs, etc."""
    out = pbf_path.with_name(pbf_path.stem + "_roads.osm.pbf")
    if out.exists():
        print(f"[skip] {out.name} already filtered")
        pbf_path.unlink(missing_ok=True)
        return out
    cmd = [
        "osmium", "tags-filter",
        str(pbf_path),
        "w/highway",            # ways tagged highway=*
        "-o", str(out),
        "--overwrite",
    ]
    print(f"[filter] {pbf_path.name} -> {out.name}")
    subprocess.run(cmd, check=True)
    pbf_path.unlink()   # delete original once filtered
    print(f"[rm] {pbf_path.name}")
    return out


if __name__ == "__main__":
    for state, url in STATES.items():
        pbf = download(state, url)
        filter_roads(pbf)
    print("\nDone. Next: run 02_extract_streets.py")
