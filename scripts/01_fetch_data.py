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
STATES = {
    "rhode-island": "https://download.geofabrik.de/north-america/us/rhode-island-latest.osm.pbf",
    "connecticut":  "https://download.geofabrik.de/north-america/us/connecticut-latest.osm.pbf",
    "ohio":         "https://download.geofabrik.de/north-america/us/ohio-latest.osm.pbf",
    "kentucky":     "https://download.geofabrik.de/north-america/us/kentucky-latest.osm.pbf",
}

def download(state: str, url: str) -> Path:
    dest = DATA_RAW / f"{state}.osm.pbf"
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
    return out


if __name__ == "__main__":
    for state, url in STATES.items():
        pbf = download(state, url)
        filter_roads(pbf)
    print("\nDone. Next: run 02_extract_streets.py")
