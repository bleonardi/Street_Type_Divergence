"""
Download official road centerline data from ODOT and KYTC.
Both include speed limits and lane counts as standard attributes.

ODOT: https://gis.dot.state.oh.us/tims/Data/Download
KYTC: https://transportation.ky.gov/Planning/Pages/Centerlines.aspx
"""

import urllib.request
import zipfile
import subprocess
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

# KYTC HIS extract URLs — attributes stored as separate LRS layers
BASE = "http://transportation.ky.gov/Planning/Highway%20Information%20System%20Extracts"
SOURCES = {
    # State System centerlines (geometry + route IDs)
    "kytc_ss": {
        "url":     f"{BASE}/SS.zip",
        "zip":     DATA_RAW / "kytc_ss.zip",
        "out_dir": DATA_RAW / "kytc_ss",
    },
    # Speed Limit — LRS layer (RT_UNIQUE + BEG_MP + END_MP + SPD_LIM)
    "kytc_sl": {
        "url":     f"{BASE}/SL.zip",
        "zip":     DATA_RAW / "kytc_sl.zip",
        "out_dir": DATA_RAW / "kytc_sl",
    },
    # Through Lanes — LRS layer (RT_UNIQUE + BEG_MP + END_MP + LANES)
    "kytc_ln": {
        "url":     f"{BASE}/LN.zip",
        "zip":     DATA_RAW / "kytc_ln.zip",
        "out_dir": DATA_RAW / "kytc_ln",
    },
}


def download(name: str, url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  [skip] {dest.name} already downloaded")
        return True
    print(f"  [download] {name} from {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [ok] {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [error] {e}")
        return False


def unzip(zip_path: Path, out_dir: Path):
    if out_dir.exists() and any(out_dir.glob("*.shp")):
        print(f"  [skip] {out_dir.name} already extracted")
        return
    out_dir.mkdir(exist_ok=True)
    print(f"  [unzip] -> {out_dir.name}/")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


if __name__ == "__main__":
    for name, cfg in SOURCES.items():
        print(f"\n{name.upper()}:")
        ok = download(name, cfg["url"], cfg["zip"])
        if ok:
            unzip(cfg["zip"], cfg["out_dir"])

    print("\nDone. Next: run 01d_normalize_dot_roads.py")
