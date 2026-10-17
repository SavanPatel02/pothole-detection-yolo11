"""Download the Roboflow Pothole dataset (665 images, ODbL v1.0) in YOLO format.

Needs a free Roboflow API key: taken from the ROBOFLOW_API_KEY env var, or asked for
at a hidden prompt. The key is never printed or saved.

    python download_data.py            # latest version
    python download_data.py --version 1
"""

import argparse
import getpass
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

WORKSPACE = "brad-dwyer"
PROJECT = "pothole-voxrl"
API = "https://api.roboflow.com"
OUT = Path(__file__).parent / "datasets" / "pothole"


def get(url, key):
    r = requests.get(url, params={"api_key": key}, timeout=60)
    if r.status_code != 200:
        sys.exit(f"Roboflow API error {r.status_code} for {url}: {r.text[:300]}")
    return r.json()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", type=int, help="dataset version (default: latest)")
    args = p.parse_args()

    key = os.environ.get("ROBOFLOW_API_KEY") or getpass.getpass("Roboflow API key (hidden): ").strip()
    if not key:
        sys.exit("No API key given.")

    project = get(f"{API}/{WORKSPACE}/{PROJECT}", key)
    versions = sorted(int(v["id"].rsplit("/", 1)[-1]) for v in project.get("versions", []))
    print("available versions:", versions)
    version = args.version or versions[-1]

    # Export generation can take a few seconds on first request.
    for _ in range(30):
        export = get(f"{API}/{WORKSPACE}/{PROJECT}/{version}/yolov11", key).get("export", {})
        if export.get("link"):
            break
        time.sleep(5)
    else:
        sys.exit("Export link not ready, try again in a minute.")

    print(f"downloading version {version} ...")
    data = requests.get(export["link"], timeout=300).content
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(OUT)
    for split in ("train", "valid", "test"):
        n = len(list((OUT / split / "images").glob("*"))) if (OUT / split).exists() else 0
        print(f"{split}: {n} images")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
