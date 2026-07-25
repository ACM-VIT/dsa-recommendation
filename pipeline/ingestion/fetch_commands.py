"""
DSA Engine -- Fetch Commands
============================
Loads 1000_manifest_final.json for the ingestion pipeline.

Resolution order (no hardcoded paths):
  1. MANIFEST_PATH environment variable
  2. data/1000_manifest_final.json  (repo root data folder)
  3. Same directory as this script
  4. Repo root (one level above pipeline/ingestion/)
  5. Current working directory

Set env var to override:
    $env:MANIFEST_PATH = "path/to/1000_manifest_final.json"   # Windows
    export MANIFEST_PATH="path/to/1000_manifest_final.json"    # Linux/Mac
"""

import os
import json
from pathlib import Path

MANIFEST_FILENAME = "1000_manifest_final.json"

# Repo root = two levels up from pipeline/ingestion/
_REPO_ROOT = Path(__file__).parent.parent.parent


def _resolve_path() -> Path:
    """Try env var -> data/ folder -> script dir -> repo root -> cwd."""
    env = os.environ.get("MANIFEST_PATH")
    if env:
        return Path(env)

    candidates = [
        _REPO_ROOT / "data" / MANIFEST_FILENAME,
        Path(__file__).parent / MANIFEST_FILENAME,
        _REPO_ROOT / MANIFEST_FILENAME,
        Path.cwd() / MANIFEST_FILENAME,
    ]
    for c in candidates:
        if c.exists():
            return c

    return _REPO_ROOT / "data" / MANIFEST_FILENAME  # default for error message


def _unwrap(data, path) -> list:
    if isinstance(data, list):
        return data
    for key in ("problems", "data", "records", "questions"):
        if key in data:
            return data[key]
    raise ValueError(f"Unexpected JSON structure in {path}. Keys: {list(data.keys())}")


def fetch_local(path: str = None, encoding: str = "utf-8") -> list:
    """
    Load manifest from disk.

    Usage:
        records = fetch_local()                        # auto-resolve
        records = fetch_local("data/manifest.json")    # relative path
    """
    p = Path(path) if path else _resolve_path()
    if not p.exists():
        raise FileNotFoundError(
            f"Manifest not found at: {p.resolve()}\n"
            f"Options:\n"
            f"  1. Place {MANIFEST_FILENAME} in the data/ folder at repo root\n"
            f"  2. Pass path: fetch_local('path/to/{MANIFEST_FILENAME}')\n"
            f"  3. Set env var: MANIFEST_PATH=path/to/{MANIFEST_FILENAME}"
        )
    with open(p, "r", encoding=encoding) as f:
        data = json.load(f)
    records = _unwrap(data, p)
    print(f"[[OK]] Loaded {len(records)} records from {p.resolve()}")
    return records


def fetch_remote(url: str, timeout: int = 30) -> list:
    """Download manifest from a URL."""
    import urllib.request
    print(f"[->] Fetching from: {url}")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    records = _unwrap(data, url)
    print(f"[[OK]] Fetched {len(records)} records")
    return records


def auto_fetch() -> list:
    """Auto-resolve manifest from standard locations."""
    try:
        return fetch_local()
    except FileNotFoundError:
        pass
    raise FileNotFoundError(
        f"\nCould not locate {MANIFEST_FILENAME}.\n\n"
        "Options:\n"
        f"  1. Place {MANIFEST_FILENAME} in the data/ folder at repo root\n"
        "  2. Pass path explicitly:\n"
        "       uv run pipeline/ingestion/run_pipeline.py --input path/to/manifest.json\n"
        "  3. Set env var:\n"
        "       $env:MANIFEST_PATH = 'path/to/manifest.json'  # Windows\n"
        "       export MANIFEST_PATH='path/to/manifest.json'  # Linux/Mac"
    )


def print_fetch_guide():
    print(f"""
+==================================================================+
|         DSA ENGINE -- MANIFEST FETCH GUIDE                       |
+==================================================================+

Place {MANIFEST_FILENAME} in the data/ folder at repo root:

    dsa-recommendation/
    └── data/
        └── {MANIFEST_FILENAME}   <-- put it here

Then run from repo root:
    uv run pipeline/ingestion/run_pipeline.py

Or pass path explicitly:
    uv run pipeline/ingestion/run_pipeline.py --input path/to/{MANIFEST_FILENAME}

Or set env var (Windows):
    $env:MANIFEST_PATH = "path/to/{MANIFEST_FILENAME}"
    uv run pipeline/ingestion/run_pipeline.py
""")


if __name__ == "__main__":
    print_fetch_guide()
