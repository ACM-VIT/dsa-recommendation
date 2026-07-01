"""
DSA Engine -- Full Pipeline Runner
===================================
Single entrypoint that:
  1. Fetches the manifest
  2. Transforms all records into the vector pool schema
  3. Validates every row against Qdrant readiness gates
  4. Saves clean Parquet + JSON + rejection manifest

Run from repo root:
    uv run pipeline/ingestion/run_pipeline.py
    uv run pipeline/ingestion/run_pipeline.py --input data/1000_manifest_final.json
    uv run pipeline/ingestion/run_pipeline.py --input data/1000_manifest_final.json --preview 5
"""

import argparse
import sys
from pathlib import Path

# Allow imports from this folder (pipeline/ingestion/) regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

# Repo root = two levels up from pipeline/ingestion/
_REPO_ROOT = Path(__file__).parent.parent.parent

from fetch_commands import auto_fetch, fetch_local, print_fetch_guide
from ingest import build_vector_pool, save_outputs, preview
from validator import validate_dataframe, rejection_manifest


def parse_args():
    p = argparse.ArgumentParser(description="DSA Engine -- Full Pipeline")
    p.add_argument("--input",   "-i", default=None,
                   help="Path to manifest JSON (default: data/1000_manifest_final.json)")
    p.add_argument("--output",  "-o",
                   default=str(_REPO_ROOT / "data" / "vector_pool"),
                   help="Output directory (default: data/vector_pool/)")
    p.add_argument("--preview", "-p", type=int, default=3,
                   help="Number of records to preview (0 = skip)")
    p.add_argument("--guide", action="store_true",
                   help="Print fetch guide and exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.guide:
        print_fetch_guide()
        return

    print("\n" + "=" * 60)
    print("  DSA ENGINE -- VECTOR POOL PIPELINE")
    print("=" * 60)

    # -- Step 1: Fetch --------------------------------------------------------
    if args.input:
        print(f"[1/4] Loading manifest from: {args.input}")
        records = fetch_local(args.input)
    else:
        print("[1/4] Auto-fetching manifest...")
        sample = _REPO_ROOT / "data" / "_sample_manifest.json"
        try:
            records = auto_fetch()
        except FileNotFoundError:
            if sample.exists():
                print(f"[!]  Using sample manifest: {sample}")
                records = fetch_local(str(sample))
            else:
                print("[[X]] No manifest found.")
                print("      Run with --guide for help, or pass --input path/to/manifest.json")
                sys.exit(1)

    print(f"     -> {len(records)} problems loaded")

    # -- Step 2: Transform ----------------------------------------------------
    print("\n[2/4] Transforming records -> vector pool schema...")
    df = build_vector_pool(records)
    print(f"     -> {df.shape[0]} rows x {df.shape[1]} columns")

    # -- Step 3: Validate -----------------------------------------------------
    print("\n[3/4] Validating rows for Qdrant readiness...")
    clean_df, rejected_df, results = validate_dataframe(df, verbose=True)

    # -- Step 4: Save ---------------------------------------------------------
    print(f"\n[4/4] Saving outputs -> {args.output}")
    save_outputs(clean_df, args.output)

    rej_path = Path(args.output) / "rejected.json"
    rejection_manifest(rejected_df, results, str(rej_path))

    raw_path = Path(args.output) / "vector_pool_raw.parquet"
    try:
        df.reset_index(drop=True).to_parquet(raw_path, index=False)
        print(f"[[OK]] Raw pool saved -> {raw_path}")
    except ImportError:
        from ingest import _save_csv_fallback
        _save_csv_fallback(df.reset_index(drop=True), Path(args.output))

    if args.preview > 0 and len(clean_df) > 0:
        preview(clean_df, n=min(args.preview, len(clean_df)))

    # -- Summary --------------------------------------------------------------
    total    = len(df)
    passed   = len(clean_df)
    rejected = len(rejected_df)
    parquet_out = Path(args.output) / "vector_pool.parquet"

    print("=" * 60)
    print(f"  PIPELINE COMPLETE")
    print(f"  Total    : {total}")
    print(f"  [OK] Clean   : {passed}  -> {parquet_out}")
    print(f"  [X]  Rejected: {rejected}  -> {rej_path}")
    print("=" * 60)
    print()
    print("  NEXT STEP: Generate embeddings")
    print(f"  uv run pipeline/embeddings/embedder.py --input {parquet_out} --output {Path(args.output)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
