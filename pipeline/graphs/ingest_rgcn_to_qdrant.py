"""
DSA Engine -- RGCN Step 4: Ingest into Qdrant
=============================================
Writes all embeddings into Qdrant as separate collections:

    problems_question  : 1024-d  BGE question embedding
    problems_solution  : 768-d   GraphCodeBERT solution embedding
    problems_rgcn      : 128-d   pure RGCN structural embedding
    problems_full      : 1920-d  L2-norm concat(QS 1792 + RGCN 128)

question + solution + rgcn are stored as individual collections so the
rec engine can query each signal independently or fuse them at query time.

By default reads ARTIFACTS (graph.pt + rgcn_problem_embeddings.npy) for
rgcn/full, and the embedded parquet for question/solution.
Use --from-parquet to source everything from the parquet artifact.

Run:
    python pipeline/graphs/ingest_rgcn_to_qdrant.py
    python pipeline/graphs/ingest_rgcn_to_qdrant.py --no-question --no-solution
    python pipeline/graphs/ingest_rgcn_to_qdrant.py --from-parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import config as C

EMBED_KEYS = {"question_embedding", "solution_embedding", "rgcn_embedding",
              "question_solution_embedding", "full_embedding"}


# ---------------------------------------------------------------------------
# Core upsert
# ---------------------------------------------------------------------------

def ingest_embeddings(client, collection, ids, vectors, payloads, batch_size=128):
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, OptimizersConfigDiff,
    )
    valid = [(i, v) for i, v in enumerate(vectors) if v is not None]
    if not valid:
        print(f"[X] nothing to ingest into '{collection}' (all vectors None)")
        return
    dim = len(valid[0][1])
    print(f"\n[->] {collection}: dim={dim}  points={len(valid)}")

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )
        print(f"[OK] created '{collection}'")
    else:
        print(f"[->] '{collection}' exists -- upserting")

    points = [
        PointStruct(id=i, vector=list(map(float, v)), payload=payloads[i])
        for i, v in valid
    ]
    t0 = time.time()
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=collection,
                      points=points[start:start + batch_size])
        pct = min(100, int(100 * (start + batch_size) / max(len(points), 1)))
        print(f"\r  {pct}%", end="", flush=True)
    client.update_collection(
        collection_name=collection,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20000))
    cnt = client.count(collection_name=collection).count
    print(f"\n[OK] '{collection}' now has {cnt} points  ({time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# Payload helper
# ---------------------------------------------------------------------------

def _clean_payload(meta: dict) -> dict:
    out = {}
    for k, v in (meta or {}).items():
        if k in EMBED_KEYS or v is None:
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, float) and np.isnan(v):
            continue
        else:
            out[k] = v
    return out


def _as_vec(val):
    if val is None:
        return None
    try:
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0 or np.any(np.isnan(arr)):
            return None
        return arr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Load from artifacts + parquet
# ---------------------------------------------------------------------------

def from_artifacts_and_parquet():
    """
    Primary path:
      - RGCN vectors + payload from graph.pt + rgcn_problem_embeddings.npy
      - question / solution vectors from the embedded parquet (if available)
    """
    import torch, pandas as pd

    if not C.GRAPH_PATH.exists() or not C.EMB_NPY.exists():
        raise FileNotFoundError(
            f"Need {C.GRAPH_PATH} and {C.EMB_NPY} -- "
            f"run build_graph.py + train_rgcn.py first.")

    graph = torch.load(C.GRAPH_PATH, weights_only=False)
    rgcn  = np.load(C.EMB_NPY).astype(np.float32)
    ids   = graph["problem_ids"]
    meta  = graph.get("problem_meta", [{} for _ in ids])
    feats = graph["problem_features"].numpy().astype(np.float32)  # QS 1792-d

    # build payloads
    payloads = []
    for i, pid in enumerate(ids):
        pl = _clean_payload(meta[i])
        pl.setdefault("problem_id", pid)
        payloads.append(pl)

    # full = QS + RGCN
    full = []
    for i in range(len(ids)):
        v = np.concatenate([feats[i], rgcn[i]]).astype(np.float32)
        n = np.linalg.norm(v)
        full.append(v / (n or 1.0))

    # question + solution from parquet
    q_vecs  = [None] * len(ids)
    s_vecs  = [None] * len(ids)
    qs_vecs = list(feats)   # already have QS from graph features

    if C.INPUT_PARQUET.exists():
        df = pd.read_parquet(C.INPUT_PARQUET)
        pid_to_row = {str(r["problem_id"]): r for _, r in df.iterrows()}
        for i, pid in enumerate(ids):
            row = pid_to_row.get(str(pid))
            if row is None:
                continue
            q_vecs[i] = _as_vec(row.get("question_embedding"))
            s_vecs[i] = _as_vec(row.get("solution_embedding"))
        n_q = sum(1 for v in q_vecs if v is not None)
        n_s = sum(1 for v in s_vecs if v is not None)
        print(f"[OK] parquet: {n_q} question vecs, {n_s} solution vecs")
    else:
        print(f"[!] {C.INPUT_PARQUET} not found -- question/solution collections will be empty")

    return ids, payloads, {
        "question":  q_vecs,
        "solution":  s_vecs,
        "rgcn":      list(rgcn),
        "full":      full,
    }


def from_parquet_only(path: Path):
    """Fallback: load everything from the rgcn parquet artifact."""
    import pandas as pd
    df = pd.read_parquet(path)
    ids, payloads = [], []
    vecs = {"question": [], "solution": [], "rgcn": [], "full": []}
    for _, row in df.iterrows():
        if _as_vec(row.get("rgcn_embedding")) is None:
            continue
        ids.append(str(row.get("problem_id", "")))
        payloads.append(_clean_payload(
            {k: row[k] for k in df.columns if k not in EMBED_KEYS}))
        for key, col in [("question", "question_embedding"),
                         ("solution", "solution_embedding"),
                         ("rgcn",     "rgcn_embedding"),
                         ("full",     "full_embedding")]:
            vecs[key].append(_as_vec(row.get(col)))
    print(f"[->] {len(ids)} rows from {path.name}")
    return ids, payloads, vecs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url",         default=C.QDRANT_URL)
    ap.add_argument("--rgcn-collection",    default=C.QDRANT_COLLECTION_RGCN)
    ap.add_argument("--full-collection",    default=C.QDRANT_COLLECTION_FULL)
    ap.add_argument("--question-collection",default="problems_question")
    ap.add_argument("--solution-collection",default="problems_solution")
    ap.add_argument("--from-parquet",  action="store_true")
    ap.add_argument("--no-rgcn",       action="store_true")
    ap.add_argument("--no-full",       action="store_true")
    ap.add_argument("--no-question",   action="store_true")
    ap.add_argument("--no-solution",   action="store_true")
    args = ap.parse_args()

    print("\n" + "=" * 62)
    print("  RGCN STEP 4 -- QDRANT INGEST")
    print("=" * 62)

    if args.from_parquet:
        ids, payloads, vecs = from_parquet_only(C.OUTPUT_PARQUET)
    else:
        ids, payloads, vecs = from_artifacts_and_parquet()

    print(f"[->] {len(ids)} problems to ingest")

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=args.qdrant_url,
                              api_key=C.QDRANT_API_KEY, timeout=30)
        client.get_collections()
    except Exception as e:
        print(f"[X] cannot reach Qdrant at {args.qdrant_url}: "
              f"{e.__class__.__name__}: {str(e)[:120]}")
        print("    start it:  docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(1)

    if not args.no_question:
        ingest_embeddings(client, args.question_collection,
                          ids, vecs["question"], payloads)
    if not args.no_solution:
        ingest_embeddings(client, args.solution_collection,
                          ids, vecs["solution"], payloads)
    if not args.no_rgcn:
        ingest_embeddings(client, args.rgcn_collection,
                          ids, vecs["rgcn"], payloads)
    if not args.no_full:
        ingest_embeddings(client, args.full_collection,
                          ids, vecs["full"], payloads)

    print("\n" + "=" * 62)
    print("  QDRANT INGEST COMPLETE")
    print("  Collections:")
    for col in [args.question_collection, args.solution_collection,
                args.rgcn_collection, args.full_collection]:
        try:
            cnt = client.count(collection_name=col).count
            dim = client.get_collection(col).config.params.vectors.size
            print(f"    {col:<25} {cnt} points  dim={dim}")
        except Exception:
            pass
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
