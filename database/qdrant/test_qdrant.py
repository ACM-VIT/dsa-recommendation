"""
DSA Engine -- Qdrant Search Tests
===================================
Tests the vector pool end-to-end after upload.

Run from repo root:
    uv run database/qdrant/test_qdrant.py
    uv run database/qdrant/test_qdrant.py --url http://localhost:6333 --collection dsa_problems
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Repo root = two levels up from database/qdrant/
_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_PARQUET = _REPO_ROOT / "data" / "vector_pool" / "vector_pool_embedded.parquet"


def load_client(url: str) -> QdrantClient:
    c = QdrantClient(url=url, timeout=10)
    c.get_collections()
    return c


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def test_collection_info(client, collection):
    print("\n" + "="*55)
    print("  TEST 1 -- Collection Info")
    print("="*55)
    info  = client.get_collection(collection)
    count = client.count(collection_name=collection).count
    print(f"  Points          : {count}")
    print(f"  Vector size     : {info.config.params.vectors.size}")
    print(f"  Distance metric : {info.config.params.vectors.distance}")
    print(f"  Status          : {info.status}")
    assert count > 0, "Collection is empty!"
    assert info.config.params.vectors.size == 1792, f"Expected 1792-dim, got {info.config.params.vectors.size}"
    print("  [PASS]")


def test_payload(client, collection):
    print("\n" + "="*55)
    print("  TEST 2 -- Payload Spot Check (first 3 points)")
    print("="*55)
    pts = client.retrieve(collection, ids=[0, 1, 2], with_payload=True, with_vectors=False)
    for p in pts:
        pl = p.payload
        title = pl.get("title", "?")
        pid   = pl.get("problem_id", "?")
        diff  = pl.get("difficulty_score", "?")
        tags  = pl.get("topic_tags", [])
        print(f"  [{p.id}] {title}")
        print(f"       problem_id : {pid}")
        print(f"       difficulty : {diff}")
        print(f"       topic_tags : {tags}")
        assert title, "title missing from payload"
        assert pid,   "problem_id missing from payload"
    print("  [PASS]")


def test_similarity_search(client, collection, df):
    print("\n" + "="*55)
    print("  TEST 3 -- Similarity Search")
    print("="*55)
    test_cases = [
        ("Two Sum",                                       "hash map / complement lookup"),
        ("Longest Substring Without Repeating Characters","sliding window"),
        ("Add Two Numbers",                               "linked list"),
    ]
    for title, expect in test_cases:
        row = df[df["title"] == title]
        if row.empty:
            print(f"  [SKIP] '{title}' not in parquet")
            continue
        query_vec = np.array(row.iloc[0]["question_solution_embedding"]).tolist()
        hits = client.query_points(collection_name=collection, query=query_vec,
                                   limit=6, with_payload=True).points
        print(f"\n  Query: '{title}'  (expect: {expect})")
        for i, h in enumerate(hits):
            marker = "  -->" if i == 0 else "     "
            print(f"{marker} [{h.score:.4f}] {h.payload.get('title','?')}")
        top = hits[0].payload.get("title", "")
        assert top == title, f"Self not top hit! Got: {top}"
    print("\n  [PASS]")


def test_filtered_search(client, collection, df):
    print("\n" + "="*55)
    print("  TEST 4 -- Filtered Search (difficulty + vector)")
    print("="*55)
    row = df[df["title"] == "Two Sum"]
    if row.empty:
        print("  [SKIP] Two Sum not found")
        return
    query_vec = np.array(row.iloc[0]["question_solution_embedding"]).tolist()
    hits = client.query_points(
        collection_name=collection, query=query_vec,
        query_filter=Filter(must=[FieldCondition(key="difficulty_score", range=Range(lte=0.45))]),
        limit=5, with_payload=True,
    ).points
    print(f"  Similar to 'Two Sum' filtered to difficulty <= 0.45:")
    for h in hits:
        diff = h.payload.get("difficulty_score", "?")
        print(f"     [{h.score:.4f}] (diff={diff:.3f}) {h.payload.get('title','?')}")
        assert float(diff) <= 0.45, f"Filter broke! Got difficulty={diff}"
    print("  [PASS]")


def test_cross_topic_transfer(client, collection, df):
    print("\n" + "="*55)
    print("  TEST 5 -- Cross-Topic Transfer (pattern similarity)")
    print("="*55)
    query_title = "Longest Substring Without Repeating Characters"
    row = df[df["title"] == query_title]
    if row.empty:
        print("  [SKIP]")
        return
    query_vec = np.array(row.iloc[0]["question_solution_embedding"]).tolist()
    hits = client.query_points(collection_name=collection, query=query_vec,
                               limit=10, with_payload=True).points
    sw_hits = [h for h in hits if
               "sliding_window" in (h.payload.get("topic_tags") or []) or
               "sliding_window" in (h.payload.get("patterns") or []) or
               "sliding_window" in (h.payload.get("algorithm_tags") or [])]
    print(f"  Query: '{query_title}'")
    print(f"  Top 10 hits with sliding_window tag: {len(sw_hits)}/10")
    for h in hits[:5]:
        print(f"     [{h.score:.4f}] {h.payload.get('title','?')}  tags={h.payload.get('topic_tags',[])}")
    assert len(sw_hits) >= 2, f"Expected >= 2 sliding_window hits in top 10, got {len(sw_hits)}"
    print("  [PASS]")


def test_random_probe(client, collection, df):
    print("\n" + "="*55)
    print("  TEST 6 -- Random Vector Probe (10 random points)")
    print("="*55)
    sample = df.sample(10, random_state=42)
    for _, row in sample.iterrows():
        vec = row["question_solution_embedding"]
        assert vec is not None, f"NULL vector for {row['title']}"
        arr = np.array(vec)
        assert arr.shape == (1792,), f"Wrong shape {arr.shape} for {row['title']}"
        assert not np.any(np.isnan(arr)), f"NaN in vector for {row['title']}"
        assert not np.all(arr == 0), f"Zero vector for {row['title']}"
    print(f"  10 random vectors: all 1792-dim, no NaN, no zeros")
    print("  [PASS]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url",        default="http://localhost:6333")
    p.add_argument("--collection", default="dsa_problems")
    p.add_argument("--parquet",    default=str(_DEFAULT_PARQUET))
    args = p.parse_args()

    print("\n" + "="*55)
    print("  DSA ENGINE -- QDRANT SEARCH TESTS")
    print(f"  {args.url} / {args.collection}")
    print("="*55)

    client = load_client(args.url)
    df     = load_parquet(args.parquet)

    passed = failed = 0
    tests = [
        ("Collection info",      lambda: test_collection_info(client, args.collection)),
        ("Payload spot check",   lambda: test_payload(client, args.collection)),
        ("Similarity search",    lambda: test_similarity_search(client, args.collection, df)),
        ("Filtered search",      lambda: test_filtered_search(client, args.collection, df)),
        ("Cross-topic transfer", lambda: test_cross_topic_transfer(client, args.collection, df)),
        ("Random probe",         lambda: test_random_probe(client, args.collection, df)),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            failed += 1

    print("\n" + "="*55)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
