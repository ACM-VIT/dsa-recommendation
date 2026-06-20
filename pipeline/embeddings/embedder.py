"""
DSA Engine -- Embedding Generator
==================================
Generates semantic embeddings for the vector pool.

Pipeline per problem:
    Question  : title + explanation_text + tag phrases  -> BGE-Large    -> Q (1024-dim)
    Solution  : Tree-sitter -> AST -> CFG + DFG -> PDG  -> GraphCodeBERT -> S (768-dim)
    Combined  : concat(Q, S) re-normalised              -> QS (1792-dim)

Run from repo root:
    uv run pipeline/embeddings/embedder.py --input data/vector_pool/vector_pool.parquet --output data/vector_pool
    uv run pipeline/embeddings/embedder.py --input data/vector_pool/vector_pool.parquet --output data/vector_pool --resume
    uv run pipeline/embeddings/embedder.py --input data/vector_pool/vector_pool.parquet --output data/vector_pool \\
        --qdrant-url http://localhost:6333 --collection dsa_problems
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Repo root = two levels up from pipeline/embeddings/
_REPO_ROOT = Path(__file__).parent.parent.parent

# Allow importing code_analyzer from utils/
sys.path.insert(0, str(_REPO_ROOT / "utils"))

CHECKPOINT_EVERY = 100
_BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def build_question_text(row: dict) -> str:
    parts = []

    def _str(val):
        if val is None: return ""
        try: return str(val).strip()
        except: return ""

    def _to_list(val):
        if val is None: return []
        try: return list(val)
        except: return []

    title = _str(row.get("title"))
    if title:
        parts.append(title)

    exp = _str(row.get("explanation_text"))
    if exp and exp != 'nan':
        parts.append(exp[:800])

    tag_phrases = []
    topics   = _to_list(row.get("topic_tags"))
    algos    = _to_list(row.get("algorithm_tags"))
    patterns = _to_list(row.get("patterns"))
    if topics:   tag_phrases.append("Topics: "     + ", ".join(t.replace("_", " ") for t in topics))
    if algos:    tag_phrases.append("Algorithms: " + ", ".join(t.replace("_", " ") for t in algos))
    if patterns: tag_phrases.append("Patterns: "   + ", ".join(t.replace("_", " ") for t in patterns))
    if tag_phrases:
        parts.append(". ".join(tag_phrases))

    return " | ".join(parts)


def build_solution_text(row: dict) -> Optional[str]:
    sol = row.get("canonical_solution")
    if sol is None or (hasattr(sol, '__len__') and len(sol) == 0):
        return None
    try:
        sol = str(sol).strip()
    except Exception:
        return None
    if not sol or sol == 'nan':
        return None

    try:
        from code_analyzer import build_semantic_text
        enriched = build_semantic_text(sol)
        if enriched:
            return enriched[:1500]
    except Exception:
        pass

    return sol[:1500]


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

_question_model = None
_solution_model = None


def _load_question_model():
    global _question_model
    if _question_model is None:
        from sentence_transformers import SentenceTransformer
        print("[->] Loading BAAI/bge-large-en-v1.5 (question encoder)...")
        _question_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        print("[OK] Question model loaded")
    return _question_model


def _load_solution_model():
    global _solution_model
    if _solution_model is None:
        from sentence_transformers import SentenceTransformer
        print("[->] Loading microsoft/graphcodebert-base (solution encoder)...")
        _solution_model = SentenceTransformer("microsoft/graphcodebert-base")
        print("[OK] Solution model loaded")
    return _solution_model


# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------

def embed_questions(texts: List[str], batch_size: int = 32) -> np.ndarray:
    model = _load_question_model()
    prefixed = [_BGE_INSTRUCTION + t for t in texts]
    embs = model.encode(prefixed, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)
    return embs.astype(np.float32)


def embed_solutions(texts: List[str], batch_size: int = 32) -> np.ndarray:
    model = _load_solution_model()
    embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)
    return embs.astype(np.float32)


def concat_and_normalise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    combined = np.concatenate([a, b], axis=1).astype(np.float32)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return combined / norms


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(output_dir: str) -> Path:
    return Path(output_dir) / "vector_pool_checkpoint.parquet"


def save_checkpoint(df: pd.DataFrame, output_dir: str) -> None:
    path = _checkpoint_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    n_done = df["question_embedding"].notna().sum()
    print(f"  [checkpoint] {n_done}/{len(df)} rows saved -> {path}")


def load_checkpoint(output_dir: str) -> Optional[pd.DataFrame]:
    path = _checkpoint_path(output_dir)
    if path.exists():
        df = pd.read_parquet(path)
        n_done = df["question_embedding"].notna().sum()
        print(f"[->] Checkpoint found: {n_done}/{len(df)} rows already embedded")
        return df
    return None


# ---------------------------------------------------------------------------
# Main embedding loop
# ---------------------------------------------------------------------------

def embed_dataframe(
    df: pd.DataFrame,
    batch_size: int = 32,
    resume: bool = False,
    output_dir: str = None,
) -> pd.DataFrame:
    if output_dir is None:
        output_dir = str(_REPO_ROOT / "data" / "vector_pool")

    if resume:
        ckpt = load_checkpoint(output_dir)
        if ckpt is not None:
            df = ckpt

    print(f"[->] Embedding {len(df)} rows (batch_size={batch_size})")

    todo_idx = [i for i, v in enumerate(df["question_embedding"]) if v is None or
                (hasattr(v, '__len__') and len(v) == 0)]
    if not todo_idx:
        print("[OK] All rows already embedded")
        return df

    # -- Step 1: Question embeddings (BGE-Large) --
    print(f"\n[1/3] Building question texts...")
    q_texts = [build_question_text(df.iloc[i].to_dict()) for i in todo_idx]
    print(f"[1/3] Encoding question embeddings (BGE-Large)...")
    t0 = time.time()
    q_embs = embed_questions(q_texts, batch_size=batch_size)
    print(f"[OK] Questions done in {time.time()-t0:.1f}s  shape={q_embs.shape}")

    for df_row_i, local_i in enumerate(todo_idx):
        df.at[local_i, "question_embedding"] = q_embs[df_row_i]

    save_checkpoint(df, output_dir)

    # -- Step 2: Solution embeddings (GraphCodeBERT) --
    print(f"\n[2/3] Building semantic solution texts (AST->CFG/DFG->PDG)...")
    s_texts_raw = [build_solution_text(df.iloc[i].to_dict()) for i in todo_idx]
    has_sol = [t is not None for t in s_texts_raw]
    s_texts = [t if t else "" for t in s_texts_raw]

    try:
        import tree_sitter
        print("[OK] Tree-sitter available -- using AST/CFG/DFG/PDG enrichment")
    except ImportError:
        print("[!]  Tree-sitter not available -- using raw code fallback")

    print(f"[2/3] Encoding {len(s_texts)} solution embeddings (GraphCodeBERT)...")
    t0 = time.time()
    s_embs = embed_solutions(s_texts, batch_size=batch_size)
    print(f"[OK] Solutions done in {time.time()-t0:.1f}s  shape={s_embs.shape}")

    for df_row_i, local_i in enumerate(todo_idx):
        df.at[local_i, "solution_embedding"] = s_embs[df_row_i] if has_sol[df_row_i] else None

    save_checkpoint(df, output_dir)

    # -- Step 3: Concat embeddings --
    print(f"\n[3/3] Building concat embeddings (dim={q_embs.shape[1]}+{s_embs.shape[1]}={q_embs.shape[1]+s_embs.shape[1]})...")
    qs_embs_full = concat_and_normalise(q_embs, s_embs)
    print(f"[OK] Concat done  shape=(N, {qs_embs_full.shape[1]})")

    for df_row_i, local_i in enumerate(todo_idx):
        qs = qs_embs_full[df_row_i]
        df.at[local_i, "question_solution_embedding"] = None if np.any(np.isnan(qs)) else qs
        df.at[local_i, "rgcn_embedding"]  = None
        df.at[local_i, "full_embedding"]  = None

    save_checkpoint(df, output_dir)
    print("[OK] DataFrame updated")
    return df


# ---------------------------------------------------------------------------
# Save final output
# ---------------------------------------------------------------------------

def save_embedded(df: pd.DataFrame, output_dir: str) -> None:
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "vector_pool_embedded.parquet"
    df.to_parquet(path, index=False)
    print(f"[OK] Saved -> {path}  ({len(df)} rows)")

    n_q  = df["question_embedding"].notna().sum()
    n_s  = df["solution_embedding"].notna().sum()
    n_qs = df["question_solution_embedding"].notna().sum()

    q_sample  = df["question_embedding"].dropna()
    qs_sample = df["question_solution_embedding"].dropna()
    q_dim     = len(q_sample.iloc[0])  if len(q_sample)  > 0 else "?"
    qs_dim    = len(qs_sample.iloc[0]) if len(qs_sample) > 0 else "?"

    print(f"    question_embedding          : {n_q}/{len(df)}  dim={q_dim}")
    print(f"    solution_embedding          : {n_s}/{len(df)}  dim=768")
    print(f"    question_solution_embedding : {n_qs}/{len(df)}  dim={qs_dim}")
    print(f"    rgcn_embedding              : 0/{len(df)} (future -- needs RGCN training)")
    print(f"    full_embedding              : 0/{len(df)} (future -- needs rgcn_embedding)")

    ckpt = _checkpoint_path(output_dir)
    if ckpt.exists():
        ckpt.unlink()
        print(f"[OK] Checkpoint removed (no longer needed)")


# ---------------------------------------------------------------------------
# Qdrant upload
# ---------------------------------------------------------------------------

def upload_to_qdrant(df, url, collection, embedding_col="question_solution_embedding", batch_size=100):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct, OptimizersConfigDiff
    except ImportError:
        print("[X] qdrant-client not installed. Run: uv pip install qdrant-client")
        return

    try:
        client = QdrantClient(url=url, timeout=5)
        client.get_collections()
    except Exception as e:
        print(f"[X] Cannot connect to Qdrant at {url}: {e}")
        print("    Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print(f"    Then re-run: uv run pipeline/embeddings/embedder.py --input data/vector_pool/vector_pool_embedded.parquet --output data/vector_pool --qdrant-url {url} --collection {collection}")
        return

    sample_col = df[embedding_col].dropna()
    if len(sample_col) == 0:
        embedding_col = "question_embedding"
        sample_col    = df[embedding_col].dropna()

    vec_size = len(sample_col.iloc[0])
    print(f"[->] Qdrant at {url}  |  collection={collection}  |  dim={vec_size}")

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )
        print(f"[OK] Collection '{collection}' created")
    else:
        print(f"[->] Collection '{collection}' exists -- upserting")

    points  = []
    skipped = 0
    for i, (_, row) in enumerate(df.iterrows()):
        vec = row.get(embedding_col)
        if vec is None:
            vec = row.get("question_embedding")
        if vec is None:
            skipped += 1
            continue
        payload = {
            col: (row[col].tolist() if isinstance(row[col], np.ndarray) else row[col])
            for col in df.columns
            if col not in ("question_embedding", "solution_embedding", "rgcn_embedding",
                           "question_solution_embedding", "full_embedding")
            and not (isinstance(row[col], float) and np.isnan(row[col]))
        }
        points.append(PointStruct(id=i, vector=vec.tolist(), payload=payload))

    print(f"[->] Uploading {len(points)} points ({skipped} skipped)...")
    t0 = time.time()
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        client.upsert(collection_name=collection, points=batch)
        pct = min(100, int(100 * (start + len(batch)) / len(points)))
        print(f"\r  {pct}% ({start + len(batch)}/{len(points)})", end="", flush=True)

    client.update_collection(collection_name=collection,
                             optimizers_config=OptimizersConfigDiff(indexing_threshold=20000))
    print(f"\n[OK] Upload done in {time.time()-t0:.1f}s")
    print(f"[OK] Collection '{collection}' has {client.count(collection_name=collection).count} points")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DSA Engine -- Embedding Generator")
    p.add_argument("--input",      "-i",
                   default=str(_REPO_ROOT / "data" / "vector_pool" / "vector_pool.parquet"))
    p.add_argument("--output",     "-o",
                   default=str(_REPO_ROOT / "data" / "vector_pool"))
    p.add_argument("--batch-size", "-b", type=int, default=32)
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--qdrant-url",    default=None)
    p.add_argument("--collection",    default="dsa_problems")
    p.add_argument("--qdrant-vector", default="question_solution_embedding",
                   choices=["question_embedding", "solution_embedding", "question_solution_embedding"])
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*62)
    print("  DSA ENGINE -- EMBEDDING GENERATOR")
    print("="*62)

    path = Path(args.input)
    if not path.exists():
        print(f"[X] File not found: {path}")
        print("    Run ingestion pipeline first:")
        print("    uv run pipeline/ingestion/run_pipeline.py")
        sys.exit(1)

    print(f"[->] Loading {path}...")
    df = pd.read_parquet(path)
    print(f"[OK] {len(df)} rows x {len(df.columns)} columns loaded")

    already_done = df["question_embedding"].notna().all()
    if already_done and not args.resume:
        print("[OK] All embeddings already present -- skipping embedding step")
        n_q  = df["question_embedding"].notna().sum()
        n_s  = df["solution_embedding"].notna().sum()
        n_qs = df["question_solution_embedding"].notna().sum()
        print(f"    question_embedding          : {n_q}/{len(df)}")
        print(f"    solution_embedding          : {n_s}/{len(df)}")
        print(f"    question_solution_embedding : {n_qs}/{len(df)}")
        embedded_path = Path(args.output) / "vector_pool_embedded.parquet"
        if path.resolve() != embedded_path.resolve():
            save_embedded(df, args.output)
    else:
        df = embed_dataframe(df, batch_size=args.batch_size,
                             resume=args.resume, output_dir=args.output)
        save_embedded(df, args.output)

    if args.qdrant_url:
        print("\n[->] Uploading to Qdrant...")
        upload_to_qdrant(df, url=args.qdrant_url, collection=args.collection,
                         embedding_col=args.qdrant_vector)

    print("\n" + "="*62)
    print("  EMBEDDING COMPLETE")
    print(f"  Output: {(Path(args.output) / 'vector_pool_embedded.parquet').resolve()}")
    if args.qdrant_url:
        print(f"  Qdrant: {args.qdrant_url} / {args.collection}")
    print("="*62 + "\n")


if __name__ == "__main__":
    main()
