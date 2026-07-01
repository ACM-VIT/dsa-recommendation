"""
DSA Engine — Vector Pool Schema
=================================
Single source of truth for every column in the vector pool DataFrame.
All other files import from here — never hardcode column names elsewhere.
"""

# ── Universal join key ───────────────────────────────────────────────────────
JOIN_KEY = "problem_id"

# ── Core identity columns (100% coverage expected) ───────────────────────────
IDENTITY_COLUMNS = [
    "problem_id",       # str  — CUID, universal join key across all artifacts
    "title",            # str  — problem title
    "title_slug",       # str  — URL slug e.g. "two-sum"  (69% coverage)
]

# ── Content columns ──────────────────────────────────────────────────────────
CONTENT_COLUMNS = [
    "description",          # str  — cleaned prose built from explanation_text / markdown
    "explanation_text",     # str  — raw explanation text from dataset (68.9% coverage)
    "solution_signature",   # str  — first def/class line extracted from canonical solution
    "canonical_solution",   # str|None — best available Python solution (see priority below)
]

# ── Difficulty & engagement signals ─────────────────────────────────────────
SIGNAL_COLUMNS = [
    "difficulty_score",     # float [0,1]  — derived from rating + frequency
    "frequency",            # float [0,1]  — how often this problem appears in interviews (50.4%)
    "rating",               # float [0,1]  — like ratio  likes/(likes+dislikes)  (50.4%)
    "likes",                # int          — raw like count (100% coverage)
    "dislikes",             # int          — raw dislike count (100% coverage)
    "asked_by_faang",       # bool         — FAANG flag (100% coverage)
]

# ── Tag columns (all inferred via tag_inference.py) ─────────────────────────
TAG_COLUMNS = [
    "topic_tags",           # List[str] — broad topic: arrays, trees, graphs, dp …
    "algorithm_tags",       # List[str] — algorithm: binary_search, bfs, backtracking …
    "data_structure_tags",  # List[str] — structure used: hash_map, stack, heap …
    "patterns",             # List[str] — Blind-75-style: sliding_window, two_pointers …
    "techniques",           # List[str] — impl technique: memoization, dummy_node …
    "skill_tags",           # List[str] — union of all above (for ConceptGapProfile)
]

# ── Relational columns ───────────────────────────────────────────────────────
RELATIONAL_COLUMNS = [
    "similar_problem_ids",  # List[str] — cleaned slugs from similar_questions
    "companies",            # List[str] — companies that asked this (50.4% coverage)
]

# ── Embedding columns — populated by embedder.py ─────────────────────────────
#
# question_embedding         1024-dim float32
#   Model : BAAI/bge-large-en-v1.5
#   Input : BGE instruction prefix + title + explanation_text + tag phrases
#   Why   : Top-ranked on MTEB retrieval benchmarks. Instruction prefix
#            boosts asymmetric search quality by ~3-5 NDCG@10 points.
#   Note  : bge-large outputs 1024-dim (bge-base is 768-dim — different model)
#
# solution_embedding         768-dim float32
#   Model : microsoft/graphcodebert-base
#   Input : canonical_solution via Tree-sitter AST->CFG+DFG->PDG (first 1500 chars)
#   Why   : Pre-trained on code + docstrings across 6 languages. Captures
#            structural/semantic patterns text models miss.
#   Note  : None for problems with no Python solution (SQL/Shell)
#
# rgcn_embedding             dim TBD
#   Model : RGCN (not yet trained)
#   Input : graph neighbourhood of problem node in Neo4j
#   Note  : None until graph + RGCN training complete
#
# question_solution_embedding  1792-dim float32  (1024 + 768)
#   Derived : concat(question_embedding, solution_embedding), L2-normalised
#   PRIMARY Qdrant vector — combines problem semantics + solution structure
#   Note  : None until both Q and S embeddings are populated
#
# full_embedding             dim TBD  (1792 + rgcn_dim)
#   Derived : concat(question, solution, rgcn), L2-normalised
#   Note  : None until rgcn_embedding exists
#
EMBEDDING_COLUMNS = [
    "question_embedding",           # 1024-dim | BAAI/bge-large-en-v1.5
    "solution_embedding",           # 768-dim  | microsoft/graphcodebert-base
    "rgcn_embedding",               # 128-dim  | RGCN (pipeline/graphs/)
    "question_solution_embedding",  # 1792-dim | concat(Q 1024 + S 768), L2-normed
    "full_embedding",               # 1920-dim | concat(QS 1792 + RGCN 128), L2-normed
]

# ── Final ordered column list (this is the DataFrame column order) ───────────
FINAL_COLUMNS = (
    IDENTITY_COLUMNS
    + CONTENT_COLUMNS
    + SIGNAL_COLUMNS
    + TAG_COLUMNS
    + RELATIONAL_COLUMNS
    + EMBEDDING_COLUMNS
)

# ── Canonical solution priority chain ────────────────────────────────────────
# 1. python_solution       — community-voted, highest quality signal
# 2. python_solution_eri   — ERI reference solution
# 3. community_solutions   — highest upvoted among community posts
# 4. None                  — no Python solution exists (SQL/Shell problems)
#                            → will be rejected by validator gate G4

# ── Columns stripped from final DataFrame (used only during preprocessing) ───
EXCLUDE_COLUMNS = {
    # Raw solution sources — merged into canonical_solution, then dropped
    "python_solution",
    "python_solution_eri",
    "python_solution_source",
    "python_solution_upvotes",
    "community_solutions",
    # Non-Python solutions — out of scope for this pipeline
    "java_solution",
    "cpp_solution",
    "javascript_solution",
    # Redundant / low-value
    "full_markdown_solution",   # used only to build description, then dropped
    "lc_id",                    # empty for all 2913 records
    "discuss_count",            # capped at 999 for every record — zero signal
    "has_solution",             # redundant: G4 gate catches missing solutions
    "similar_questions",        # raw messy flat list → cleaned into similar_problem_ids
}
