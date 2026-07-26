"""
training/catalog_metadata.py

Loads the real static problem-catalog metadata --
company_tag_count/frequency/rating/asked_by_faang, as
training/feature_registry.py already declares them -- from
data/1000_manifest_final.json, the real, git-tracked raw manifest
(2913 records) that pipeline/ingestion/ingest.py's build_vector_pool()
transforms into the Qdrant vector pool. This is the exact same file/schema
feature_registry.py's company_tag_count/frequency/rating/asked_by_faang
entries already cite as their source_file -- reusing it here, not
inventing a new source or fabricating values.

IMPORTANT: the manifest's own "problem_id" (this repo's internal
ingestion hash) is NOT the backend's problem.problem_id (a CUID) --
confirmed empirically: zero overlap between the two id sets. The join key
both sides actually agree on is `title_slug` (e.g. "two-sum"), exactly as
database.postgres.db.resolve_problem_ids_by_title_slugs()'s own docstring
already documents for the identical mismatch elsewhere in this codebase.
That existing resolver is reused here rather than re-implemented --
confirmed empirically to translate 1157/1159 (99.8%) of the real active
Postgres catalog's title_slugs.

Reuses:
  - pipeline.ingestion.ingest.load_manifest() for reading the manifest
    (the same parser the offline ingestion pipeline itself uses)
  - database.postgres.db.resolve_problem_ids_by_title_slugs() for the
    title_slug -> real problem_id translation (the same resolver
    save_recommendation_log() already uses for this exact mismatch)

Requires a live Postgres connection (for the slug resolution), so --
mirroring synthetic_user_generator.py's load_real_concept_graph()
convention for the same reason -- this is meant to be called explicitly
by real generation/serving code, not by offline-testable unit tests
(which inject a small fake catalog_metadata dict instead, the same way
they already inject a fake cc_edges dict).

Cached at module level: parsed/resolved once per process. A problem
missing from the manifest, or whose title_slug doesn't resolve, simply
isn't in the returned dict -- training/feature_extractor.py's
extract_problem_features() already treats a missing/absent
catalog_metadata entry as "no data available" (native NaN), exactly as
if this loader didn't exist. Nothing here fabricates a value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pipeline.ingestion.ingest import load_manifest

from training.config import CATALOG_MANIFEST_PATH

_CATALOG_METADATA_CACHE: Optional[dict[str, dict]] = None


def load_catalog_metadata_by_problem_id(manifest_path: Path = CATALOG_MANIFEST_PATH) -> dict[str, dict]:
    """
    Returns {real_problem_id: {"companies": [...] | None, "frequency": float | None,
    "rating": float | None, "asked_by_faang": bool | None}} -- exactly the
    shape training/feature_extractor.py::extract_problem_features()'s
    `catalog_metadata` parameter already expects
    (meta.get("companies")/"frequency"/"rating"/"asked_by_faang"), keyed by
    the SAME problem_id namespace MergedCandidate.problem_id uses (the
    backend's problem.problem_id), not the manifest's own internal id.

    Requires a live Postgres connection for the title_slug -> problem_id
    resolution step (database.postgres.db.resolve_problem_ids_by_title_slugs).
    Returns {} (not a fabricated per-problem entry) if the manifest file is
    missing/unparseable, matching every other "gracefully degrade to no
    data" convention this pipeline already uses.
    """
    global _CATALOG_METADATA_CACHE
    if _CATALOG_METADATA_CACHE is not None:
        return _CATALOG_METADATA_CACHE

    try:
        records = load_manifest(str(manifest_path))
    except (FileNotFoundError, ValueError, OSError):
        _CATALOG_METADATA_CACHE = {}
        return _CATALOG_METADATA_CACHE

    metadata_by_slug = {
        record["title_slug"]: {
            "companies": record.get("companies"),
            "frequency": record.get("frequency"),
            "rating": record.get("rating"),
            "asked_by_faang": record.get("asked_by_faang"),
        }
        for record in records if record.get("title_slug")
    }

    from database.postgres.db import resolve_problem_ids_by_title_slugs
    problem_id_by_slug = resolve_problem_ids_by_title_slugs(list(metadata_by_slug.keys()))

    _CATALOG_METADATA_CACHE = {
        problem_id: metadata_by_slug[slug]
        for slug, problem_id in problem_id_by_slug.items()
    }
    return _CATALOG_METADATA_CACHE


def reset_catalog_metadata_cache() -> None:
    """Test-only hook: forces the next load_catalog_metadata_by_problem_id()
    call to re-read/re-resolve instead of returning the cached dict."""
    global _CATALOG_METADATA_CACHE
    _CATALOG_METADATA_CACHE = None
