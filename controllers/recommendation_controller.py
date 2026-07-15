"""
Recommendation controller.

Reads the user's state from the canonical UserGraph (Redis -> Neo4j),
kept fresh by StateUpdateService on every submission -- see
submission_controller.py. No Postgres access here at all: the previous
per-request get_user_mastery/get_user_hlr pre-fetch was dead work, since
get_recommendations() already builds its own graph via UserGraphService
and never consumed those pre-fetched values directly.
"""

import logging
import os

from fastapi import HTTPException

import db_env
from pipeline.recommender.services.neo4j_graph_store import Neo4jGraphStore
from pipeline.recommender.services.recommend import get_recommendations

log = logging.getLogger(__name__)

# FIX (Greptile P1 "Collection Override Is Removed"): this was hardcoded
# to "problems_full", silently dropping the QDRANT_COLLECTION env
# override. Any deployment using a versioned/environment-specific
# collection name would query the wrong one. Restored the override,
# same default value so nothing changes for the current setup.
COLLECTION = os.environ.get("QDRANT_COLLECTION", "problems_full")

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_qdrant_client = None
_neo4j_store = None


def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = db_env.qdrant_client(timeout=10)
    return _qdrant_client


def _get_neo4j_store():
    global _neo4j_store
    if _neo4j_store is None:
        driver = db_env.neo4j_driver()
        _neo4j_store = Neo4jGraphStore(
            driver, database=db_env.NEO4J_DATABASE
        )
    return _neo4j_store


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handle_recommend(user_id: str, limit: int = 10) -> dict:
    """
    Full ML pipeline recommendation.

    Builds a UserGraph (Redis -> Neo4j -- see UserGraphService), runs the
    candidate pool pipeline, returns a ranked list shaped to the backend's
    RecommendationLog schema.

    db=None -- ML never writes to Postgres. UserGraphService falls back to
    an empty cold-start graph if a user has no Redis/Neo4j state yet (e.g.
    their very first request, before any submission has been processed by
    StateUpdateService).
    """
    try:
        result = get_recommendations(
            user_id=user_id,
            db=None,
            redis=None,
            neo4j=_get_neo4j_store(),
            qdrant=_get_qdrant(),
            collection=COLLECTION,
            total_n=max(limit * 3, 30),
            k=limit,
        )
        return result.to_dict()

    except Exception as exc:
        log.error(
            "Recommendation pipeline failed for user %s: %s",
            user_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Recommendation engine encountered an error",
        )