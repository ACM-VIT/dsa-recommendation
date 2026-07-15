"""
Recommendation controller.

Reads the user's state from the canonical UserGraph (Redis -> Neo4j),
kept fresh by StateUpdateService on every submission -- see
submission_controller.py. The previous per-request get_user_mastery/
get_user_hlr pre-fetch was removed as dead work (get_recommendations()
already builds its own graph via UserGraphService and never consumed
those pre-fetched values directly) -- that's a DIFFERENT thing from the
Postgres bootstrap read below, which UserGraphService itself performs
(via `db=`) only as a fallback for a user with no Redis/Neo4j state yet.
"""

import logging
import os

from fastapi import HTTPException

import db_env
from database.postgres.db import get_user_graph_session
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
_redis_client = None
_redis_checked = False


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


def _get_redis():
    """Lazy singleton -- see submission_controller.py::_get_redis() for the
    identical pattern/rationale. Both controllers share one canonical
    UserGraph cache (same Redis keys, same TTL) once REDIS_URL is set."""
    global _redis_client, _redis_checked
    if not _redis_checked:
        _redis_client = db_env.redis_client()
        _redis_checked = True
    return _redis_client


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handle_recommend(user_id: str, limit: int = 10) -> dict:
    """
    Full ML pipeline recommendation.

    Builds a UserGraph (Redis -> Neo4j -- see UserGraphService), runs the
    candidate pool pipeline, returns a ranked list shaped to the backend's
    RecommendationLog schema.

    ML never WRITES to Postgres. `db` is only used by UserGraphService as
    a read-only bootstrap fallback: if a user has no Redis/Neo4j state yet
    (e.g. they were seeded/migrated directly into Postgres, or Redis+Neo4j
    both expired/are unreachable), it seeds the graph from their real
    Submission/UserTopicMastery/RecommendationLog/ConceptGapProfile rows
    instead of silently building an empty cold-start graph for a user who
    actually has history. None if DATABASE_URL isn't set or the connection
    fails -- same graceful degrade as everything else here.
    """
    db_session = get_user_graph_session()
    try:
        result = get_recommendations(
            user_id=user_id,
            db=db_session,
            redis=_get_redis(),
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
    finally:
        if db_session is not None:
            db_session.close()