import logging

import db_env
from pipeline.recommender.services.neo4j_graph_store import Neo4jGraphStore
from pipeline.recommender.services.state_update_service import StateUpdateService
from pipeline.recommender.services.user_graph_service import UserGraphService

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singletons -- same pattern as recommendation_controller.py, so both
# controllers share the same startup cost profile (no client is built until
# the first request needs it).
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
        _neo4j_store = Neo4jGraphStore(driver, database=db_env.NEO4J_DATABASE)
    return _neo4j_store


def _topics_to_updated_list(touched_topics, updated_mastery, updated_hlr):
    """
    Reshape into updatedTopics: [{topicId, updatedMastery, updatedHlr}] --
    restricted to `touched_topics` (the topics this submission actually
    affected), NOT every key in updated_mastery/updated_hlr. Those two dicts
    now carry the user's full current state (read off the canonical
    UserGraph, see state_update_service.py), not just this problem's
    topics, so keying off their full keyset would leak every topic the
    user has ever touched into every response.
    """
    return [
        {
            "topicId": t,
            "updatedMastery": updated_mastery.get(t),
            "updatedHlr": updated_hlr.get(t),
        }
        for t in touched_topics
    ]


def handle_update(submission):
    """
    Runs the full state-update pipeline for a graded submission: BKT + HLR
    update, then persists the result to the canonical UserGraph (Redis +
    Neo4j), with a best-effort Postgres write-through for the legacy
    read-only mastery/urgency endpoints.

    Request/response shape is unchanged from the previous stateless
    version -- the only behavioral difference is that what's computed here
    now actually gets persisted instead of being discarded after the
    response is sent.
    """
    graph_service = UserGraphService(
        db=None, redis=None, neo4j=_get_neo4j_store(),
    )
    service = StateUpdateService(graph_service, qdrant=_get_qdrant())

    result = service.process_submission(
        submission.userId, submission.model_dump(), rebuild_vector=False,
    )

    updated_topics = _topics_to_updated_list(
        result.updated_topics, result.updated_mastery, result.updated_hlr,
    )

    return {
        "userId": submission.userId,
        "problemId": submission.problemId,
        "updatedTopics": updated_topics,
        "masteredTopics": result.newly_mastered,
        "results": {"bkt": result.bkt_results, "hlr": result.hlr_results},
    }
