"""
run_full_pipeline.py

ONE file that runs the whole thing, offline through online, taking a JSON
REQUEST and returning a JSON RESPONSE.

CLOUD-READY: every Qdrant and Neo4j credential is imported directly from
db_env.py (the one canonical .env-loading module at repo root) -- nothing
is hardcoded, nothing is re-parsed here. See db_env.py's docstring for the
full list of supported .env keys.

REQUEST (stdin, or --input-json <file>, or CLI flags for quick manual runs):
    {
      "user_id":      "string, required",
      "k":             10,
      "total_n":       30,
      "session_mode":  "practice",
      "submission":    { ... optional Submission-schema object, see below ... }
    }

    If "submission" is present, it's run through StateUpdateService (BKT +
    HLR + canonical UserGraph write-through -- the same path POST /update
    uses) BEFORE recommendations are generated, so this one script
    demonstrates the full submission -> state-update -> recommend loop, not
    just recommend-on-existing-state. "submission" must match the Submission
    schema (models/schemas/submission.py): userId, problemId, verdict,
    hintsUsed, testCasesPassed, totalTestCases, submissionCount,
    normalisedScore, problemDifficulty (optional 0-1), problemTopics.
    submission.userId is ignored in favor of the top-level "user_id" if both
    are present (top-level is authoritative, matching how the API's own
    routes work -- POST /update takes userId from the body, GET /recommend
    takes it from the path).

RESPONSE (stdout -- ALWAYS pure JSON, nothing else on stdout):
    {
      "user_id": "string",
      "session_mode": "practice" | "learning" | null,
      "state_update": { ... same shape as POST /update's response, only
                         present if "submission" was in the request ... },
      "recommendations": [
        {
          "problem_id":       "string",
          "title":            "string",
          "title_slug":       "string",
          "difficulty_score": 0.42,
          "topic_tags":       ["arrays", "hash_map"],
          "source":           "course_path",
          "recommended_at":   "2026-07-08T12:00:00+00:00"
        }
      ]
    }

All progress/status output goes to STDERR, never stdout.

Usage:
    echo '{"user_id": "u1", "k": 5}' | python run_full_pipeline.py
    python run_full_pipeline.py --input-json request.json
    python run_full_pipeline.py u1 --k 5

    # Full submission -> state-update -> recommend loop, see postman/
    # pipeline_requests/submission_then_recommend.json for a ready-made
    # example:
    python run_full_pipeline.py --input-json postman/pipeline_requests/submission_then_recommend.json

Flags:
    --input-json FILE  Read the JSON request from this file instead of stdin/CLI.
    --force-offline     Re-run the offline pipeline even if Qdrant already looks populated.
    --skip-offline      Never run the offline pipeline, even if empty.
    --no-neo4j          Skip Neo4j durable storage even if reachable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import db_env   # the ONE place .env gets loaded -- see db_env.py

QDRANT_URL     = db_env.QDRANT_URL
QDRANT_API_KEY = db_env.QDRANT_API_KEY
MANIFEST_PATH  = "data/1000_manifest_final.json"
REQUIRED_COLLECTIONS = ["dsa_problems", "problems_full"]


def _qdrant_offline_step_cmd():
    """Embedder subprocess call -- includes the API key flag if one is set."""
    cmd = [
        sys.executable, "pipeline/embeddings/embedder.py",
        "--resume", "--collection", "dsa_problems",
        "--qdrant-url", QDRANT_URL,
    ]
    if QDRANT_API_KEY:
        cmd += ["--qdrant-api-key", QDRANT_API_KEY]
    return cmd


TOPIC_EDGES_PATH = "question-graph/data/problem_topic_edges_normalized.json"

OFFLINE_STEPS = [
    ("Ingestion", [
        sys.executable, "pipeline/ingestion/ingest.py",
        "--input", MANIFEST_PATH,
    ]),
]

# Only auto-generate the topic-edges file if it doesn't already exist --
# if you're using a curated version (e.g. Aashray's), this step is
# skipped entirely so it's never overwritten. Same "skip if already
# done" logic as the Qdrant collection check further down.
if not Path(TOPIC_EDGES_PATH).exists():
    OFFLINE_STEPS.append((
        "Generate problem-topic edges (for bkt.py/hlr.py)",
        [sys.executable, "pipeline/ingestion/generate_topic_edges.py"],
    ))

OFFLINE_STEPS += [
    ("Embeddings", _qdrant_offline_step_cmd()),
    ("Graph build + RGCN train + Qdrant ingest", [
        sys.executable, "pipeline/graphs/run_rgcn_pipeline.py",
        "--graph-source", "normalized",
        # build_graph.py / ingest_rgcn_to_qdrant.py import config.py, which
        # imports db_env.py -- no CLI flag needed, they pick up the same
        # credentials this process already loaded.
    ]),
]


def _log(msg: str):
    """All status/progress output goes to stderr -- stdout is reserved for the final JSON response."""
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Step 1: offline pipeline, run-once-only
# ---------------------------------------------------------------------------

def offline_pipeline_already_done(qdrant_url: str) -> bool:
    try:
        client = db_env.qdrant_client(timeout=5)
    except ImportError:
        _log("[!] qdrant-client not installed -- cannot check offline state, "
             "assuming offline pipeline has NOT been run.")
        return False

    try:
        for name in REQUIRED_COLLECTIONS:
            info = client.get_collection(name)
            count = client.count(collection_name=name).count
            if count == 0:
                _log(f"[->] Collection '{name}' exists but is empty.")
                return False
        return True
    except Exception as exc:
        _log(f"[->] Offline pipeline check failed ({exc.__class__.__name__}) "
             f"-- assuming it hasn't been run yet.")
        return False


def run_offline_pipeline():
    """Run ingestion -> embeddings -> RGCN pipeline, once, in order. Stops on first failure."""
    _log("\n" + "=" * 64)
    _log("  OFFLINE PIPELINE -- running (this only happens once)")
    _log("=" * 64)

    for i, (label, cmd) in enumerate(OFFLINE_STEPS, 1):
        _log(f"\n[{i}/{len(OFFLINE_STEPS)}] {label}")
        _log(f"    $ {' '.join(cmd)}")
        # subprocess.run() inherits this process's os.environ by default --
        # every credential db_env.py loaded from .env is automatically
        # visible to each offline-stage subprocess.
        result = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
        if result.returncode != 0:
            _log(f"\n[X] FAILED at step {i}/{len(OFFLINE_STEPS)}: {label}")
            _log(f"    Exit code: {result.returncode}")
            sys.exit(result.returncode)

    _log("\n[OK] Offline pipeline complete.\n")


def ensure_offline_pipeline(force: bool, skip: bool):
    if skip:
        _log("[->] --skip-offline set; assuming Qdrant is already populated.")
        return
    if force:
        run_offline_pipeline()
        return
    if offline_pipeline_already_done(QDRANT_URL):
        _log("[OK] Offline pipeline already done -- Qdrant is populated. "
             "Skipping ingestion/embedding/training (use --force-offline to re-run anyway).")
        return
    run_offline_pipeline()


# ---------------------------------------------------------------------------
# Step 2: Neo4j -- automatic connection via db_env.py, graceful fallback
# ---------------------------------------------------------------------------

def build_neo4j_store(disabled: bool):
    from pipeline.recommender.services.neo4j_graph_store import Neo4jGraphStore

    if disabled:
        _log("[->] --no-neo4j set; skipping durable graph storage.")
        return Neo4jGraphStore(driver=None)

    driver = db_env.neo4j_driver()
    if driver is None:
        _log(f"[!] Neo4j unavailable (not configured, or unreachable at "
             f"{db_env.NEO4J_URI}) -- durable graph storage disabled for this run.")
        return Neo4jGraphStore(driver=None)

    _log(f"[OK] Neo4j connected at {db_env.NEO4J_URI} "
         f"(instance: {db_env.NEO4J_INSTANCENAME or '?'}, "
         f"database: {db_env.NEO4J_DATABASE}) -- user graph "
         f"will persist durably across runs.")
    return Neo4jGraphStore(driver, database=db_env.NEO4J_DATABASE)


# ---------------------------------------------------------------------------
# Step 3: title resolution
# ---------------------------------------------------------------------------

def resolve_problem_ids(problem_ids: list, qdrant, collection: str = "problems_full") -> dict:
    """
    Resolve problem_id strings -> Qdrant payload using scroll() with a
    payload filter. retrieve() by point ID doesn't work here because the
    internal Qdrant point IDs are raw integers assigned at ingest time,
    not the xxhash values the old code was computing.
    """
    if not problem_ids:
        return {}
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        points, _ = qdrant.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="problem_id", match=MatchAny(any=problem_ids))
            ]),
            limit=len(problem_ids),
            with_payload=True,
            with_vectors=False,
        )
        return {
            p.payload["problem_id"]: p.payload
            for p in points
            if p.payload and "problem_id" in p.payload
        }
    except Exception as exc:
        _log(f"[!] Could not resolve problem titles: {exc}")
        return {}


def resolve_recommendations(result_dict: dict, qdrant, collection: str = "problems_full") -> dict:
    recs = result_dict.get("recommendations", [])
    problem_ids = [r["problem_id"] for r in recs if "problem_id" in r and not r.get("title")]
    payloads = resolve_problem_ids(problem_ids, qdrant, collection=collection)

    enriched_recs = []
    for r in recs:
        pid = r.get("problem_id")
        enriched = dict(r)
        if not enriched.get("title"):
            payload = payloads.get(pid, {})
            enriched["title"] = payload.get("title", "(not found in Qdrant)")
            enriched["title_slug"] = payload.get("title_slug")
        enriched_recs.append(enriched)

    out = dict(result_dict)
    out["recommendations"] = enriched_recs
    return out


# ---------------------------------------------------------------------------
# Core: request dict -> response dict
# ---------------------------------------------------------------------------

class _InProcessCache:
    """
    Minimal dict-backed stand-in for the .get/.setex/.delete interface
    UserGraphService expects from a Redis client. Shared between the
    submission step and the recommendation step below so a submission
    processed earlier in THIS SINGLE RUN is visible to the recommendation
    step that follows it, even when no real Redis is configured and Neo4j
    is unreachable (e.g. no outbound network access). This is NOT a
    substitute for real Redis in a multi-process deployment -- it only
    lives for the duration of one `python run_full_pipeline.py` invocation
    -- just enough to make this one script's submission -> recommend demo
    self-consistent regardless of infra availability.
    """
    def __init__(self):
        self._store: dict = {}

    def get(self, key):
        return self._store.get(key)

    def setex(self, key, ttl, value):
        self._store[key] = value

    def delete(self, key):
        self._store.pop(key, None)


def apply_submission(user_id: str, submission: dict, neo4j_store, qdrant, redis, db=None) -> dict:
    """
    Route an optional "submission" through StateUpdateService -- the same
    canonical path POST /update uses (BKT + HLR update, written through to
    the UserGraph via Redis + Neo4j). Returns the same shape
    submission_controller.py::handle_update returns, so this script's
    "state_update" field matches the real API response exactly.

    db: optional Postgres bootstrap-read session (see
    database/postgres/db.py::get_user_graph_session) -- same fallback
    role it plays in submission_controller.py. None (the default) just
    means the user builds cold if Redis/Neo4j also have nothing for them.
    """
    from pipeline.recommender.services.state_update_service import StateUpdateService
    from pipeline.recommender.services.user_graph_service import UserGraphService

    graph_service = UserGraphService(db=db, redis=redis, neo4j=neo4j_store)
    service = StateUpdateService(graph_service, qdrant=qdrant)

    submission_dict = dict(submission)
    submission_dict.setdefault("verdict", "OK")
    submission_dict["userId"] = user_id   # top-level user_id is authoritative

    result = service.process_submission(user_id, submission_dict, rebuild_vector=False)

    updated_topics = [
        {
            "topicId": t,
            "updatedMastery": result.updated_mastery.get(t),
            "updatedHlr": result.updated_hlr.get(t),
        }
        for t in result.updated_topics
    ]
    return {
        "userId": user_id,
        "problemId": str(submission_dict.get("problemId", "")),
        "updatedTopics": updated_topics,
        "masteredTopics": result.newly_mastered,
        "results": {"bkt": result.bkt_results, "hlr": result.hlr_results},
    }


def handle_request(request: dict) -> dict:
    user_id = request.get("user_id")
    if not user_id:
        raise ValueError("request must include \"user_id\"")

    k = request.get("k", 10)
    total_n = request.get("total_n", 30)
    session_mode = request.get("session_mode")
    submission = request.get("submission")
    force_offline = request.get("force_offline", False)
    skip_offline = request.get("skip_offline", False)
    no_neo4j = request.get("no_neo4j", False)

    ensure_offline_pipeline(force=force_offline, skip=skip_offline)
    neo4j_store = build_neo4j_store(disabled=no_neo4j)

    from pipeline.recommender.services.recommend import get_recommendations

    qdrant = db_env.qdrant_client(timeout=10)

    # Shared for the duration of this run only -- see _InProcessCache's
    # docstring. If Neo4j is reachable, its durable state is used too (and
    # would carry across separate invocations); this cache just guarantees
    # same-run consistency regardless of that.
    in_process_cache = _InProcessCache()

    # Postgres bootstrap-read session for UserGraphService's cold-start
    # fallback (same role as submission_controller.py/recommendation_
    # controller.py's get_user_graph_session() usage) -- without this, a
    # user with real Postgres history but no reachable Neo4j/no prior
    # in-process cache entry always builds cold, and _fetch_user's
    # AttributeError-on-None gets logged every run ("Failed to fetch user
    # ...: 'NoneType' object has no attribute 'execute'"). None if
    # DATABASE_URL isn't set or the connection fails -- same graceful
    # degrade as everywhere else.
    from database.postgres.db import get_user_graph_session
    db = get_user_graph_session()

    try:
        state_update_response = None
        if submission is not None:
            _log(f"\n[->] Processing submission for problemId={submission.get('problemId')} "
                 f"before generating recommendations...")
            state_update_response = apply_submission(
                user_id, submission, neo4j_store, qdrant, redis=in_process_cache, db=db)
            _log(f"[OK] State update complete: "
                 f"{len(state_update_response['updatedTopics'])} topic(s) touched, "
                 f"{len(state_update_response['masteredTopics'])} newly mastered.")

        # BKT/HLR stores are legacy get_recommendations() params, kept for
        # backward compatibility -- the canonical read path is now UserGraph
        # (via neo4j_store + in_process_cache above), which apply_submission()
        # just wrote through to if a submission was processed this run.
        bkt_store: dict = {}
        hlr_store: dict = {}

        result = get_recommendations(
            user_id=user_id, db=db, redis=in_process_cache, neo4j=neo4j_store, qdrant=qdrant,
            bkt_store=bkt_store, hlr_store=hlr_store,
            collection="problems_full", total_n=total_n, k=k,
        )
    finally:
        if db is not None:
            db.close()

    response = result.to_dict()
    response = resolve_recommendations(response, qdrant, collection="problems_full")
    response["session_mode"] = session_mode
    if state_update_response is not None:
        response["state_update"] = state_update_response
    return response


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_request(args) -> dict:
    if args.input_json:
        with open(args.input_json, encoding="utf-8") as f:
            return json.load(f)

    if args.user_id:
        req = {"user_id": args.user_id, "k": args.k, "total_n": args.total_n}
        if args.session_mode:
            req["session_mode"] = args.session_mode
        req["force_offline"] = args.force_offline
        req["skip_offline"] = args.skip_offline
        req["no_neo4j"] = args.no_neo4j
        return req

    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        if raw:
            return json.loads(raw)

    raise ValueError(
        "No request provided. Pass a user_id positionally, use --input-json "
        "FILE, or pipe a JSON request via stdin.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user_id", nargs="?", default=None)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--total-n", type=int, default=30)
    parser.add_argument("--session-mode", default=None, choices=["practice", "learning"])
    parser.add_argument("--force-offline", action="store_true")
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--no-neo4j", action="store_true")
    args = parser.parse_args()

    if args.force_offline and args.skip_offline:
        _log("[X] --force-offline and --skip-offline are mutually exclusive.")
        sys.exit(1)

    try:
        request = _load_request(args)
        response = handle_request(request)
    except Exception as exc:
        print(json.dumps({"error": str(exc), "error_type": exc.__class__.__name__}))
        sys.exit(1)

    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()