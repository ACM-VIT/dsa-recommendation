"""
postman/demo_progression.py

Chains all the telemetry samples in postman/telemetry_samples/ through
StateUpdateService within ONE process, so effects that require state to
persist across submissions (like BKT's trivial-difficulty dampening, or
just "does mastery keep moving the way you'd expect") are reliably
demonstrable regardless of whether Redis/Neo4j are reachable from this
machine.

This is NOT the same as replaying the same JSON files against POST /update
in Postman one at a time -- submission_controller.py's live route only
persists state durably if Neo4j is configured and reachable (there's no
Redis wired into the controllers at all, see README caveat). Running
against the real HTTP API from Postman with an unreachable Neo4j means
every /update call starts from a fresh, empty graph. This script sidesteps
that by sharing one in-memory cache across every step, so you can see the
full intended behavior even without live infra.

Run:
    python postman/demo_progression.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db_env
from pipeline.recommender.services.state_update_service import StateUpdateService
from pipeline.recommender.services.user_graph_service import UserGraphService

SAMPLES_DIR = Path(__file__).parent / "telemetry_samples"

# (scenario label, filename, one-line description of what it demonstrates)
STEPS = [
    ("1. Cold-start easy solve", "01_cold_start_easy_solve.json",
     "First-ever submission for this user, at low difficulty: expect mastery "
     "to jump by less than MAX_MASTERY_DELTA (the cap itself scales down for "
     "low difficulty), not the much larger raw Bayesian delta and not a flat "
     "cap regardless of difficulty -- compare to step 6's high-difficulty solve."),
    ("2. Struggling, many attempts", "02_struggling_many_attempts.json",
     "8 submissions, 4 hints, but still verdict=OK: expect a visibly lower "
     "confidence (telemetry.py) than a clean solve, dampening the gain."),
    ("3. Wrong answer", "03_wrong_answer.json",
     "verdict=WRONG_ANSWER: expect the telemetry signal capped at <=0.35 and "
     "no BKT learning-transition credit (delta from Bayesian blending only)."),
    ("4. Moderate-difficulty clean solve", "04_moderate_difficulty_clean_solve.json",
     "Difficulty ~0.5 (neutral difficulty_credit ~1.0x): a clean baseline "
     "solve with no difficulty-credit adjustment either way."),
    ("5a. Warm-up before trivial solve", "05a_warmup_before_trivial.json",
     "Raises this user's 'array' mastery so step 5b has something to be "
     "trivial RELATIVE to."),
    ("5b. Trivial-difficulty solve (dampened)", "05b_trivial_relative_to_mastery.json",
     "Same user, same 'array' topic, problemDifficulty=0.05 well below "
     "their now-higher mastery: expect a SMALLER delta than step 1's "
     "otherwise-identical inputs, despite a perfect score -- this is the "
     "trivial-solve dampening in bkt.py::update_bkt."),
    ("6. Hard stretch solve", "06_hard_stretch_solve.json",
     "problemDifficulty=0.9: expect the strongest difficulty_credit "
     "(telemetry.py, up to 1.3x) AND the highest effective cap (bkt.py, "
     "also up to 1.3x MAX_MASTERY_DELTA) -- the largest delta of any "
     "scenario here, not capped down to the same value as step 1's easy solve."),
]


class _InProcessCache:
    """Same minimal Redis stand-in as run_full_pipeline.py's -- shared across
    every step below so mastery/HLR state genuinely carries forward."""
    def __init__(self):
        self._store: dict = {}
    def get(self, key):
        return self._store.get(key)
    def setex(self, key, ttl, value):
        self._store[key] = value
    def delete(self, key):
        self._store.pop(key, None)


def main():
    qdrant = db_env.qdrant_client(timeout=10)
    cache = _InProcessCache()
    graph_service = UserGraphService(db=None, redis=cache, neo4j=None)
    service = StateUpdateService(graph_service, qdrant=qdrant)

    print("=" * 78)
    print("  DUOLINGO-STYLE PROGRESSION DEMO")
    print("  (chained in one process -- see this file's docstring for why)")
    print("=" * 78)

    for label, filename, note in STEPS:
        path = SAMPLES_DIR / filename
        submission = json.loads(path.read_text(encoding="utf-8"))
        user_id = submission["userId"]

        # snapshot mastery BEFORE, for a clean before/after print
        graph_before = service._get_or_create_graph(user_id)
        before = {s: e.mastery_score for s, e in graph_before.concept_edges.items()}

        result = service.process_submission(user_id, submission, rebuild_vector=False)

        print(f"\n--- {label} ---")
        print(f"    {note}")
        print(f"    user={user_id}  problemId={submission['problemId']}  "
              f"verdict={submission['verdict']}  difficulty={submission.get('problemDifficulty')}")
        for r in result.bkt_results:
            topic = r["topic"]
            prev = before.get(topic, r["previous_p_l"])
            print(f"    topic={topic:20s} mastery {r['previous_p_l']:.4f} -> "
                  f"{r['new_p_l']:.4f}  (delta={r['new_p_l']-r['previous_p_l']:+.4f}, "
                  f"observed_score={r['observed_score']:.4f})")
        if result.newly_mastered:
            print(f"    newly mastered: {result.newly_mastered}")

    print("\n" + "=" * 78)
    print("  Done. Every delta above should be bounded by MAX_MASTERY_DELTA (0.12).")
    print("=" * 78)


if __name__ == "__main__":
    main()
