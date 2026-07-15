from pipeline.recommender.hlr import calculate_urgency, recall_probability, parse_aware_datetime, MIN_HALF_LIFE
from datetime import datetime, timezone
from pipeline.recommender.bkt import MASTERY_THRESHOLD
from database.postgres.db import get_user_mastery, get_user_hlr


def _topic_proficiency(mastery_score: float, hlr_state: dict, now_ts: float) -> float:
    """
    Mastery decayed by HLR recall probability -- same formula as
    UserGraph.effective_proficiency (pipeline/recommender/models/user_graph.py),
    mirrored here for this endpoint's Postgres-backed read path rather than
    the live graph. No HLR data yet for this topic -> raw mastery (nothing
    to decay from).
    """
    if not hlr_state or not hlr_state.get("last_review"):
        return round(mastery_score, 4)
    last_review_dt = parse_aware_datetime(hlr_state["last_review"])
    now_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    days_since = (now_dt - last_review_dt).total_seconds() / 86400
    half_life = hlr_state.get("half_life") or MIN_HALF_LIFE
    recall = recall_probability(half_life, days_since)
    return round(mastery_score * recall, 4)


def handle_get_mastery(user_id):
    mastery = get_user_mastery(user_id)
    hlr_state = get_user_hlr(user_id)
    now_ts = datetime.now(timezone.utc).timestamp()
    proficiency = {
        topic: _topic_proficiency(score, hlr_state.get(topic), now_ts)
        for topic, score in mastery.items()
    }
    return {
        "userId": user_id,
        "mastery": mastery,
        "mastered_topics": [t for t, v in mastery.items() if v >= MASTERY_THRESHOLD],
        # Current real proficiency (mastery decayed by retention) alongside
        # raw historical mastery -- the meaningful "how good are they at
        # this topic RIGHT NOW" weight. See _topic_proficiency above.
        "proficiency": proficiency,
    }

def handle_get_urgency(user_id):
    hlr_state = get_user_hlr(user_id)
    current_time = datetime.now(timezone.utc).timestamp()
    urgency_scores = {
        topic: calculate_urgency(state, current_time)
        for topic, state in hlr_state.items()
    }
    return {
        "userId": user_id,
        "urgency_scores": urgency_scores
    }