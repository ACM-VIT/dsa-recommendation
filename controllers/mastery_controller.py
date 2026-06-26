from pipeline.recommender.hlr import calculate_urgency
from datetime import datetime, timezone
from pipeline.recommender.bkt import MASTERY_THRESHOLD
from database.postgres.db import get_user_mastery, get_user_hlr

def handle_get_mastery(user_id):
    mastery = get_user_mastery(user_id)
    return {
        "userId": user_id,
        "mastery": mastery,
        "mastered_topics": [t for t, v in mastery.items() if v >= MASTERY_THRESHOLD]
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