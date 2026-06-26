from pipeline.recommender.bkt import process_submission
from pipeline.recommender.hlr import process_hlr
from database.postgres.db import get_user_mastery, save_user_mastery, get_user_hlr, save_user_hlr
import threading

_locks = {}
_lock_guard = threading.Lock()

def get_user_lock(user_id):
    with _lock_guard:
        if user_id not in _locks:
            _locks[user_id] = threading.Lock()
        return _locks[user_id]

def handle_update_bkt(submission):
    with get_user_lock(submission.userId):
        current_mastery = get_user_mastery(submission.userId)
        updated_mastery, mastered_topics, results = process_submission(
            submission.model_dump(), current_mastery
        )
        save_user_mastery(submission.userId, updated_mastery)
    return updated_mastery, mastered_topics, results

def handle_update_hlr(submission):
    with get_user_lock(submission.userId):
        current_hlr = get_user_hlr(submission.userId)
        updated_hlr, results = process_hlr(
            submission.model_dump(), current_hlr
        )
        save_user_hlr(submission.userId, updated_hlr)
    return updated_hlr, results