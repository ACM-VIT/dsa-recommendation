from fastapi import APIRouter
from models.schemas.submission import Submission
from controllers.submission_controller import handle_update_bkt, handle_update_hlr

router = APIRouter()

@router.post("/update_bkt")
def update_bkt_endpoint(submission: Submission):
    updated_mastery, mastered_topics, results = handle_update_bkt(submission)
    return {
        "userId": submission.userId,
        "problemId": submission.problemId,
        "results": results,
        "mastered_topics": mastered_topics,
        "updated_mastery": updated_mastery
    }

@router.post("/update_hlr")
def update_hlr_endpoint(submission: Submission):
    updated_hlr, results = handle_update_hlr(submission)
    return {
        "userId": submission.userId,
        "problemId": submission.problemId,
        "results": results,
        "updated_hlr": updated_hlr
    }
