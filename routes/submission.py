from fastapi import APIRouter
from models.schemas.submission import Submission
from models.schemas.responses import UpdateResponse
from controllers.submission_controller import handle_update

router = APIRouter(tags=["submission"])


@router.post(
    "/update",
    response_model=UpdateResponse,
    summary="Record a graded submission and update learner state",
    description=(
        "Runs BKT + HLR on the submitted problem's topics and persists the "
        "result to the canonical UserGraph (Redis + Neo4j), with a "
        "best-effort write-through to Postgres for the legacy read-only "
        "mastery/urgency endpoints. Mastery moves by at most "
        "bkt.MAX_MASTERY_DELTA per call -- small, incremental changes, not "
        "one-shot jumps."
    ),
)
def update_endpoint(submission: Submission):
    return handle_update(submission)