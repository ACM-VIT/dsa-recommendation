from fastapi import APIRouter
from controllers.mastery_controller import handle_get_mastery, handle_get_urgency
from models.schemas.responses import MasteryResponse, UrgencyResponse

router = APIRouter(tags=["mastery"])


@router.get(
    "/mastery/{user_id}",
    response_model=MasteryResponse,
    summary="Get a user's raw mastery and current proficiency per topic",
    description=(
        "Reads BKT mastery_score per topic (read-only, Postgres-backed). "
        "Also returns `proficiency`: mastery decayed by HLR retention -- "
        "the meaningful 'how good are they at this topic RIGHT NOW' weight "
        "(see UserGraph.effective_proficiency)."
    ),
)
def get_mastery(user_id: str):
    return handle_get_mastery(user_id)


@router.get(
    "/urgency/{user_id}",
    response_model=UrgencyResponse,
    summary="Get a user's HLR forgetting-curve urgency per topic",
    description="Higher urgency = the user is more likely to have forgotten this topic and it's due for review.",
)
def get_urgency(user_id: str):
    return handle_get_urgency(user_id)