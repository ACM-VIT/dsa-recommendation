from fastapi import APIRouter, Query
from controllers.recommendation_controller import handle_recommend
from models.schemas.responses import RecommendResponse

MAX_RECOMMENDATIONS = 50
MIN_RECOMMENDATIONS = 1

router = APIRouter(tags=["recommendation"])


@router.get(
    "/recommend/{user_id}",
    response_model=RecommendResponse,
    summary="Get a ranked recommendation slate for a user",
    description=(
        "Runs the full candidate-pool -> filtering -> ranking -> diversity "
        "pipeline for user_id and returns up to `limit` recommendations, "
        "shaped to the backend's RecommendationLog schema."
    ),
)
async def recommend(
    user_id: str,
    limit: int = Query(
        10, ge=MIN_RECOMMENDATIONS, le=MAX_RECOMMENDATIONS,
        description="Max number of recommendations to return (1-50).",
    ),
):
    limit = max(MIN_RECOMMENDATIONS, min(limit, MAX_RECOMMENDATIONS))
    return handle_recommend(user_id, limit)