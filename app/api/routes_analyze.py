"""HTTP route for submission analysis."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import AnalyzeCallable, get_analyze_submission
from app.logging.logger import bind_submission_id, reset_submission_id
from app.models.request_schemas import AnalyzeRequest
from app.models.response_schemas import AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    analyzer: Annotated[AnalyzeCallable, Depends(get_analyze_submission)],
) -> AnalyzeResponse:
    """Analyze a completed code submission."""

    print("Route reached")
    token = bind_submission_id(request.submission_id)
    try:
        return await analyzer(request)
    finally:
        reset_submission_id(token)
