"""
Response schemas for FastAPI's `response_model=` on every typed route.

These exist purely for OpenAPI/Swagger/Postman fidelity -- each model
matches its handler's EXISTING return dict shape exactly, field for field.
This is important, not cosmetic: FastAPI serializes a handler's return
value THROUGH response_model, so a model missing a field would silently
strip that field from the real HTTP response. Every model below was built
by reading the actual handler code, not guessed.

Seeding routes (routes/seeding.py) are deliberately NOT given a
response_model here -- their return shape genuinely varies by branch (user
not found / no linked handle / provider error / success), and forcing a
single rigid schema would either misrepresent real responses or require
inventing fields that don't reflect what the handler actually returns.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# POST /update  (controllers/submission_controller.py::handle_update)
# ---------------------------------------------------------------------------

class TopicUpdate(BaseModel):
    topicId: str
    updatedMastery: Optional[float] = Field(None, description="New BKT P(L) for this topic, 0-1")
    updatedHlr: Optional[dict] = Field(None, description="New HLR state dict for this topic")


class BktTopicResult(BaseModel):
    topic: str
    previous_p_l: float
    new_p_l: float
    mastered: bool
    observed_score: float


class HlrTopicResult(BaseModel):
    topic: str
    performance: float
    previous_half_life: float
    new_half_life: float
    p_recall: float
    next_review_days: float


class UpdateResults(BaseModel):
    bkt: List[BktTopicResult]
    hlr: List[HlrTopicResult]


class UpdateResponse(BaseModel):
    userId: str
    problemId: str
    updatedTopics: List[TopicUpdate]
    masteredTopics: List[str]
    results: UpdateResults


# ---------------------------------------------------------------------------
# GET /recommend/{user_id}  (controllers/recommendation_controller.py::handle_recommend)
# ---------------------------------------------------------------------------

class RecommendationItem(BaseModel):
    problem_id: str
    title: Optional[str] = None
    title_slug: Optional[str] = None
    difficulty_score: Optional[float] = None
    topic_tags: List[str] = []
    source: str = Field(..., description="One of: graph_walk, vector_similarity, revision, sm2_review, course_path")
    recommended_at: str = Field(..., description="ISO 8601 timestamp")


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendationItem]


# ---------------------------------------------------------------------------
# GET /mastery/{user_id}  (controllers/mastery_controller.py::handle_get_mastery)
# ---------------------------------------------------------------------------

class MasteryResponse(BaseModel):
    userId: str
    mastery: Dict[str, float] = Field(..., description="Raw BKT mastery_score per topic, 0-1")
    mastered_topics: List[str]
    proficiency: Dict[str, float] = Field(
        ..., description="Mastery decayed by HLR retention -- current real "
                          "proficiency per topic, 0-1 (see UserGraph.effective_proficiency)"
    )


# ---------------------------------------------------------------------------
# GET /urgency/{user_id}  (controllers/mastery_controller.py::handle_get_urgency)
# ---------------------------------------------------------------------------

class UrgencyResponse(BaseModel):
    userId: str
    urgency_scores: Dict[str, float] = Field(..., description="HLR urgency per topic, 0-1")
