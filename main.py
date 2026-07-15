from dotenv import load_dotenv
from routes.seeding import router as seeding_router
import os
load_dotenv()


from fastapi import FastAPI
from routes.submission import router as submission_router
from routes.mastery import router as mastery_router
from routes.recommendation import router as recommendation_router

app = FastAPI(
    title="DSA Recommendation Service",
    description=(
        "Online recommendation engine: BKT/HLR learner-state updates, "
        "candidate-pool generation, and heuristic ranking for a DSA "
        "problem recommender. Import /openapi.json directly into Postman, "
        "or use the checked-in postman_collection.json at the repo root."
    ),
    version="1.0.0",
)

app.include_router(submission_router)
app.include_router(mastery_router)
app.include_router(recommendation_router)
app.include_router(seeding_router)
@app.get("/")
async def root():
    return {
        "message": "Welcome to Recommendation Service"
    }