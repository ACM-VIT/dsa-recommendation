from fastapi import FastAPI
from routes.submission import router as submission_router
from routes.mastery import router as mastery_router
from routes.recommendation import router as recommendation_router

app = FastAPI()

app.include_router(submission_router)
app.include_router(mastery_router)
app.include_router(recommendation_router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Recommendation Service"
    }