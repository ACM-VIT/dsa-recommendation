from fastapi import FastAPI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from sentence_transformers import SentenceTransformer
from bkt import process_submission, problem_to_topics
from pydantic import BaseModel
from typing import Optional
app = FastAPI()
client = QdrantClient(path="/Users/shraddhasidhan/rag-from-scratch/LANG /qdrant_storage")
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.get("/candidates")
def get_candidates(topic: str, min_difficulty: int = 1, max_difficulty: int = 3, limit: int = 10):
    query_vector = model.encode(topic).tolist()

    results = client.query_points(
        collection_name="problems",
        query=query_vector,
        limit=limit,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="difficulty_score",
                    range=Range(gte=min_difficulty, lte=max_difficulty)
                )
            ]
        )
    ).points

    return [
        {
            "question": r.payload["question"],
            "difficulty": r.payload["difficulty"],
            "tags": r.payload["tags"],
            "skill_types": r.payload["skill_types"],
            "source": r.payload["source"],
            "url": r.payload["url"],
            "score": r.score
        }
        for r in results
    ]

# In memory mastery store — replace with database later
user_mastery_store = {}

class Submission(BaseModel):
    userId: str
    problemId: str
    verdict: str
    testCasesPassed: int
    totalTestCases: int
    hintsUsed: int
    submissionCount: int
    normalisedScore: float
    timestamp: float

@app.post("/update_bkt")
def update_bkt_endpoint(submission: Submission):
    
    # Get current mastery for this user
    current_mastery = user_mastery_store.get(submission.userId, {})
    
    # Process submission
    updated_mastery, mastered_topics, results = process_submission(
        submission.dict(),
        current_mastery
    )

    
    # Save updated mastery
    user_mastery_store[submission.userId] = updated_mastery
    
    return {
        "userId": submission.userId,
        "problemId": submission.problemId,
        "results": results,
        "mastered_topics": mastered_topics,
        "updated_mastery": updated_mastery
    }

@app.get("/mastery/{user_id}")
def get_mastery(user_id: str):
    mastery = user_mastery_store.get(user_id, {})
    return {
        "userId": user_id,
        "mastery": mastery,
        "mastered_topics": [t for t, v in mastery.items() if v >= 0.75]
    }