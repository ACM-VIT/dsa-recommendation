from pydantic import BaseModel

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
