from fastapi import APIRouter
from controllers.mastery_controller import handle_get_mastery, handle_get_urgency

router = APIRouter()

@router.get("/mastery/{user_id}")
def get_mastery(user_id: str):
    return handle_get_mastery(user_id)

@router.get("/urgency/{user_id}")
def get_urgency(user_id: str):
    return handle_get_urgency(user_id)