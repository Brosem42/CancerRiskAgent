from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

@router.post("/chat")

class Chat