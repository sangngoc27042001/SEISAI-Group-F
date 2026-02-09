"""
User Feedback Router
Collects user information and feedback after prediction, stores to MongoDB.
"""

import os
from datetime import datetime, timezone

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field
from pymongo import MongoClient

feedback_router = APIRouter(prefix="/api/v1", tags=["feedback"])

# MongoDB connection (lazy init)
_mongo_client: MongoClient | None = None


def _get_collection():
    global _mongo_client
    if _mongo_client is None:
        uri = os.environ.get("MONGODB_URI", "")
        if not uri:
            raise HTTPException(status_code=500, detail="MONGODB_URI is not configured")
        _mongo_client = MongoClient(uri)
    db = _mongo_client["seisai"]
    return db["user_input_logs"]


class UserFeedbackInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: EmailStr
    had_stroke: bool = Field(..., description="Whether the user has experienced a stroke")
    comment: str = Field("", max_length=2000, description="Free-text feedback")
    prediction_input: Optional[dict[str, Any]] = Field(None, description="Symptom selections the user submitted")
    prediction_result: Optional[dict[str, Any]] = Field(None, description="AI prediction results returned to the user")


@feedback_router.post("/feedback")
def submit_feedback(feedback: UserFeedbackInput) -> dict:
    """Store user feedback to MongoDB."""
    try:
        collection = _get_collection()
        doc = {
            "name": feedback.name,
            "email": feedback.email,
            "had_stroke": feedback.had_stroke,
            "comment": feedback.comment,
            "prediction_input": feedback.prediction_input,
            "prediction_result": feedback.prediction_result,
            "created_at": datetime.now(timezone.utc),
        }
        collection.insert_one(doc)
        return {"status": "ok", "message": "Thank you for your feedback!"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")
