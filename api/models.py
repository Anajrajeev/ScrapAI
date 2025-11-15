"""API data models."""

from pydantic import BaseModel
from typing import Dict, List, Optional


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    detections: List[Dict]
    recyclability: Dict
    inference_time: float


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    message: str
