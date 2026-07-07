from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language maintenance question")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)


class SourceItem(BaseModel):
    acn: str
    aircraft_model: str
    flight_phase: str
    excerpt: str
    relevance_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]


class RULPredictionRequest(BaseModel):
    sensor_readings: List[List[float]] = Field(
        ...,
        description="30 timesteps x 14 sensor readings, most recent cycle last"
    )


class RULPredictionResponse(BaseModel):
    predicted_rul: float
    cycles_remaining: int
    warning_level: str