from pydantic import BaseModel, ConfigDict
from typing import Optional
from uuid import UUID
from datetime import datetime
from app.enums import TrackStatus, PublishJobStatus


class TrackBase(BaseModel):
    brand: str
    series: Optional[str] = None
    title: Optional[str] = None
    slug: Optional[str] = None
    prompt: str
    genre: str
    mood: str
    bpm: Optional[int] = None
    duration_sec: Optional[int] = None


class TrackResponse(TrackBase):
    id: UUID
    status: TrackStatus
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class JobResponse(BaseModel):
    id: UUID
    track_id: Optional[UUID] = None
    channel: str
    content_type: str
    status: PublishJobStatus
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class MetricSummary(BaseModel):
    entity_id: UUID
    views: int
    watch_time_seconds: float
    sales_count: int
    score: float
