import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    Enum,
    ForeignKey,
    UniqueConstraint,
    BigInteger,
    Numeric,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db import Base
from app.enums import (
    TrackStatus,
    AssetType,
    PublishChannel,
    PublishContentType,
    PublishJobStatus,
    CompilationStatus,
)
from app.config import settings


def get_json_type():
    if settings.DATABASE_URL.startswith("sqlite"):
        return JSON
    return JSONB


def utcnow():
    return datetime.now(timezone.utc)


class Track(Base):
    __tablename__ = "tracks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_ref = Column(String, nullable=True, index=True, unique=True)
    brand = Column(String, nullable=False)
    series = Column(String, nullable=True)
    title = Column(String, nullable=True)
    slug = Column(String, nullable=True, unique=True)
    prompt = Column(Text, nullable=False)
    genre = Column(String, nullable=False)
    mood = Column(String, nullable=False)
    bpm = Column(Integer, nullable=True)
    duration_sec = Column(Integer, nullable=True)
    qc_score = Column(Float, nullable=True)
    status = Column(Enum(TrackStatus), nullable=False, default=TrackStatus.raw)
    generation_provider = Column(String, nullable=True)
    image_provider = Column(String, nullable=True)

    audio_master_key = Column(String, nullable=True)
    audio_preview_key = Column(String, nullable=True)
    audio_loop_key = Column(String, nullable=True)
    cover_key = Column(String, nullable=True)
    waveform_video_key = Column(String, nullable=True)

    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    approved_at = Column(DateTime, nullable=True)
    rejected_at = Column(DateTime, nullable=True)
    published_at = Column(DateTime, nullable=True)

    assets = relationship("Asset", back_populates="track")
    short_videos = relationship("ShortVideo", back_populates="track")
    product_page = relationship("ProductPage", back_populates="track", uselist=False)


class Asset(Base):
    __tablename__ = "assets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id = Column(UUID(as_uuid=True), ForeignKey("tracks.id"), nullable=False)
    asset_type = Column(Enum(AssetType), nullable=False)
    storage_key = Column(String, nullable=False)
    mime_type = Column(String, nullable=True)
    byte_size = Column(BigInteger, nullable=True)
    checksum_sha256 = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    track = relationship("Track", back_populates="assets")


class Compilation(Base):
    __tablename__ = "compilations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False)
    slug = Column(String, nullable=True, unique=True)
    description = Column(Text, nullable=True)
    source_track_ids_json = Column(get_json_type(), nullable=False)
    audio_key = Column(String, nullable=True)
    video_key = Column(String, nullable=True)
    cover_key = Column(String, nullable=True)
    status = Column(
        Enum(CompilationStatus), nullable=False, default=CompilationStatus.draft
    )
    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    published_at = Column(DateTime, nullable=True)


class PublishingJob(Base):
    __tablename__ = "publishing_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id = Column(UUID(as_uuid=True), ForeignKey("tracks.id"), nullable=True)
    compilation_id = Column(
        UUID(as_uuid=True), ForeignKey("compilations.id"), nullable=True
    )
    channel = Column(Enum(PublishChannel), nullable=False)
    content_type = Column(Enum(PublishContentType), nullable=False)
    status = Column(
        Enum(PublishJobStatus), nullable=False, default=PublishJobStatus.queued
    )
    scheduled_for = Column(DateTime, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    idempotency_key = Column(String, nullable=False, unique=True)
    external_post_id = Column(String, nullable=True)
    external_url = Column(String, nullable=True)
    request_payload_json = Column(get_json_type(), nullable=True)
    response_payload_json = Column(get_json_type(), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)


class ShortVideo(Base):
    __tablename__ = "short_videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id = Column(UUID(as_uuid=True), ForeignKey("tracks.id"), nullable=False)
    variant_index = Column(Integer, nullable=False)
    storage_key = Column(String, nullable=False)
    duration_sec = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint("track_id", "variant_index", name="uix_track_variant"),
    )
    track = relationship("Track", back_populates="short_videos")


class ProductPage(Base):
    __tablename__ = "product_pages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id = Column(
        UUID(as_uuid=True), ForeignKey("tracks.id"), unique=True, nullable=False
    )
    cms_page_id = Column(String, nullable=True)
    cms_url = Column(String, nullable=True)
    store_product_id = Column(String, nullable=True)
    pricing_json = Column(get_json_type(), nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    track = relationship("Track", back_populates="product_page")


class AnalyticsDaily(Base):
    __tablename__ = "analytics_daily"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id = Column(UUID(as_uuid=True), ForeignKey("tracks.id"), nullable=True)
    compilation_id = Column(
        UUID(as_uuid=True), ForeignKey("compilations.id"), nullable=True
    )
    channel = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)
    views = Column(BigInteger, nullable=True)
    watch_time_seconds = Column(Float, nullable=True)
    likes = Column(BigInteger, nullable=True)
    comments = Column(BigInteger, nullable=True)
    clicks = Column(BigInteger, nullable=True)
    revenue_estimate = Column(Numeric(12, 2), nullable=True)
    sales_count = Column(Integer, nullable=True)
    captured_at = Column(DateTime, nullable=False)


class RunLog(Base):
    __tablename__ = "run_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pipeline_name = Column(String, nullable=False)
    stage_name = Column(String, nullable=False)
    entity_id = Column(UUID(as_uuid=True), nullable=True)
    severity = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    context_json = Column(get_json_type(), nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)


class SchedulerLock(Base):
    __tablename__ = "scheduler_locks"

    name = Column(String, primary_key=True)
    locked_until = Column(DateTime, nullable=False)
    owner_id = Column(String, nullable=False)
