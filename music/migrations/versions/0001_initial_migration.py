"""Initial migration

Revision ID: 0001
Revises:
Create Date: 2024-06-15 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Tracks
    op.create_table('tracks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('external_ref', sa.String(), nullable=True),
        sa.Column('brand', sa.String(), nullable=False),
        sa.Column('series', sa.String(), nullable=True),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('slug', sa.String(), nullable=True),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('genre', sa.String(), nullable=False),
        sa.Column('mood', sa.String(), nullable=False),
        sa.Column('bpm', sa.Integer(), nullable=True),
        sa.Column('duration_sec', sa.Integer(), nullable=True),
        sa.Column('qc_score', sa.Float(), nullable=True),
        sa.Column('status', sa.Enum('raw', 'approved', 'rejected', 'published', name='trackstatus'), nullable=False),
        sa.Column('generation_provider', sa.String(), nullable=True),
        sa.Column('image_provider', sa.String(), nullable=True),
        sa.Column('audio_master_key', sa.String(), nullable=True),
        sa.Column('audio_preview_key', sa.String(), nullable=True),
        sa.Column('audio_loop_key', sa.String(), nullable=True),
        sa.Column('cover_key', sa.String(), nullable=True),
        sa.Column('waveform_video_key', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('approved_at', sa.DateTime(), nullable=True),
        sa.Column('rejected_at', sa.DateTime(), nullable=True),
        sa.Column('published_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('slug')
    )
    op.create_index(op.f('ix_tracks_external_ref'), 'tracks', ['external_ref'], unique=True)

    # Assets
    op.create_table('assets',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('track_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('asset_type', sa.Enum('source_audio', 'master_audio', 'preview_audio', 'loop_audio', 'cover_image', 'youtube_video', 'short_video', 'compilation_audio', 'compilation_video', 'metadata_json', name='assettype'), nullable=False),
        sa.Column('storage_key', sa.String(), nullable=False),
        sa.Column('mime_type', sa.String(), nullable=True),
        sa.Column('byte_size', sa.BigInteger(), nullable=True),
        sa.Column('checksum_sha256', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['track_id'], ['tracks.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Compilations
    op.create_table('compilations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('slug', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('source_track_ids_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('audio_key', sa.String(), nullable=True),
        sa.Column('video_key', sa.String(), nullable=True),
        sa.Column('cover_key', sa.String(), nullable=True),
        sa.Column('status', sa.Enum('draft', 'rendered', 'published', 'failed', name='compilationstatus'), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('published_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('slug')
    )

    # Publishing Jobs
    op.create_table('publishing_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('track_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('compilation_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('channel', sa.Enum('youtube', 'tiktok', 'instagram', 'site', name='publishchannel'), nullable=False),
        sa.Column('content_type', sa.Enum('track_longform', 'track_short', 'compilation', 'product_page', name='publishcontenttype'), nullable=False),
        sa.Column('status', sa.Enum('queued', 'processing', 'done', 'failed', 'dead_letter', name='publishjobstatus'), nullable=False),
        sa.Column('scheduled_for', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False),
        sa.Column('max_retries', sa.Integer(), nullable=False),
        sa.Column('idempotency_key', sa.String(), nullable=False),
        sa.Column('external_post_id', sa.String(), nullable=True),
        sa.Column('external_url', sa.String(), nullable=True),
        sa.Column('request_payload_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('response_payload_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['compilation_id'], ['compilations.id'], ),
        sa.ForeignKeyConstraint(['track_id'], ['tracks.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('idempotency_key')
    )

    # Short Videos
    op.create_table('short_videos',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('track_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('variant_index', sa.Integer(), nullable=False),
        sa.Column('storage_key', sa.String(), nullable=False),
        sa.Column('duration_sec', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['track_id'], ['tracks.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('track_id', 'variant_index', name='uix_track_variant')
    )

    # Product Pages
    op.create_table('product_pages',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('track_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('cms_page_id', sa.String(), nullable=True),
        sa.Column('cms_url', sa.String(), nullable=True),
        sa.Column('store_product_id', sa.String(), nullable=True),
        sa.Column('pricing_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['track_id'], ['tracks.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('track_id')
    )

    # Analytics Daily
    op.create_table('analytics_daily',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('track_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('compilation_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('channel', sa.String(), nullable=False),
        sa.Column('entity_type', sa.String(), nullable=False),
        sa.Column('views', sa.BigInteger(), nullable=True),
        sa.Column('watch_time_seconds', sa.Float(), nullable=True),
        sa.Column('likes', sa.BigInteger(), nullable=True),
        sa.Column('comments', sa.BigInteger(), nullable=True),
        sa.Column('clicks', sa.BigInteger(), nullable=True),
        sa.Column('revenue_estimate', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('sales_count', sa.Integer(), nullable=True),
        sa.Column('captured_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['compilation_id'], ['compilations.id'], ),
        sa.ForeignKeyConstraint(['track_id'], ['tracks.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Run Logs
    op.create_table('run_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('pipeline_name', sa.String(), nullable=False),
        sa.Column('stage_name', sa.String(), nullable=False),
        sa.Column('entity_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('severity', sa.String(), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('context_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Scheduler Locks
    op.create_table('scheduler_locks',
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('locked_until', sa.DateTime(), nullable=False),
        sa.Column('owner_id', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('name')
    )


def downgrade() -> None:
    op.drop_table('scheduler_locks')
    op.drop_table('run_logs')
    op.drop_table('analytics_daily')
    op.drop_table('product_pages')
    op.drop_table('short_videos')
    op.drop_table('publishing_jobs')
    op.drop_table('compilations')
    op.drop_table('assets')
    op.drop_index(op.f('ix_tracks_external_ref'), table_name='tracks')
    op.drop_table('tracks')

    # Drop enums
    sa.Enum('draft', 'rendered', 'published', 'failed', name='compilationstatus').drop(op.get_bind())
    sa.Enum('queued', 'processing', 'done', 'failed', 'dead_letter', name='publishjobstatus').drop(op.get_bind())
    sa.Enum('track_longform', 'track_short', 'compilation', 'product_page', name='publishcontenttype').drop(op.get_bind())
    sa.Enum('youtube', 'tiktok', 'instagram', 'site', name='publishchannel').drop(op.get_bind())
    sa.Enum('source_audio', 'master_audio', 'preview_audio', 'loop_audio', 'cover_image', 'youtube_video', 'short_video', 'compilation_audio', 'compilation_video', 'metadata_json', name='assettype').drop(op.get_bind())
    sa.Enum('raw', 'approved', 'rejected', 'published', name='trackstatus').drop(op.get_bind())
