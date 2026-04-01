from sqlalchemy.orm import Session
from app.models import Track, PublishingJob, ShortVideo, Compilation
from app.enums import (
    PublishChannel,
    PublishContentType,
    PublishJobStatus,
    TrackStatus,
    CompilationStatus,
)
from app.utils.time import utcnow
from app.utils.hashing import generate_idempotency_key
from uuid import UUID
import os

from app.services.render_service import RenderService
from app.services.storage_service import StorageService
from app.services.catalog_service import CatalogService
from app.providers import (
    get_youtube_provider,
    get_tiktok_provider,
    get_instagram_provider,
)


class PublishingService:
    def __init__(self, db: Session):
        self.db = db
        self.storage = StorageService()

    def enqueue_track_publish_jobs(self, track_id: UUID):
        jobs = []
        now = utcnow()

        # YouTube Longform
        jobs.append(
            PublishingJob(
                track_id=track_id,
                channel=PublishChannel.youtube,
                content_type=PublishContentType.track_longform,
                scheduled_for=now,
                idempotency_key=generate_idempotency_key(
                    str(track_id), "youtube", "longform"
                ),
            )
        )

        # Site Product
        jobs.append(
            PublishingJob(
                track_id=track_id,
                channel=PublishChannel.site,
                content_type=PublishContentType.product_page,
                scheduled_for=now,
                idempotency_key=generate_idempotency_key(
                    str(track_id), "site", "product"
                ),
            )
        )

        # We defer shorts queueing until they are generated in the shorts pipeline

        for job in jobs:
            existing = (
                self.db.query(PublishingJob)
                .filter_by(idempotency_key=job.idempotency_key)
                .first()
            )
            if not existing:
                self.db.add(job)

        self.db.commit()


class ExtendedPublishingService(PublishingService):

    def publish_next_youtube_job(self):
        job = (
            self.db.query(PublishingJob)
            .filter(
                PublishingJob.channel == PublishChannel.youtube,
                PublishingJob.status == PublishJobStatus.queued,
            )
            .order_by(PublishingJob.scheduled_for)
            .first()
        )

        if not job:
            return None

        job.status = PublishJobStatus.processing
        job.started_at = utcnow()
        self.db.commit()

        try:
            track = self.db.query(Track).filter_by(id=job.track_id).first()
            if not track:
                raise Exception("Track not found")

            # render longform if missing
            youtube_key = self.storage.compute_key(
                "tracks", str(track.id), "youtube.mp4"
            )
            if not self.storage.exists(youtube_key):
                audio_path = f"/tmp/pub_audio_{track.id}.mp3"
                cover_path = f"/tmp/pub_cover_{track.id}.png"
                out_path = f"/tmp/pub_yt_{track.id}.mp4"

                self.storage.download_file(track.audio_master_key, audio_path)
                self.storage.download_file(track.cover_key, cover_path)

                RenderService.render_youtube_video(
                    str(track.id), audio_path, cover_path, out_path, track.title
                )

                self.storage.upload_file(out_path, youtube_key)

                os.remove(audio_path)
                os.remove(cover_path)
                os.remove(out_path)

            track.waveform_video_key = youtube_key
            self.db.commit()

            yt = get_youtube_provider()
            res = yt.upload_video(youtube_key, track.title, "desc", [], [], "public")

            job.status = PublishJobStatus.done
            job.finished_at = utcnow()
            job.external_post_id = res.external_id
            job.external_url = res.external_url

            # mark track published
            if track.status == TrackStatus.approved:
                track.status = TrackStatus.published
                track.published_at = utcnow()

            self.db.commit()
            return job

        except Exception as e:
            self.db.rollback()
            job.retry_count += 1
            if job.retry_count >= job.max_retries:
                job.status = PublishJobStatus.dead_letter
            else:
                job.status = PublishJobStatus.queued
            job.error_message = str(e)
            self.db.commit()
            raise e

    def publish_next_short_jobs(self, limit: int = 3):
        # 1. find approved tracks lacking short renders
        from app.config import settings
        import json

        approved_tracks = (
            self.db.query(Track)
            .filter(Track.status.in_([TrackStatus.approved, TrackStatus.published]))
            .all()
        )

        for track in approved_tracks:
            shorts_count = (
                self.db.query(ShortVideo).filter_by(track_id=track.id).count()
            )
            if shorts_count < settings.DAILY_SHORTS_PER_TRACK:
                # 2. Render 3 shorts if missing

                # Fetch metadata to get short captions
                meta_key = self.storage.compute_key(
                    "tracks", str(track.id), "metadata.json"
                )
                captions = [
                    f"Short video {i} for {track.title}"
                    for i in range(settings.DAILY_SHORTS_PER_TRACK)
                ]
                if self.storage.exists(meta_key):
                    meta_path = f"/tmp/meta_{track.id}.json"
                    self.storage.download_file(meta_key, meta_path)
                    with open(meta_path, "r") as f:
                        meta_data = json.load(f)
                    if "short_captions" in meta_data:
                        captions = meta_data["short_captions"][
                            : settings.DAILY_SHORTS_PER_TRACK
                        ]
                    os.remove(meta_path)

                audio_path = f"/tmp/short_audio_{track.id}.mp3"
                cover_path = f"/tmp/short_cover_{track.id}.png"
                out_dir = f"/tmp/shorts_dir_{track.id}"

                self.storage.download_file(track.audio_preview_key, audio_path)
                self.storage.download_file(track.cover_key, cover_path)

                outputs = RenderService.render_short_videos(
                    str(track.id), audio_path, cover_path, out_dir, captions
                )

                # 3. Create short_video rows
                for i, out_path in enumerate(outputs):
                    short_key = self.storage.compute_key(
                        "tracks", str(track.id), f"short_{i+1}.mp4"
                    )
                    self.storage.upload_file(out_path, short_key)

                    sv = ShortVideo(
                        track_id=track.id,
                        variant_index=i + 1,
                        storage_key=short_key,
                        duration_sec=30,
                    )
                    self.db.add(sv)

                    # 4. Enqueue TikTok and Instagram jobs
                    for channel in [PublishChannel.tiktok, PublishChannel.instagram]:
                        job = PublishingJob(
                            track_id=track.id,
                            channel=channel,
                            content_type=PublishContentType.track_short,
                            scheduled_for=utcnow(),
                            idempotency_key=generate_idempotency_key(
                                str(track.id), str(channel.value), f"short_{i+1}"
                            ),
                        )
                        self.db.add(job)

                self.db.commit()

                import shutil

                shutil.rmtree(out_dir, ignore_errors=True)
                os.remove(audio_path)
                os.remove(cover_path)

        # 5. publish queued short jobs
        jobs = (
            self.db.query(PublishingJob)
            .filter(
                PublishingJob.content_type == PublishContentType.track_short,
                PublishingJob.status == PublishJobStatus.queued,
            )
            .order_by(PublishingJob.scheduled_for)
            .limit(limit)
            .all()
        )

        processed = []
        for job in jobs:
            job.status = PublishJobStatus.processing
            job.started_at = utcnow()
            self.db.commit()

            try:
                # determine variant
                parts = job.idempotency_key.split("|")
                # parts[2] should be "short_1", "short_2" etc based on hashing format
                variant_index = int(parts[-1].split("_")[-1]) if "_" in parts[-1] else 1

                sv = (
                    self.db.query(ShortVideo)
                    .filter_by(track_id=job.track_id, variant_index=variant_index)
                    .first()
                )
                if not sv:
                    raise Exception(f"ShortVideo missing for variant {variant_index}")

                caption = "Relax with this rainy lofi beat 🌧️🎧 #lofi"  # fallback

                res = None
                if job.channel == PublishChannel.tiktok:
                    provider = get_tiktok_provider()
                    res = provider.upload_video(sv.storage_key, caption)
                elif job.channel == PublishChannel.instagram:
                    provider = get_instagram_provider()
                    res = provider.upload_reel(sv.storage_key, caption)

                if res:
                    job.status = PublishJobStatus.done
                    job.finished_at = utcnow()
                    job.external_post_id = res.external_id
                    job.external_url = res.external_url

                self.db.commit()
                processed.append(job)

            except Exception as e:
                self.db.rollback()
                job.retry_count += 1
                job.status = (
                    PublishJobStatus.dead_letter
                    if job.retry_count >= job.max_retries
                    else PublishJobStatus.queued
                )
                job.error_message = str(e)
                self.db.commit()

        return processed

    def publish_next_site_job(self):
        job = (
            self.db.query(PublishingJob)
            .filter(
                PublishingJob.channel == PublishChannel.site,
                PublishingJob.content_type == PublishContentType.product_page,
                PublishingJob.status == PublishJobStatus.queued,
            )
            .order_by(PublishingJob.scheduled_for)
            .first()
        )

        if not job:
            return None

        job.status = PublishJobStatus.processing
        job.started_at = utcnow()
        self.db.commit()

        try:
            cat_service = CatalogService(self.db)
            cat_service.process_product_page(str(job.track_id))

            job.status = PublishJobStatus.done
            job.finished_at = utcnow()
            self.db.commit()
            return job
        except Exception as e:
            self.db.rollback()
            job.retry_count += 1
            job.status = (
                PublishJobStatus.dead_letter
                if job.retry_count >= job.max_retries
                else PublishJobStatus.queued
            )
            job.error_message = str(e)
            self.db.commit()
            raise e

    def publish_compilation(self, compilation_id: UUID):
        job = (
            self.db.query(PublishingJob)
            .filter(
                PublishingJob.compilation_id == compilation_id,
                PublishingJob.status == PublishJobStatus.queued,
            )
            .first()
        )

        if not job:
            return None

        job.status = PublishJobStatus.processing
        job.started_at = utcnow()
        self.db.commit()

        try:
            comp = self.db.query(Compilation).filter_by(id=compilation_id).first()
            if not comp or not comp.video_key:
                raise Exception("Compilation not ready")

            yt = get_youtube_provider()
            res = yt.upload_video(
                comp.video_key, comp.title, "Weekly Compilation", [], [], "public"
            )

            job.status = PublishJobStatus.done
            job.finished_at = utcnow()
            job.external_post_id = res.external_id
            job.external_url = res.external_url

            comp.status = CompilationStatus.published
            comp.published_at = utcnow()

            self.db.commit()
            return job

        except Exception as e:
            self.db.rollback()
            job.retry_count += 1
            job.status = (
                PublishJobStatus.dead_letter
                if job.retry_count >= job.max_retries
                else PublishJobStatus.queued
            )
            job.error_message = str(e)
            self.db.commit()
            raise e
