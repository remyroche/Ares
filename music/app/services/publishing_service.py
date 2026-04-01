from sqlalchemy.orm import Session
from app.models import Track, PublishingJob
from app.enums import PublishChannel, PublishContentType, PublishJobStatus, TrackStatus
from app.utils.time import utcnow
from app.utils.hashing import generate_idempotency_key
from uuid import UUID

from app.services.render_service import RenderService
from app.services.storage_service import StorageService
from app.providers import get_youtube_provider


class PublishingService:
    def __init__(self, db: Session):
        self.db = db

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

        for job in jobs:
            existing = (
                self.db.query(PublishingJob)
                .filter_by(idempotency_key=job.idempotency_key)
                .first()
            )
            if not existing:
                self.db.add(job)

        self.db.commit()

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
            storage = StorageService()
            youtube_key = storage.compute_key("tracks", str(track.id), "youtube.mp4")
            if not storage.exists(youtube_key):
                audio_path = f"/tmp/pub_audio_{track.id}.mp3"
                cover_path = f"/tmp/pub_cover_{track.id}.png"
                out_path = f"/tmp/pub_yt_{track.id}.mp4"

                storage.download_file(track.audio_master_key, audio_path)
                storage.download_file(track.cover_key, cover_path)

                RenderService.render_youtube_video(
                    str(track.id), audio_path, cover_path, out_path, track.title
                )

                storage.upload_file(out_path, youtube_key)
                import os

                os.remove(audio_path)
                os.remove(cover_path)
                os.remove(out_path)

            track.waveform_video_key = youtube_key
            self.db.commit()

            # Dummy Upload for now
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

    def publish_next_short_jobs(self, limit: int):
        pass

    def publish_next_site_job(self):
        pass

    def publish_compilation(self, compilation_id: UUID):
        pass


class ExtendedPublishingService(PublishingService):
    pass
