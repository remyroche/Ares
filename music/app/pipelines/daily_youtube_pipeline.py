from app.db import SessionLocal


class DailyYoutubePipeline:
    @staticmethod
    def run():
        # Using a mock inline logic since ExtendedPublishingService might not be fully available to Celery
        db = SessionLocal()
        try:
            # We fetch a job from DB manually for celery worker
            from app.models import PublishingJob, Track
            from app.enums import (
                PublishChannel,
                PublishContentType,
                PublishJobStatus,
                TrackStatus,
            )
            from app.utils.time import utcnow
            from app.services.render_service import RenderService
            from app.services.storage_service import StorageService
            from app.providers import get_youtube_provider

            job = (
                db.query(PublishingJob)
                .filter(
                    PublishingJob.channel == PublishChannel.youtube,
                    PublishingJob.content_type == PublishContentType.track_longform,
                    PublishingJob.status == PublishJobStatus.queued,
                )
                .order_by(PublishingJob.scheduled_for)
                .first()
            )

            if not job:
                return None

            job.status = PublishJobStatus.processing
            job.started_at = utcnow()
            db.commit()

            try:
                track = db.query(Track).filter_by(id=job.track_id).first()
                if not track:
                    raise Exception("Track not found")

                storage = StorageService()
                youtube_key = storage.compute_key(
                    "tracks", str(track.id), "youtube.mp4"
                )

                if not storage.exists(youtube_key):
                    import os

                    audio_path = f"/tmp/pub_audio_{track.id}.mp3"
                    cover_path = f"/tmp/pub_cover_{track.id}.png"
                    out_path = f"/tmp/pub_yt_{track.id}.mp4"

                    storage.download_file(track.audio_master_key, audio_path)
                    storage.download_file(track.cover_key, cover_path)

                    RenderService.render_youtube_video(
                        str(track.id), audio_path, cover_path, out_path, track.title
                    )
                    storage.upload_file(out_path, youtube_key)

                    os.remove(audio_path)
                    os.remove(cover_path)
                    os.remove(out_path)

                yt = get_youtube_provider()
                res = yt.upload_video(
                    youtube_key, track.title, track.title, ["#lofi"], [], "public"
                )

                job.status = PublishJobStatus.done
                job.finished_at = utcnow()
                job.external_post_id = res.external_id
                job.external_url = res.external_url

                if track.status == TrackStatus.approved:
                    track.status = TrackStatus.published
                    track.published_at = utcnow()

                db.commit()
                return str(job.id)

            except Exception as e:
                db.rollback()
                job.retry_count += 1
                job.status = (
                    PublishJobStatus.dead_letter
                    if job.retry_count >= job.max_retries
                    else PublishJobStatus.queued
                )
                job.error_message = str(e)
                db.commit()
                raise e
        finally:
            db.close()
