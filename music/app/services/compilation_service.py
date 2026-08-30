from sqlalchemy.orm import Session
from datetime import date, timedelta
from app.models import Track, Compilation
from app.enums import TrackStatus, CompilationStatus
from app.services.storage_service import StorageService
from app.services.render_service import RenderService
from app.providers import get_image_provider
from app.utils.slug import slugify
from app.utils.time import utcnow
from app.config import settings
import os


class CompilationService:
    def __init__(self, db: Session):
        self.db = db
        self.storage = StorageService()

    def create_weekly_compilation(self, reference_date: date) -> Compilation:
        start_date = reference_date - timedelta(days=7)

        # 1. select approved tracks
        tracks = (
            self.db.query(Track)
            .filter(
                Track.status == TrackStatus.published,
                Track.published_at >= start_date,
                Track.published_at < reference_date,
            )
            .order_by(Track.published_at)
            .all()
        )

        if len(tracks) < 3:
            return None  # Skip gracefully

        title = f"{settings.BRAND_NAME} Weekly Compilation {reference_date.strftime('%Y-%m-%d')}"
        slug = slugify(title)

        # 3. create row
        comp = Compilation(
            title=title,
            slug=slug,
            source_track_ids_json=[str(t.id) for t in tracks],
            status=CompilationStatus.draft,
        )
        self.db.add(comp)
        self.db.commit()
        self.db.refresh(comp)

        comp_dir = f"/tmp/music_factory/comp/{comp.id}"
        os.makedirs(comp_dir, exist_ok=True)

        try:
            # download masters
            list_file_path = os.path.join(comp_dir, "list.txt")
            with open(list_file_path, "w") as f:
                for t in tracks:
                    local_path = os.path.join(comp_dir, f"{t.id}.mp3")
                    self.storage.download_file(t.audio_master_key, local_path)
                    f.write(f"file '{local_path}'\n")

            # 4. concatenate audio
            out_audio = os.path.join(comp_dir, "comp_audio.mp3")
            os.system(
                f"ffmpeg -y -f concat -safe 0 -i {list_file_path} -c copy {out_audio} >/dev/null 2>&1"
            )

            # 6. generate cover
            provider = get_image_provider()
            cover_res = provider.generate_cover(
                "weekly compilation cover", "2048x2048", {}
            )
            cover_path = os.path.join(comp_dir, "cover.png")
            if cover_res.image_bytes:
                with open(cover_path, "wb") as f:
                    f.write(cover_res.image_bytes)

            # 7. render video
            out_video = os.path.join(comp_dir, "comp_video.mp4")
            RenderService.render_compilation_video(
                str(comp.id), out_audio, cover_path, out_video, title
            )

            # Upload
            comp.audio_key = self.storage.compute_key(
                "compilations", str(comp.id), "audio.mp3"
            )
            comp.cover_key = self.storage.compute_key(
                "compilations", str(comp.id), "cover.png"
            )
            comp.video_key = self.storage.compute_key(
                "compilations", str(comp.id), "video.mp4"
            )

            self.storage.upload_file(out_audio, comp.audio_key)
            self.storage.upload_file(cover_path, comp.cover_key)
            self.storage.upload_file(out_video, comp.video_key)

            comp.status = CompilationStatus.rendered
            self.db.commit()

            # Enqueue publish jobs (mocked for this service to be picked up by publisher)
            from app.models import PublishingJob
            from app.enums import PublishChannel, PublishContentType
            from app.utils.hashing import generate_idempotency_key

            job = PublishingJob(
                compilation_id=comp.id,
                channel=PublishChannel.youtube,
                content_type=PublishContentType.compilation,
                scheduled_for=utcnow(),
                idempotency_key=generate_idempotency_key(
                    str(comp.id), "youtube", "compilation"
                ),
            )
            self.db.add(job)
            self.db.commit()

        except Exception as e:
            self.db.rollback()
            comp.status = CompilationStatus.failed
            self.db.commit()
            raise e
        finally:
            import shutil

            shutil.rmtree(comp_dir, ignore_errors=True)

        return comp
