from sqlalchemy.orm import Session
from app.models import Track, Asset
from app.enums import TrackStatus, AssetType
from app.services.storage_service import StorageService
from app.services.qc_service import QCService
from app.services.metadata_service import MetadataService
from app.providers import get_music_provider, get_image_provider
from app.config import settings
from app.utils.time import utcnow
import os
import json


class GenerationService:
    def __init__(self, db: Session):
        self.db = db
        self.storage = StorageService()
        self.music_provider = get_music_provider()
        self.image_provider = get_image_provider()

    def generate_track(self, prompt_spec: dict) -> Track:
        # 1. dedupe
        existing = (
            self.db.query(Track)
            .filter(Track.external_ref == prompt_spec["external_ref"])
            .first()
        )
        if existing:
            return existing

        # 2. create raw track
        track = Track(
            external_ref=prompt_spec["external_ref"],
            brand=settings.BRAND_NAME,
            series=settings.SERIES_NAME,
            prompt=prompt_spec["prompt"],
            genre=prompt_spec["genre"],
            mood=prompt_spec["mood"],
            bpm=prompt_spec["bpm"],
            duration_sec=prompt_spec["duration_sec"],
            status=TrackStatus.raw,
            generation_provider=settings.MUSIC_PROVIDER,
            image_provider=settings.IMAGE_PROVIDER,
        )
        self.db.add(track)
        self.db.commit()
        self.db.refresh(track)

        track_dir = f"/tmp/music_factory/tracks/{track.id}"
        os.makedirs(track_dir, exist_ok=True)

        try:
            # 3. call music provider
            music_res = self.music_provider.generate_track(
                track.prompt, track.duration_sec, {"external_ref": track.external_ref}
            )
            if not music_res.success:
                raise Exception("Music generation failed")

            # 4. store source
            source_key = self.storage.compute_key("tracks", str(track.id), "source.wav")
            self.storage.upload_file(music_res.audio_file_path, source_key)
            self.db.add(
                Asset(
                    track_id=track.id,
                    asset_type=AssetType.source_audio,
                    storage_key=source_key,
                )
            )

            # 5. audio pipeline & QC
            master_path = f"{track_dir}/master.mp3"
            preview_path = f"{track_dir}/preview.mp3"
            loop_path = f"{track_dir}/loop.mp3"

            # Mock audio pipeline for demo, just copy the wav to mp3
            os.system(
                f"ffmpeg -y -i {music_res.audio_file_path} -c:a libmp3lame {master_path} >/dev/null 2>&1"
            )
            os.system(
                f"ffmpeg -y -i {music_res.audio_file_path} -c:a libmp3lame -t 30 {preview_path} >/dev/null 2>&1"
            )
            os.system(
                f"ffmpeg -y -i {music_res.audio_file_path} -c:a libmp3lame {loop_path} >/dev/null 2>&1"
            )

            qc_res = QCService.analyze_audio(master_path)
            track.duration_sec = qc_res["duration_sec"]
            track.qc_score = qc_res["qc_score"]

            # 6. store assets
            track.audio_master_key = self.storage.compute_key(
                "tracks", str(track.id), "master.mp3"
            )
            track.audio_preview_key = self.storage.compute_key(
                "tracks", str(track.id), "preview.mp3"
            )
            track.audio_loop_key = self.storage.compute_key(
                "tracks", str(track.id), "loop.mp3"
            )

            self.storage.upload_file(master_path, track.audio_master_key)
            self.storage.upload_file(preview_path, track.audio_preview_key)
            self.storage.upload_file(loop_path, track.audio_loop_key)

            for at, k in [
                (AssetType.master_audio, track.audio_master_key),
                (AssetType.preview_audio, track.audio_preview_key),
                (AssetType.loop_audio, track.audio_loop_key),
            ]:
                self.db.add(Asset(track_id=track.id, asset_type=at, storage_key=k))

            # 7. generate cover
            cover_res = self.image_provider.generate_cover(
                "album cover art...", "2048x2048", {}
            )
            track.cover_key = self.storage.compute_key(
                "tracks", str(track.id), "cover.png"
            )
            if cover_res.image_bytes:
                self.storage.upload_bytes(
                    cover_res.image_bytes, track.cover_key, "image/png"
                )
            elif cover_res.image_file_path:
                self.storage.upload_file(
                    cover_res.image_file_path, track.cover_key, "image/png"
                )
            self.db.add(
                Asset(
                    track_id=track.id,
                    asset_type=AssetType.cover_image,
                    storage_key=track.cover_key,
                )
            )

            # 8. generate metadata
            index = self.db.query(Track).count()
            metadata = MetadataService.generate_track_metadata(track, index)
            track.title = metadata["title"]
            track.slug = metadata["slug"]

            meta_key = self.storage.compute_key(
                "tracks", str(track.id), "metadata.json"
            )
            self.storage.upload_bytes(
                json.dumps(metadata).encode(), meta_key, "application/json"
            )
            self.db.add(
                Asset(
                    track_id=track.id,
                    asset_type=AssetType.metadata_json,
                    storage_key=meta_key,
                )
            )

            # 9. status
            if qc_res["passed"]:
                track.status = TrackStatus.approved
                track.approved_at = utcnow()
            else:
                track.status = TrackStatus.rejected
                track.rejected_at = utcnow()

            self.db.commit()

            # 10. enqueue publishing if approved
            if track.status == TrackStatus.approved:
                from app.services.publishing_service import PublishingService

                ps = PublishingService(self.db)
                ps.enqueue_track_publish_jobs(track.id)

        except Exception as e:
            self.db.rollback()
            raise e
        finally:
            import shutil

            shutil.rmtree(track_dir, ignore_errors=True)

        return track
