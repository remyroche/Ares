from app.providers.base import PublishResult
from app.providers.youtube import YouTubeProvider, PublishResult
import uuid


class DummyYouTubeProvider(YouTubeProvider):
    def upload_video(
        self,
        file_path: str,
        title: str,
        description: str,
        tags: list[str],
        playlist_ids: list[str],
        privacy_status: str,
    ) -> PublishResult:
        vid_id = str(uuid.uuid4())[:11]
        return PublishResult(
            success=True,
            external_id=vid_id,
            external_url=f"https://youtube.com/watch?v={vid_id}",
        )
