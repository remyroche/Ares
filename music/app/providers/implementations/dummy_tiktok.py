from app.providers.base import PublishResult
from app.providers.tiktok import TikTokProvider, PublishResult
import uuid


class DummyTikTokProvider(TikTokProvider):
    def upload_video(self, file_path: str, caption: str) -> PublishResult:
        vid_id = str(uuid.uuid4())[:19]
        return PublishResult(
            success=True,
            external_id=vid_id,
            external_url=f"https://tiktok.com/@dummy/video/{vid_id}",
        )
