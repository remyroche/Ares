from app.providers.base import InstagramProvider, PublishResult
import uuid


class DummyInstagramProvider(InstagramProvider):
    def upload_reel(self, file_path: str, caption: str) -> PublishResult:
        vid_id = str(uuid.uuid4())[:11]
        return PublishResult(
            success=True,
            external_id=vid_id,
            external_url=f"https://instagram.com/reel/{vid_id}/",
        )
