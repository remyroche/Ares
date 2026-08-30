from app.providers.tiktok import TikTokProvider
from app.providers.base import PublishResult
from app.config import settings

class TikTokAPIProvider(TikTokProvider):
    def upload_video(self, file_path: str, caption: str) -> PublishResult:
        if settings.DRY_RUN:
            return PublishResult(success=True, external_id="dry-run-tt", external_url="https://tiktok.com/dryrun")

        return PublishResult(success=False, error_message="TikTok API real implementation pending complete credentials.")
