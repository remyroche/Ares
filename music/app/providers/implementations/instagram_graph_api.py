from app.providers.instagram import InstagramProvider
from app.providers.base import PublishResult
from app.config import settings

class InstagramGraphAPIProvider(InstagramProvider):
    def upload_reel(self, file_path: str, caption: str) -> PublishResult:
        if settings.DRY_RUN:
            return PublishResult(success=True, external_id="dry-run-ig", external_url="https://instagram.com/dryrun")

        return PublishResult(success=False, error_message="IG API real implementation pending complete credentials.")
