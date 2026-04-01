from app.providers.cms import CMSProvider
from app.providers.base import CMSPageResult
from app.config import settings
import httpx

class WordpressRestProvider(CMSProvider):
    def create_track_page(self, payload: dict) -> CMSPageResult:
        if settings.DRY_RUN:
            return CMSPageResult(success=True, page_id="dry-run-wp-track", page_url="https://wp/dryrun")

        headers = {"Authorization": f"Bearer {settings.CMS_API_KEY}"}
        wp_payload = {
            "title": payload["title"],
            "content": payload["description"],
            "status": "publish",
            "slug": payload["slug"]
        }

        with httpx.Client() as client:
            resp = client.post(f"{settings.CMS_BASE_URL}/wp-json/wp/v2/pages", json=wp_payload, headers=headers)
            if resp.status_code in (200, 201):
                data = resp.json()
                return CMSPageResult(success=True, page_id=str(data["id"]), page_url=data["link"], raw_response=data)
            return CMSPageResult(success=False, raw_response={"status": resp.status_code, "text": resp.text})

    def create_compilation_page(self, payload: dict) -> CMSPageResult:
        return self.create_track_page(payload) # Use same logic for WP pages
