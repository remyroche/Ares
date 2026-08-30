from app.providers.base import CMSPageResult
from app.providers.cms import CMSProvider, CMSPageResult
import uuid
import os
import json


class DummyCMSProvider(CMSProvider):
    def create_track_page(self, payload: dict) -> CMSPageResult:
        os.makedirs("exports/site", exist_ok=True)
        page_id = str(uuid.uuid4())
        with open(f"exports/site/track_{page_id}.json", "w") as f:
            json.dump(payload, f)
        return CMSPageResult(
            success=True, page_id=page_id, page_url=f"https://dummy.cms/track/{page_id}"
        )

    def create_compilation_page(self, payload: dict) -> CMSPageResult:
        os.makedirs("exports/site", exist_ok=True)
        page_id = str(uuid.uuid4())
        with open(f"exports/site/compilation_{page_id}.json", "w") as f:
            json.dump(payload, f)
        return CMSPageResult(
            success=True,
            page_id=page_id,
            page_url=f"https://dummy.cms/compilation/{page_id}",
        )
