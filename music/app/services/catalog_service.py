from sqlalchemy.orm import Session
from app.models import Track, ProductPage
from app.providers import get_cms_provider, get_store_provider
from app.config import settings
from app.utils.time import utcnow
from app.services.storage_service import StorageService


class CatalogService:
    def __init__(self, db: Session):
        self.db = db
        self.cms_provider = get_cms_provider()
        self.store_provider = get_store_provider()
        self.storage = StorageService()

    def process_product_page(self, track_id: str):
        track = self.db.query(Track).filter_by(id=track_id).first()
        if not track:
            return

        preview_url = self.storage.signed_url(track.audio_preview_key)
        cover_url = self.storage.signed_url(track.cover_key)

        # 1. Store Product
        store_payload = {
            "title": track.title,
            "type": "digital_audio",
            "variants": [
                {
                    "name": "Personal License",
                    "price": settings.LICENSE_PRICE_PERSONAL_EUR,
                },
                {
                    "name": "Creator License",
                    "price": settings.LICENSE_PRICE_CREATOR_EUR,
                },
                {
                    "name": "Commercial License",
                    "price": settings.LICENSE_PRICE_COMMERCIAL_EUR,
                },
            ],
        }
        store_res = self.store_provider.create_product(store_payload)

        # 2. CMS Page
        cms_payload = {
            "title": track.title,
            "slug": track.slug,
            "preview_url": preview_url,
            "cover_url": cover_url,
            "store_url": store_res.product_url,
            "description": f"Listen and license {track.title}",
        }
        cms_res = self.cms_provider.create_track_page(cms_payload)

        # 3. Persist
        page = self.db.query(ProductPage).filter_by(track_id=track.id).first()
        if not page:
            page = ProductPage(
                track_id=track.id, pricing_json=store_payload["variants"]
            )
            self.db.add(page)

        page.cms_page_id = cms_res.page_id
        page.cms_url = cms_res.page_url
        page.store_product_id = store_res.product_id
        page.updated_at = utcnow()

        self.db.commit()
