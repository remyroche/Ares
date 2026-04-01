from app.db import SessionLocal
from app.services.publishing_service import ExtendedPublishingService


class DailySitePipeline:
    @staticmethod
    def run():
        db = SessionLocal()
        try:
            pub_service = ExtendedPublishingService(db)
            job = pub_service.publish_next_site_job()
            if job:
                return str(job.id)
            return None
        finally:
            db.close()
