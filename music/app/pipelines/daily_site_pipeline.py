from app.db import SessionLocal
from app.services.catalog_service import CatalogService


class DailySitePipeline:
    @staticmethod
    def run():
        db = SessionLocal()
        try:
            from app.models import PublishingJob
            from app.enums import PublishChannel, PublishContentType, PublishJobStatus
            from app.utils.time import utcnow

            job = (
                db.query(PublishingJob)
                .filter(
                    PublishingJob.channel == PublishChannel.site,
                    PublishingJob.content_type == PublishContentType.product_page,
                    PublishingJob.status == PublishJobStatus.queued,
                )
                .order_by(PublishingJob.scheduled_for)
                .first()
            )

            if not job:
                return None

            job.status = PublishJobStatus.processing
            job.started_at = utcnow()
            db.commit()

            try:
                cat_service = CatalogService(db)
                cat_service.process_product_page(str(job.track_id))

                job.status = PublishJobStatus.done
                job.finished_at = utcnow()
                db.commit()
                return str(job.id)
            except Exception as e:
                db.rollback()
                job.retry_count += 1
                job.status = (
                    PublishJobStatus.dead_letter
                    if job.retry_count >= job.max_retries
                    else PublishJobStatus.queued
                )
                job.error_message = str(e)
                db.commit()
                raise e
        finally:
            db.close()
