from app.db import SessionLocal
from app.services.publishing_service import ExtendedPublishingService


class DailyShortsPipeline:
    @staticmethod
    def run():
        db = SessionLocal()
        try:
            pub_service = ExtendedPublishingService(db)
            jobs_processed = []

            # Find approved tracks lacking short renders and enqueue if needed (handled in service)
            # Actually process the next queued jobs
            limit = 3  # process 3 at a time for instance

            processed = pub_service.publish_next_short_jobs(limit)
            if processed:
                for j in processed:
                    jobs_processed.append(str(j.id))

            return jobs_processed
        finally:
            db.close()
