from app.db import SessionLocal
from app.services.analytics_service import AnalyticsService
from app.utils.time import localnow


class DailyAnalyticsPipeline:
    @staticmethod
    def run():
        db = SessionLocal()
        try:
            anal_service = AnalyticsService(db)
            anal_service.refresh_daily_metrics(localnow().date())
            return True
        finally:
            db.close()
