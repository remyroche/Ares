from sqlalchemy.orm import Session
from datetime import date
from app.models import AnalyticsDaily, Track, Compilation
from app.providers import get_analytics_provider
from app.utils.time import utcnow


class AnalyticsService:
    def __init__(self, db: Session):
        self.db = db
        self.provider = get_analytics_provider()

    def refresh_daily_metrics(self, reference_date: date):
        # mock loop for top 10 recent tracks and compilations
        tracks = self.db.query(Track).limit(10).all()
        for t in tracks:
            metrics = self.provider.fetch_track_metrics(str(t.id), "youtube")

            ad = AnalyticsDaily(
                track_id=t.id,
                channel="youtube",
                entity_type="track",
                views=metrics.get("views"),
                watch_time_seconds=metrics.get("watch_time_seconds"),
                likes=metrics.get("likes"),
                captured_at=utcnow(),
            )
            self.db.add(ad)

        comps = self.db.query(Compilation).limit(5).all()
        for c in comps:
            metrics = self.provider.fetch_compilation_metrics(str(c.id), "youtube")

            ad = AnalyticsDaily(
                compilation_id=c.id,
                channel="youtube",
                entity_type="compilation",
                views=metrics.get("views"),
                watch_time_seconds=metrics.get("watch_time_seconds"),
                likes=metrics.get("likes"),
                captured_at=utcnow(),
            )
            self.db.add(ad)

        self.db.commit()

    def get_top_performers(self):
        # mock summary
        return [
            {
                "entity_id": "dummy-uuid",
                "views": 1000,
                "watch_time_seconds": 5000,
                "sales_count": 2,
                "score": 1000 * 0.35 + 5000 * 0.40 + 2 * 100 * 0.25,
            }
        ]
