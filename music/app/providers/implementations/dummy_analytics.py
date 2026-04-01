from app.providers.base import AnalyticsProvider
import random


class DummyAnalyticsProvider(AnalyticsProvider):
    def fetch_track_metrics(self, track_id: str, channel: str) -> dict:
        return {
            "views": random.randint(100, 10000),
            "watch_time_seconds": random.randint(500, 50000),
            "likes": random.randint(10, 500),
        }

    def fetch_compilation_metrics(self, compilation_id: str, channel: str) -> dict:
        return {
            "views": random.randint(1000, 50000),
            "watch_time_seconds": random.randint(5000, 500000),
            "likes": random.randint(100, 2500),
        }
