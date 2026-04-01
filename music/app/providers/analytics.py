from abc import ABC, abstractmethod

class AnalyticsProvider(ABC):
    @abstractmethod
    def fetch_track_metrics(self, track_id: str, channel: str) -> dict: pass
    @abstractmethod
    def fetch_compilation_metrics(self, compilation_id: str, channel: str) -> dict: pass
