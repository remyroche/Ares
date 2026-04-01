from abc import ABC, abstractmethod
from app.providers.base import PublishResult

class YouTubeProvider(ABC):
    @abstractmethod
    def upload_video(self, file_path: str, title: str, description: str, tags: list[str], playlist_ids: list[str], privacy_status: str) -> PublishResult: pass
