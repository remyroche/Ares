from abc import ABC, abstractmethod
from app.providers.base import PublishResult

class TikTokProvider(ABC):
    @abstractmethod
    def upload_video(self, file_path: str, caption: str) -> PublishResult: pass
