from abc import ABC, abstractmethod
from app.providers.base import PublishResult

class InstagramProvider(ABC):
    @abstractmethod
    def upload_reel(self, file_path: str, caption: str) -> PublishResult: pass
