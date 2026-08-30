from abc import ABC, abstractmethod
from app.providers.base import MusicGenerationResult

class MusicGenerationProvider(ABC):
    @abstractmethod
    def generate_track(self, prompt: str, duration_sec: int, metadata: dict) -> MusicGenerationResult: pass
