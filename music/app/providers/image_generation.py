from abc import ABC, abstractmethod
from app.providers.base import ImageGenerationResult

class ImageGenerationProvider(ABC):
    @abstractmethod
    def generate_cover(self, prompt: str, size: str, metadata: dict) -> ImageGenerationResult: pass
