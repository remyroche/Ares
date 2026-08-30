from abc import ABC, abstractmethod
from app.providers.base import CMSPageResult

class CMSProvider(ABC):
    @abstractmethod
    def create_track_page(self, payload: dict) -> CMSPageResult: pass
    @abstractmethod
    def create_compilation_page(self, payload: dict) -> CMSPageResult: pass
