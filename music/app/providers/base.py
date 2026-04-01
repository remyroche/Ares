from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel


class MusicGenerationResult(BaseModel):
    success: bool
    external_job_id: Optional[str] = None
    audio_file_path: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    raw_response: Optional[Dict[str, Any]] = None


class ImageGenerationResult(BaseModel):
    success: bool
    image_file_path: Optional[str] = None
    image_bytes: Optional[bytes] = None
    raw_response: Optional[Dict[str, Any]] = None


class PublishResult(BaseModel):
    success: bool
    external_id: Optional[str] = None
    external_url: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class CMSPageResult(BaseModel):
    success: bool
    page_id: Optional[str] = None
    page_url: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class StoreProductResult(BaseModel):
    success: bool
    product_id: Optional[str] = None
    product_url: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class MusicGenerationProvider(ABC):
    @abstractmethod
    def generate_track(
        self, prompt: str, duration_sec: int, metadata: dict
    ) -> MusicGenerationResult:
        pass


class ImageGenerationProvider(ABC):
    @abstractmethod
    def generate_cover(
        self, prompt: str, size: str, metadata: dict
    ) -> ImageGenerationResult:
        pass


class YouTubeProvider(ABC):
    @abstractmethod
    def upload_video(
        self,
        file_path: str,
        title: str,
        description: str,
        tags: list[str],
        playlist_ids: list[str],
        privacy_status: str,
    ) -> PublishResult:
        pass


class TikTokProvider(ABC):
    @abstractmethod
    def upload_video(self, file_path: str, caption: str) -> PublishResult:
        pass


class InstagramProvider(ABC):
    @abstractmethod
    def upload_reel(self, file_path: str, caption: str) -> PublishResult:
        pass


class CMSProvider(ABC):
    @abstractmethod
    def create_track_page(self, payload: dict) -> CMSPageResult:
        pass

    @abstractmethod
    def create_compilation_page(self, payload: dict) -> CMSPageResult:
        pass


class StoreProvider(ABC):
    @abstractmethod
    def create_product(self, payload: dict) -> StoreProductResult:
        pass


class AnalyticsProvider(ABC):
    @abstractmethod
    def fetch_track_metrics(self, track_id: str, channel: str) -> dict:
        pass

    @abstractmethod
    def fetch_compilation_metrics(self, compilation_id: str, channel: str) -> dict:
        pass
