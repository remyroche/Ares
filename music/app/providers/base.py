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
