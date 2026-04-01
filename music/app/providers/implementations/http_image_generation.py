from app.providers.image_generation import ImageGenerationProvider
from app.providers.base import ImageGenerationResult
from app.config import settings
import httpx

class HTTPImageGenerationProvider(ImageGenerationProvider):
    def generate_cover(self, prompt: str, size: str, metadata: dict) -> ImageGenerationResult:
        if settings.DRY_RUN:
            return ImageGenerationResult(success=True, image_bytes=b"fake_image")

        headers = {"Authorization": f"Bearer {settings.IMAGE_PROVIDER_API_KEY}"}
        payload = {"prompt": prompt, "size": size}

        with httpx.Client(timeout=60) as client:
            resp = client.post(f"{settings.IMAGE_PROVIDER_BASE_URL}/generate", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            image_url = data.get("image_url")

            if image_url:
                img_resp = client.get(image_url)
                img_resp.raise_for_status()
                return ImageGenerationResult(success=True, image_bytes=img_resp.content, raw_response=data)

            return ImageGenerationResult(success=False, raw_response=data)
