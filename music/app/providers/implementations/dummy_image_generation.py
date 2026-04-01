from app.providers.base import ImageGenerationProvider, ImageGenerationResult
from PIL import Image
import io


class DummyImageGenerationProvider(ImageGenerationProvider):
    def generate_cover(
        self, prompt: str, size: str, metadata: dict
    ) -> ImageGenerationResult:
        img = Image.new("RGB", (2048, 2048), color=(73, 109, 137))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return ImageGenerationResult(success=True, image_bytes=buf.getvalue())
