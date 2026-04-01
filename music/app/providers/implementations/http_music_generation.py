from app.providers.music_generation import MusicGenerationProvider
from app.providers.base import MusicGenerationResult
from app.config import settings
import httpx
import time

class HTTPMusicGenerationProvider(MusicGenerationProvider):
    def generate_track(self, prompt: str, duration_sec: int, metadata: dict) -> MusicGenerationResult:
        if settings.DRY_RUN:
            return MusicGenerationResult(success=True, audio_bytes=b"fake_audio")

        headers = {"Authorization": f"Bearer {settings.MUSIC_PROVIDER_API_KEY}"}
        payload = {
            "prompt": prompt,
            "duration": duration_sec,
            "metadata": metadata
        }

        with httpx.Client(timeout=30) as client:
            resp = client.post(f"{settings.MUSIC_PROVIDER_BASE_URL}/generate", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            job_id = data.get("job_id")
            if not job_id:
                # Sync response
                return MusicGenerationResult(
                    success=True,
                    audio_file_path=data.get("audio_url"),
                    raw_response=data
                )

            # Async polling
            for _ in range(30):
                time.sleep(10)
                poll_resp = client.get(f"{settings.MUSIC_PROVIDER_BASE_URL}/jobs/{job_id}", headers=headers)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()

                if poll_data.get("status") == "completed":
                    audio_url = poll_data.get("audio_url")

                    # Download the final audio
                    out_path = f"/tmp/http_music_{job_id}.wav"
                    with client.stream("GET", audio_url) as r:
                        r.raise_for_status()
                        with open(out_path, "wb") as f:
                            for chunk in r.iter_bytes():
                                f.write(chunk)

                    return MusicGenerationResult(
                        success=True,
                        external_job_id=job_id,
                        audio_file_path=out_path,
                        raw_response=poll_data
                    )
                elif poll_data.get("status") in ["failed", "error"]:
                    return MusicGenerationResult(success=False, raw_response=poll_data)

            return MusicGenerationResult(success=False, raw_response={"error": "Polling timeout"})
