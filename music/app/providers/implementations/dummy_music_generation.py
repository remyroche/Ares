from app.providers.base import MusicGenerationResult
from app.providers.music_generation import MusicGenerationProvider, MusicGenerationResult
import os


class DummyMusicGenerationProvider(MusicGenerationProvider):
    def generate_track(
        self, prompt: str, duration_sec: int, metadata: dict
    ) -> MusicGenerationResult:
        os.makedirs("exports", exist_ok=True)
        out_path = f"exports/dummy_track_{metadata.get('external_ref', '1')}.wav"
        # create silent wav file via shell or just touch it. Using ffmpeg for valid structure.
        os.system(
            f"ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=stereo -t {duration_sec} {out_path} >/dev/null 2>&1"
        )
        return MusicGenerationResult(success=True, audio_file_path=out_path)
