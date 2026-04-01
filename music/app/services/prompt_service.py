from datetime import date
from typing import List, Dict
import random
from app.config import settings


class PromptService:
    @staticmethod
    def generate_daily_prompts(count: int, target_date: date) -> List[Dict]:
        prompts = []
        base_seed = int(target_date.strftime("%Y%m%d"))

        for i in range(count):
            random.seed(base_seed + i)
            bpm = random.randint(72, 82)

            prompt_text = f"{settings.DEFAULT_GENRE} instrumental, tokyo night rain, soft keys, vinyl texture, calm focus, no vocals, subtle bass, sleepy atmosphere, loop friendly, {bpm} bpm"

            prompts.append(
                {
                    "prompt": prompt_text,
                    "genre": settings.DEFAULT_GENRE,
                    "mood": settings.DEFAULT_MOOD,
                    "duration_sec": settings.DEFAULT_TRACK_DURATION_SEC,
                    "bpm": bpm,
                    "external_ref": f"{target_date.strftime('%Y-%m-%d')}-{i+1}",
                }
            )

        return prompts
