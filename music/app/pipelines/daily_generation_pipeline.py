from app.db import SessionLocal
from app.services.generation_service import GenerationService
from app.services.prompt_service import PromptService
from app.config import settings
from app.utils.time import localnow


class DailyGenerationPipeline:
    @staticmethod
    def run():
        db = SessionLocal()
        try:
            target_date = localnow().date()
            prompts = PromptService.generate_daily_prompts(
                settings.DAILY_TRACK_COUNT, target_date
            )

            gen_service = GenerationService(db)
            results = []

            for p in prompts:
                track = gen_service.generate_track(p)
                results.append(str(track.id))

            return results
        finally:
            db.close()
