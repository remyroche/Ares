from celery import shared_task
from app.pipelines.daily_generation_pipeline import DailyGenerationPipeline


@shared_task(name="app.workers.tasks_generation.generate_daily_tracks_task")
def generate_daily_tracks_task():
    return DailyGenerationPipeline.run()
