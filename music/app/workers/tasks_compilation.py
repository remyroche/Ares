from celery import shared_task
from app.pipelines.weekly_compilation_pipeline import WeeklyCompilationPipeline


@shared_task(name="app.workers.tasks_compilation.weekly_compilation_task")
def weekly_compilation_task():
    return WeeklyCompilationPipeline.run()
