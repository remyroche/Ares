from celery import shared_task
from app.pipelines.daily_analytics_pipeline import DailyAnalyticsPipeline


@shared_task(name="app.workers.tasks_analytics.refresh_daily_analytics_task")
def refresh_daily_analytics_task():
    return DailyAnalyticsPipeline.run()
