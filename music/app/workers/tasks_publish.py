from celery import shared_task
from app.pipelines.daily_youtube_pipeline import DailyYoutubePipeline
from app.pipelines.daily_shorts_pipeline import DailyShortsPipeline
from app.pipelines.daily_site_pipeline import DailySitePipeline


@shared_task(name="app.workers.tasks_publish.publish_daily_youtube_task")
def publish_daily_youtube_task():
    return DailyYoutubePipeline.run()


@shared_task(name="app.workers.tasks_publish.publish_daily_shorts_task")
def publish_daily_shorts_task():
    return DailyShortsPipeline.run()


@shared_task(name="app.workers.tasks_publish.publish_daily_site_products_task")
def publish_daily_site_products_task():
    return DailySitePipeline.run()
