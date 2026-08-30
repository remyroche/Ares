from celery import Celery
from celery.schedules import crontab
from app.config import settings

celery_app = Celery(
    "music_factory_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.workers.tasks_generation",
        "app.workers.tasks_render",
        "app.workers.tasks_publish",
        "app.workers.tasks_compilation",
        "app.workers.tasks_analytics",
        "app.workers.tasks_maintenance",
    ],
)

celery_app.conf.update(
    timezone=settings.SCHEDULE_TIMEZONE,
    enable_utc=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    task_time_limit=3600,
    worker_concurrency=2,
)

# default schedule
celery_app.conf.beat_schedule = {
    "daily_generate_tracks": {
        "task": "app.workers.tasks_generation.generate_daily_tracks_task",
        "schedule": crontab(hour=2, minute=0),
    },
    "daily_publish_youtube": {
        "task": "app.workers.tasks_publish.publish_daily_youtube_task",
        "schedule": crontab(minute="*/30"),
    },
    "daily_publish_shorts": {
        "task": "app.workers.tasks_publish.publish_daily_shorts_task",
        "schedule": crontab(minute="*/20"),
    },
    "daily_publish_site_products": {
        "task": "app.workers.tasks_publish.publish_daily_site_products_task",
        "schedule": crontab(minute="0"),
    },
    "daily_refresh_analytics": {
        "task": "app.workers.tasks_analytics.refresh_daily_analytics_task",
        "schedule": crontab(hour=8, minute=0),
    },
    "weekly_compilation": {
        "task": "app.workers.tasks_compilation.weekly_compilation_task",
        "schedule": crontab(
            day_of_week=settings.WEEKLY_COMPILATION_DAY,
            hour=settings.WEEKLY_COMPILATION_HOUR,
            minute=0,
        ),
    },
    "retry_failed_jobs": {
        "task": "app.workers.tasks_maintenance.retry_failed_jobs_task",
        "schedule": crontab(minute="0"),
    },
    "cleanup_tmp_files": {
        "task": "app.workers.tasks_maintenance.cleanup_tmp_files_task",
        "schedule": crontab(hour=4, minute=30),
    },
}
