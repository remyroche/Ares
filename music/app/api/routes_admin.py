from fastapi import APIRouter, Depends
from app.security import verify_api_key
from app.workers.tasks_generation import generate_daily_tracks_task
from app.workers.tasks_publish import (
    publish_daily_youtube_task,
    publish_daily_shorts_task,
    publish_daily_site_products_task,
)
from app.workers.tasks_compilation import weekly_compilation_task
from app.workers.tasks_analytics import refresh_daily_analytics_task

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/run/daily-generate")
def run_daily_generate():
    task = generate_daily_tracks_task.delay()
    return {"message": "Daily generate task queued", "task_id": task.id}


@router.post("/run/publish-youtube")
def run_publish_youtube():
    task = publish_daily_youtube_task.delay()
    return {"message": "Publish YouTube task queued", "task_id": task.id}


@router.post("/run/publish-shorts")
def run_publish_shorts():
    task = publish_daily_shorts_task.delay()
    return {"message": "Publish Shorts task queued", "task_id": task.id}


@router.post("/run/publish-site")
def run_publish_site():
    task = publish_daily_site_products_task.delay()
    return {"message": "Publish Site task queued", "task_id": task.id}


@router.post("/run/weekly-compilation")
def run_weekly_compilation():
    task = weekly_compilation_task.delay()
    return {"message": "Weekly compilation task queued", "task_id": task.id}


@router.post("/run/refresh-analytics")
def run_refresh_analytics():
    task = refresh_daily_analytics_task.delay()
    return {"message": "Refresh analytics task queued", "task_id": task.id}
