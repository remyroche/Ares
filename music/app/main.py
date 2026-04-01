from fastapi import FastAPI
from app.config import settings
from app.api import (
    routes_health,
    routes_admin,
    routes_tracks,
    routes_jobs,
    routes_analytics,
)
import logging

logging.basicConfig(level=logging.INFO if not settings.APP_DEBUG else logging.DEBUG)

app = FastAPI(title=settings.APP_NAME)

app.include_router(routes_health.router)
app.include_router(routes_admin.router, prefix="/admin", tags=["admin"])
app.include_router(routes_tracks.router, prefix="/tracks", tags=["tracks"])
app.include_router(routes_jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(routes_analytics.router, prefix="/analytics", tags=["analytics"])
