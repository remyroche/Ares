from datetime import datetime, timezone
import zoneinfo
from app.config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def localnow() -> datetime:
    return datetime.now(zoneinfo.ZoneInfo(settings.SCHEDULE_TIMEZONE))
