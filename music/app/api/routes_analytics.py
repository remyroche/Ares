from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app.api.deps import get_db_session
from app.schemas import MetricSummary
from app.security import verify_api_key
from app.services.analytics_service import AnalyticsService

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("/top-performers", response_model=List[MetricSummary])
def get_top_performers(db: Session = Depends(get_db_session)):
    service = AnalyticsService(db)
    return service.get_top_performers()


@router.get("/summary")
def get_summary(db: Session = Depends(get_db_session)):
    return {"total_tracks_processed": 100, "status": "active"}
