from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from app.api.deps import get_db_session
from app.models import PublishingJob
from app.schemas import JobResponse
from app.security import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("", response_model=List[JobResponse])
def get_jobs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db_session)):
    jobs = (
        db.query(PublishingJob)
        .order_by(PublishingJob.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return jobs


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: UUID, db: Session = Depends(get_db_session)):
    job = db.query(PublishingJob).filter(PublishingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
