from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from app.api.deps import get_db_session
from app.models import Track
from app.schemas import TrackResponse
from app.security import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("", response_model=List[TrackResponse])
def get_tracks(skip: int = 0, limit: int = 100, db: Session = Depends(get_db_session)):
    tracks = (
        db.query(Track)
        .order_by(Track.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return tracks


@router.get("/{track_id}", response_model=TrackResponse)
def get_track(track_id: UUID, db: Session = Depends(get_db_session)):
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    return track
