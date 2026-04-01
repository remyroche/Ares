from sqlalchemy.orm import Session
from app.models import Track
from uuid import UUID

class TrackService:
    def __init__(self, db: Session):
        self.db = db

    def get_track_by_id(self, track_id: UUID) -> Track:
        return self.db.query(Track).filter_by(id=track_id).first()

    def get_tracks(self, skip: int = 0, limit: int = 100):
        return self.db.query(Track).order_by(Track.created_at.desc()).offset(skip).limit(limit).all()

    def update_track_status(self, track_id: UUID, new_status: str):
        track = self.get_track_by_id(track_id)
        if track:
            track.status = new_status
            self.db.commit()
            self.db.refresh(track)
        return track
