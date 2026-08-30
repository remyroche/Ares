from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.models import SchedulerLock
from app.utils.time import utcnow
from datetime import timedelta


class SchedulerService:
    def __init__(self, db: Session):
        self.db = db

    def acquire_lock(self, lock_name: str, owner_id: str, duration_sec: int) -> bool:
        now = utcnow()
        lock = (
            self.db.query(SchedulerLock)
            .filter_by(name=lock_name)
            .with_for_update()
            .first()
        )

        if lock:
            if lock.locked_until > now:
                return False
            lock.locked_until = now + timedelta(seconds=duration_sec)
            lock.owner_id = owner_id
        else:
            lock = SchedulerLock(
                name=lock_name,
                locked_until=now + timedelta(seconds=duration_sec),
                owner_id=owner_id,
            )
            self.db.add(lock)

        try:
            self.db.commit()
            return True
        except IntegrityError:
            self.db.rollback()
            return False

    def release_lock(self, lock_name: str, owner_id: str):
        lock = (
            self.db.query(SchedulerLock)
            .filter_by(name=lock_name, owner_id=owner_id)
            .first()
        )
        if lock:
            self.db.delete(lock)
            self.db.commit()
