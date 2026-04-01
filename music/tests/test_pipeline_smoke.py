import pytest
from app.pipelines.daily_generation_pipeline import DailyGenerationPipeline
from app.db import Base, engine, SessionLocal
from app.models import Track
import uuid


@pytest.fixture(scope="module")
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_daily_generation_smoke(setup_database):
    # In Demo Mode (set via config), this should execute entirely locally
    from app.config import settings

    settings.DEMO_MODE = True
    settings.DAILY_TRACK_COUNT = 1

    track_ids = DailyGenerationPipeline.run()

    assert len(track_ids) == 1

    db = SessionLocal()
    track_id_uuid = uuid.UUID(track_ids[0])
    track = db.query(Track).filter_by(id=track_id_uuid).first()
    assert track is not None
    assert track.status in ["approved", "rejected"]
    db.close()
