from app.services.metadata_service import MetadataService
from app.models import Track


def test_metadata_generation():
    track = Track(brand="Test Brand", mood="sleepy", genre="lofi")

    meta = MetadataService.generate_track_metadata(track, 5)

    assert meta["title"] == "Tokyo Rain Study 005 | LoFi Rain Focus Ambient"
    assert "tokyo-rain-study-005" in meta["slug"]
    assert len(meta["hashtags"]) == 5
    assert len(meta["short_captions"]) == 3
    assert "Personal License" in meta["site_description"]
