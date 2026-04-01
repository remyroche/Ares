# State machine is implicitly handled inside GenerationService and PublishingService.
from app.enums import TrackStatus


def test_enums():
    assert TrackStatus.raw == "raw"
    assert TrackStatus.approved == "approved"
    assert TrackStatus.rejected == "rejected"
    assert TrackStatus.published == "published"
