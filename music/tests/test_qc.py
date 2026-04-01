from app.services.qc_service import QCService
from app.config import settings
import os

def test_qc_scoring():
    settings.DEMO_MODE = True
    test_file = "/tmp/test_qc.mp3"
    os.system(f"ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=stereo -t 180 {test_file} >/dev/null 2>&1")

    result = QCService.analyze_audio(test_file)
    assert result["duration_sec"] >= 179
    assert result["qc_score"] == 1.0
    assert result["passed"] is True

    os.remove(test_file)

def test_qc_failing():
    settings.DEMO_MODE = True
    test_file = "/tmp/test_qc_fail.mp3"
    os.system(f"ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=stereo -t 30 {test_file} >/dev/null 2>&1")

    result = QCService.analyze_audio(test_file)
    assert result["duration_sec"] < 150
    assert result["qc_score"] < 1.0
    assert result["passed"] is False

    os.remove(test_file)
