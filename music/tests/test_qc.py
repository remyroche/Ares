from app.services.qc_service import QCService
import os


def test_qc_scoring():
    # Test fallback logic if file doesn't exist (we mock it indirectly via duration logic)
    # The actual implementation calls ffmpeg, so we'll test the logic scoring manually via mocking if we were unit testing fully.
    # We will test the basic structural return of QCService.analyze_audio using a generated silent file

    test_file = "/tmp/test_qc.mp3"
    os.system(
        f"ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=stereo -t 180 {test_file} >/dev/null 2>&1"
    )

    result = QCService.analyze_audio(test_file)
    assert result["duration_sec"] >= 179
    assert result["qc_score"] == 1.0
    assert result["passed"] is True

    os.remove(test_file)


def test_qc_failing():
    test_file = "/tmp/test_qc_fail.mp3"
    os.system(
        f"ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=stereo -t 30 {test_file} >/dev/null 2>&1"
    )

    result = QCService.analyze_audio(test_file)
    assert result["duration_sec"] < 150
    assert result["qc_score"] < 1.0
    assert result["passed"] is False

    os.remove(test_file)
