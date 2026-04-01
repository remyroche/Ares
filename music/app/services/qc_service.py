from typing import Dict, Any
from app.utils.ffmpeg import get_duration
from app.config import settings


class QCService:
    @staticmethod
    def analyze_audio(file_path: str) -> Dict[str, Any]:
        duration_sec = get_duration(file_path)

        # dummy implementation for silence and clipping to simplify ffmpeg dependencies
        silence_start_sec = 0.0
        silence_end_sec = 0.0
        clipping_detected = False

        # In a real implementation:
        # run ffmpeg silencedetect and parse output
        # run ffmpeg volumedetect to check for clipping

        score = 1.0
        if duration_sec < settings.QC_MIN_DURATION_SEC:
            score -= 0.35
        if silence_start_sec > settings.QC_MAX_SILENCE_START_SEC:
            score -= 0.20
        if silence_end_sec > settings.QC_MAX_SILENCE_END_SEC:
            score -= 0.20
        if clipping_detected:
            score -= 0.35

        score = max(0.0, min(1.0, score))
        passed = (
            score >= settings.QC_MIN_SCORE
            and not clipping_detected
            and duration_sec >= settings.QC_MIN_DURATION_SEC
        )

        return {
            "duration_sec": duration_sec,
            "silence_start_sec": silence_start_sec,
            "silence_end_sec": silence_end_sec,
            "clipping_detected": clipping_detected,
            "qc_score": score,
            "passed": passed,
        }
