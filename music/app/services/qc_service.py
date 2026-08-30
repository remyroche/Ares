from typing import Dict, Any
from app.utils.ffmpeg import get_duration
from app.config import settings
import subprocess
import re

class QCService:
    @staticmethod
    def analyze_audio(file_path: str) -> Dict[str, Any]:
        duration_sec = get_duration(file_path)

        silence_start_sec = 0.0
        silence_end_sec = 0.0
        clipping_detected = False

        if not settings.DEMO_MODE:
            # Silence detect
            cmd_silence = [
                "ffmpeg", "-i", file_path, "-af", "silencedetect=noise=-50dB:d=0.5", "-f", "null", "-"
            ]
            try:
                res_silence = subprocess.run(cmd_silence, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                out = res_silence.stderr

                # Check silence starts/ends
                starts = re.findall(r'silence_start: (\d+\.?\d*)', out)
                ends = re.findall(r'silence_end: (\d+\.?\d*)', out)

                if starts and float(starts[0]) == 0.0 and ends:
                    silence_start_sec = float(ends[0])
                if starts and float(starts[-1]) > duration_sec - 5.0:
                    silence_end_sec = duration_sec - float(starts[-1])
            except Exception:
                pass

            # Volume detect for clipping
            cmd_vol = [
                "ffmpeg", "-i", file_path, "-af", "volumedetect", "-f", "null", "-"
            ]
            try:
                res_vol = subprocess.run(cmd_vol, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                out = res_vol.stderr
                max_vol = re.search(r'max_volume: ([-\d\.]+) dB', out)
                if max_vol and float(max_vol.group(1)) >= 0.0:
                    clipping_detected = True
            except Exception:
                pass

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
        passed = (score >= settings.QC_MIN_SCORE and not clipping_detected and duration_sec >= settings.QC_MIN_DURATION_SEC)

        return {
            "duration_sec": duration_sec,
            "silence_start_sec": silence_start_sec,
            "silence_end_sec": silence_end_sec,
            "clipping_detected": clipping_detected,
            "qc_score": score,
            "passed": passed
        }
