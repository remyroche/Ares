import subprocess
import json
from typing import Dict, Any


def run_ffprobe(file_path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def get_duration(file_path: str) -> float:
    info = run_ffprobe(file_path)
    return float(info["format"]["duration"])


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-y"] + args
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
