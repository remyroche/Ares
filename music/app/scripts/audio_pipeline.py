import argparse
import os
import json
import sys
from app.utils.ffmpeg import run_ffmpeg
from app.services.qc_service import QCService


def process_audio(input_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    master = os.path.join(output_dir, "master.mp3")
    preview = os.path.join(output_dir, "preview_30s.mp3")
    loop = os.path.join(output_dir, "loop.mp3")
    report = os.path.join(output_dir, "qc_report.json")

    try:
        run_ffmpeg(["-i", input_file, "-c:a", "libmp3lame", master])
        run_ffmpeg(["-i", input_file, "-c:a", "libmp3lame", "-t", "30", preview])
        run_ffmpeg(["-i", input_file, "-c:a", "libmp3lame", loop])

        qc_res = QCService.analyze_audio(master)

        with open(report, "w") as f:
            json.dump(qc_res, f, indent=2)

        if not qc_res["passed"]:
            print("QC Failed")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    process_audio(args.input, args.output_dir)
