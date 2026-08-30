import argparse
import sys
from app.services.render_service import RenderService


def main():
    parser = argparse.ArgumentParser(description="Render short videos")
    parser.add_argument("--track-id", required=True, help="Track ID")
    parser.add_argument("--audio", required=True, help="Input preview audio file")
    parser.add_argument("--cover", required=True, help="Input cover image")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--captions", required=True, nargs="+", help="Captions for the shorts"
    )

    args = parser.parse_args()

    try:
        outputs = RenderService.render_short_videos(
            track_id=args.track_id,
            audio_path=args.audio,
            cover_path=args.cover,
            output_dir=args.output_dir,
            captions=args.captions,
        )
        print(f"Rendered {len(outputs)} shorts successfully.")
        for out in outputs:
            print(f" - {out}")
    except Exception as e:
        print(f"Failed to render shorts: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
