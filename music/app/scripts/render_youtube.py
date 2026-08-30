import argparse
import sys
from app.services.render_service import RenderService


def main():
    parser = argparse.ArgumentParser(description="Render longform YouTube video")
    parser.add_argument("--track-id", required=True, help="Track ID")
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--cover", required=True, help="Input cover image")
    parser.add_argument("--output", required=True, help="Output MP4 file")
    parser.add_argument("--title", required=True, help="Track Title")

    args = parser.parse_args()

    try:
        RenderService.render_youtube_video(
            track_id=args.track_id,
            audio_path=args.audio,
            cover_path=args.cover,
            output_path=args.output,
            title=args.title,
        )
        print(f"Rendered {args.output} successfully.")
    except Exception as e:
        print(f"Failed to render video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
