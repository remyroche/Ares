import argparse
import sys
from app.services.render_service import RenderService


def main():
    parser = argparse.ArgumentParser(description="Render compilation video directly")
    parser.add_argument("--comp-id", required=True, help="Compilation ID")
    parser.add_argument("--audio", required=True, help="Input concatenated audio file")
    parser.add_argument("--cover", required=True, help="Input cover image")
    parser.add_argument("--output", required=True, help="Output MP4 file")
    parser.add_argument("--title", required=True, help="Compilation Title")

    args = parser.parse_args()

    try:
        RenderService.render_compilation_video(
            compilation_id=args.comp_id,
            audio_path=args.audio,
            cover_path=args.cover,
            output_path=args.output,
            title=args.title,
        )
        print(f"Rendered {args.output} successfully.")
    except Exception as e:
        print(f"Failed to render compilation video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
