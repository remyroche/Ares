from app.utils.ffmpeg import run_ffmpeg
import os


class RenderService:
    @staticmethod
    def render_youtube_video(
        track_id: str, audio_path: str, cover_path: str, output_path: str, title: str
    ):
        # 1920x1080 static cover with audio
        cmd = [
            "-loop",
            "1",
            "-i",
            cover_path,
            "-i",
            audio_path,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            "-vf",
            "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            output_path,
        ]
        run_ffmpeg(cmd)

    @staticmethod
    def render_short_videos(
        track_id: str,
        audio_path: str,
        cover_path: str,
        output_dir: str,
        captions: list[str],
    ) -> list[str]:
        os.makedirs(output_dir, exist_ok=True)
        outputs = []
        for i, caption in enumerate(captions):
            out_path = os.path.join(output_dir, f"short_{i+1}.mp4")
            # 1080x1920 static cover with audio, max 30s
            cmd = [
                "-loop",
                "1",
                "-i",
                cover_path,
                "-i",
                audio_path,
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-pix_fmt",
                "yuv420p",
                "-t",
                "30",
                "-shortest",
                "-vf",
                "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                out_path,
            ]
            run_ffmpeg(cmd)
            outputs.append(out_path)
        return outputs

    @staticmethod
    def render_compilation_video(
        compilation_id: str,
        audio_path: str,
        cover_path: str,
        output_path: str,
        title: str,
    ):
        cmd = [
            "-loop",
            "1",
            "-i",
            cover_path,
            "-i",
            audio_path,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            "-vf",
            "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            output_path,
        ]
        run_ffmpeg(cmd)
