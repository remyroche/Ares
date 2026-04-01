import os


def generate(out_path, duration):
    os.system(
        f"ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=stereo -t {duration} {out_path} >/dev/null 2>&1"
    )


if __name__ == "__main__":
    import sys

    generate(sys.argv[1], int(sys.argv[2]))
