from PIL import Image
import sys


def generate(out_path):
    img = Image.new("RGB", (2048, 2048), color=(73, 109, 137))
    img.save(out_path, format="PNG")


if __name__ == "__main__":
    generate(sys.argv[1])
