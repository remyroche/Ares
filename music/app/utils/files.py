import os


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def cleanup_file(path: str):
    if os.path.exists(path):
        os.remove(path)
