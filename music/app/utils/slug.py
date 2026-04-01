import re


def slugify(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")
