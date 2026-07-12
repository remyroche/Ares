"""Validate Conventional Commit subjects for the commit-msg hook."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED_TYPES = (
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "test",
)
MAX_SUBJECT_LENGTH = 72
CONVENTIONAL_SUBJECT = re.compile(
    rf"^(?:{'|'.join(ALLOWED_TYPES)})(?:\([a-z0-9][a-z0-9._/-]*\))?!?: \S.*$"
)
AUTOMATIC_SUBJECTS = ("Merge ", "Revert ", "fixup! ", "squash! ")


def read_subject(message_path: Path) -> str:
    for line in message_path.read_text(encoding="utf-8").splitlines():
        subject = line.strip()
        if subject and not subject.startswith("#"):
            return subject
    return ""


def validate_subject(subject: str) -> str | None:
    if not subject:
        return "Commit message subject must not be empty."
    if subject.startswith(AUTOMATIC_SUBJECTS):
        return None
    if len(subject) > MAX_SUBJECT_LENGTH:
        return f"Commit subject must be at most {MAX_SUBJECT_LENGTH} characters."
    if subject.endswith("."):
        return "Commit subject must not end with a period."
    if not CONVENTIONAL_SUBJECT.fullmatch(subject):
        allowed = ", ".join(ALLOWED_TYPES)
        return (
            "Use Conventional Commits, for example 'feat(inference): add latency guard'. "
            f"Allowed types: {allowed}."
        )
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "usage: validate_commit_message.py <commit-message-file>", file=sys.stderr
        )
        return 2

    error = validate_subject(read_subject(Path(argv[1])))
    if error:
        print(f"Invalid commit message: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
