import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "validate_commit_message.py"


def validate(tmp_path: Path, subject: str) -> subprocess.CompletedProcess[str]:
    message_path = tmp_path / "COMMIT_EDITMSG"
    message_path.write_text(f"{subject}\n", encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(message_path)],
        capture_output=True,
        check=False,
        text=True,
    )


def test_accepts_valid_conventional_commit(tmp_path: Path) -> None:
    assert validate(tmp_path, "feat(inference): add latency guard").returncode == 0


def test_allows_automatic_commit_subjects(tmp_path: Path) -> None:
    assert (
        validate(tmp_path, "fixup! feat(inference): add latency guard").returncode == 0
    )


def test_rejects_invalid_commit_type(tmp_path: Path) -> None:
    result = validate(tmp_path, "add latency guard")

    assert result.returncode == 1
    assert "Use Conventional Commits" in result.stderr


def test_rejects_long_commit_subject(tmp_path: Path) -> None:
    result = validate(tmp_path, f"feat: {'x' * 67}")

    assert result.returncode == 1
    assert "at most 72 characters" in result.stderr
