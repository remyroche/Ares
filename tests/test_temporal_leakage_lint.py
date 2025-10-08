from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.training.steps.pre_training.validation import (
    TemporalLintError,
    lint_for_temporal_leakage,
    run_temporal_linting,
)


def _write(tmp_path: Path, filename: str, content: str) -> Path:
    path = tmp_path / filename
    path.write_text(textwrap.dedent(content))
    return path


def test_detects_center_true_violation(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "feature_center.py",
        """
        import pandas as pd

        def build_features(df: pd.DataFrame) -> pd.Series:
            return df["close"].rolling(window=5, center=True).mean()
        """,
    )

    violations = lint_for_temporal_leakage(file_path)
    assert any("center=True" in message for message in violations)


def test_detects_missing_closed_argument(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "feature_missing_closed.py",
        """
        def add_rolling_mean(df):
            df["ma"] = df["close"].rolling(window=10).mean()
            return df
        """,
    )

    violations = lint_for_temporal_leakage(file_path)
    assert any("closed" in message for message in violations)


def test_allows_explicit_closed_argument(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "feature_with_closed.py",
        """
        def add_rolling_mean(df):
            df["ma"] = df["close"].rolling(window=10, closed="left").mean()
            return df
        """,
    )

    violations = lint_for_temporal_leakage(file_path)
    assert violations == []


def test_negative_shift_requires_label_context(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "feature_shift.py",
        """
        def build_features(df):
            df["future_price"] = df["close"].shift(-1)
            return df
        """,
    )

    violations = lint_for_temporal_leakage(file_path)
    assert any("shift(-n)" in message for message in violations)


def test_negative_shift_allowed_in_label_context(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "label_shift.py",
        """
        def build_labels(df):
            df["target"] = df["close"].shift(-1)
            return df
        """,
    )

    violations = lint_for_temporal_leakage(file_path)
    assert violations == []


def test_negative_shift_allowed_with_comment(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "feature_shift_comment.py",
        """
        def build_features(df):
            df["future_price"] = df["close"].shift(-2)  # temporal-lint: allow-shift
            return df
        """,
    )

    violations = lint_for_temporal_leakage(file_path)
    assert violations == []


def test_center_true_allowed_with_comment(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "feature_center_comment.py",
        """
        def build_features(df):
            return df["close"].rolling(window=5, center=True, closed="right").mean()  # temporal-lint: allow-center
        """,
    )

    violations = lint_for_temporal_leakage(file_path)
    assert violations == []


def test_run_temporal_linting_aggregates_and_raises(tmp_path: Path) -> None:
    bad_file = _write(
        tmp_path,
        "feature_bad.py",
        """
        def build_features(df):
            df["future"] = df["close"].shift(-1)
            return df
        """,
    )

    with pytest.raises(TemporalLintError) as exc:
        run_temporal_linting([bad_file])

    message = str(exc.value)
    assert str(bad_file) in message
    assert "shift(-n)" in message


def test_run_temporal_linting_returns_results_when_not_raising(tmp_path: Path) -> None:
    bad_file = _write(
        tmp_path,
        "feature_bad.py",
        """
        def build_features(df):
            df["future"] = df["close"].shift(-1)
            return df
        """,
    )

    results = run_temporal_linting([bad_file], raise_on_violation=False)
    assert str(bad_file.resolve()) in results
    assert results[str(bad_file.resolve())]


def test_run_temporal_linting_skips_irrelevant_files(tmp_path: Path) -> None:
    irrelevant = _write(
        tmp_path,
        "utility.py",
        """
        def helper(df):
            return df.assign(flag=lambda frame: frame.index)
        """,
    )

    results = run_temporal_linting([irrelevant])
    assert results == {}


def test_ci_temporal_lint_runs_on_validation_package() -> None:
    results = run_temporal_linting(
        [Path("src/training/steps/pre_training/validation")],
        raise_on_violation=False,
    )
    assert results == {}
