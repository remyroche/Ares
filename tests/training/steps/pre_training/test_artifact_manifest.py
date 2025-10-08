"""Tests for the artifact manifest workflow used by pre-training steps."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.steps.pre_training.artifacts.manifest import ArtifactManifest, DataLocator
from src.training.steps.pre_training.final_feature_selection_step import FinalFeatureSelectionStep


def _standardized_label_payload(rows: int) -> list[float]:
    """Create sigma-normalised label payload with deterministic variance."""
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, rows)
    values = (values - values.mean()) / values.std(ddof=1)
    return values.tolist()


class _ManifestAwareLookback:
    """Minimal consumer that mirrors the manifest-driven lookup logic."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("ManifestAwareLookback")

    def _normalize_labeling_result(self, payload: object) -> dict[str, pd.DataFrame] | None:
        if not payload:
            return None
        if isinstance(payload, dict):
            data = payload.get("labeled_data", payload)
        else:
            data = payload
        try:
            labeled_df = pd.DataFrame(data)
        except Exception:  # pragma: no cover - defensive
            return None
        if labeled_df.empty:
            return None
        return {"labeled_data": labeled_df}

    def load(self, symbol: str, exchange: str, timeframe: str) -> dict[str, pd.DataFrame] | None:
        manifest = ArtifactManifest()
        logical_name = DataLocator.build_logical_name(
            "market_analysis_multi_horizon_profit_labeler_outcome",
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
        )
        entry = manifest.get_latest(logical_name)
        if not entry:
            return None
        path = Path(entry.path)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as handle:
            outcome_data = json.load(handle)
        artifacts = outcome_data.get("artifacts", {}) if isinstance(outcome_data, dict) else {}
        mh_result = artifacts.get("multi_horizon_labeling_result") if isinstance(artifacts, dict) else None
        return self._normalize_labeling_result(mh_result)


def test_manifest_records_checksum_and_latest_entry(tmp_path, monkeypatch):
    """Registering artifacts should persist checksums and support lookup."""
    base_dir = tmp_path / "artifacts"
    monkeypatch.setenv("ARES_ARTIFACTS_DIR", str(base_dir))

    locator = DataLocator()
    manifest = ArtifactManifest()

    logical_name = DataLocator.build_logical_name(
        "market_analysis_multi_horizon_profit_labeler_outcome",
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
    )

    first_path, first_version = locator.resolve_artifact_path(
        "market_analysis_multi_horizon_profit_labeler_outcome",
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
        version="20240101_000000",
    )
    first_path.parent.mkdir(parents=True, exist_ok=True)
    first_path.write_text(json.dumps({"artifacts": {}}), encoding="utf-8")
    manifest.register(logical_name=logical_name, path=first_path, version=first_version)

    second_path, second_version = locator.resolve_artifact_path(
        "market_analysis_multi_horizon_profit_labeler_outcome",
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
        version="20240101_000100",
    )
    second_path.write_text(json.dumps({"artifacts": {}}), encoding="utf-8")
    manifest.register(logical_name=logical_name, path=second_path, version=second_version)

    reloaded = ArtifactManifest()
    latest = reloaded.get_latest(logical_name)
    assert latest is not None
    assert latest.version == second_version
    assert latest.path == str(second_path.resolve())
    assert latest.checksum == ArtifactManifest.compute_checksum(second_path)


def test_final_feature_selection_loads_from_manifest(tmp_path, monkeypatch):
    """The final feature selection step should resolve standardized labels via the manifest."""
    base_dir = tmp_path / "artifacts"
    monkeypatch.setenv("ARES_ARTIFACTS_DIR", str(base_dir))

    locator = DataLocator()
    manifest = ArtifactManifest()

    artifact_path, version = locator.resolve_artifact_path(
        "market_analysis_multi_horizon_profit_labeler_outcome",
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
        version="20240101_010101",
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    labels = {
        "immediate_opportunity": _standardized_label_payload(150),
        "short_term_opportunity": _standardized_label_payload(150),
    }
    outcome_payload = {
        "config": {"symbol": "ETHUSDT", "exchange": "binance", "timeframe": "1h"},
        "artifacts": {
            "standardized_output": {
                "labels": labels,
                "weights": {"small": 0.7, "medium": 0.3},
                "target_columns": list(labels.keys()),
                "validation_results": {"is_valid": True},
            }
        },
    }
    artifact_path.write_text(json.dumps(outcome_payload), encoding="utf-8")

    logical_name = DataLocator.build_logical_name(
        "market_analysis_multi_horizon_profit_labeler_outcome",
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
    )
    manifest.register(logical_name=logical_name, path=artifact_path, version=version)

    # Create an invalid fallback file to ensure the manifest entry is used first
    fallback_dir = tmp_path / "outcomes"
    fallback_dir.mkdir()
    (fallback_dir / "market_analysis_multi_horizon_profit_labeler_outcome_INVALID.json").write_text("{}", encoding="utf-8")

    step = FinalFeatureSelectionStep()
    result = step._load_target_data_from_standardized_format_sync("ETHUSDT", "binance", "1h", "unused")
    assert result is not None
    assert list(result.columns) == ["immediate_opportunity"]
    assert len(result) == 150


def test_feature_lookback_manifest_lookup(tmp_path, monkeypatch):
    """Feature lookback optimization should pull labeling artifacts from the manifest when present."""
    base_dir = tmp_path / "artifacts"
    monkeypatch.setenv("ARES_ARTIFACTS_DIR", str(base_dir))

    locator = DataLocator()
    manifest = ArtifactManifest()

    artifact_path, version = locator.resolve_artifact_path(
        "market_analysis_multi_horizon_profit_labeler_outcome",
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
        version="20240102_020202",
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    labeled_rows = [
        {"timestamp": "2024-01-01T00:00:00Z", "immediate_opportunity": 1.0},
        {"timestamp": "2024-01-01T01:00:00Z", "immediate_opportunity": -1.0},
        {"timestamp": "2024-01-01T02:00:00Z", "immediate_opportunity": 0.5},
    ]
    outcome_payload = {
        "config": {"symbol": "ETHUSDT", "exchange": "binance", "timeframe": "1h"},
        "artifacts": {"multi_horizon_labeling_result": {"labeled_data": labeled_rows}},
    }
    artifact_path.write_text(json.dumps(outcome_payload), encoding="utf-8")

    logical_name = DataLocator.build_logical_name(
        "market_analysis_multi_horizon_profit_labeler_outcome",
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
    )
    manifest.register(logical_name=logical_name, path=artifact_path, version=version)

    fallback_dir = tmp_path / "outcomes"
    fallback_dir.mkdir()
    (fallback_dir / "market_analysis_multi_horizon_profit_labeler_outcome_BAD.json").write_text("{}", encoding="utf-8")

    consumer = _ManifestAwareLookback()
    result = consumer.load("ETHUSDT", "binance", "1h")
    assert result is not None
    labeled_df = result["labeled_data"]
    assert isinstance(labeled_df, pd.DataFrame)
    assert not labeled_df.empty
