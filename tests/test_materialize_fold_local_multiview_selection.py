from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.materialize_fold_local_multiview_selection import (
    chronological_period_folds,
    materialize_fold_local_multiview_selection,
)
from scripts.materialize_multiview_regime_panel import sha256


def _write_signed_manifest(directory: Path, schema: str) -> None:
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"schema": schema}) + "\n", encoding="utf-8")
    (directory / "manifest.sha256").write_text(
        f"{sha256(manifest)}  manifest.json\n", encoding="utf-8"
    )


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    panel_root, ledger_root = tmp_path / "panel", tmp_path / "ledger"
    panel_root.mkdir()
    ledger_root.mkdir()
    source = pd.date_range("2024-01-01", "2024-07-31", freq="D", tz="UTC")
    x = np.linspace(-3.0, 3.0, len(source))
    panel = pd.DataFrame(
        {
            "source_utc": source,
            "calendar_segment_id": np.where(np.arange(len(source)) % 2, 7, 8),
            "mv__breadth__delta_1h": x,
            "mv__breadth__delta_3h": x + 1e-5 * np.sin(x),
            "mv__funding__robust_z_24h": np.sin(2.0 * x),
            "mv__funding__realized_vol_24h": np.abs(x),
            "mv__liquidity__volume_proxy__stress_1h": -x,
            "mv__dependence__eig1_share_24h": np.cos(1.3 * x),
        }
    )
    ledger = panel.loc[:, ["source_utc", "calendar_segment_id"]].copy()
    ledger["target__pooled_state"] = (np.sin(2.0 * x) > 0).astype(int)
    ledger["target__transition_active"] = (np.cos(1.3 * x) > 0).astype(int)
    ledger["target__available_utc"] = ledger["source_utc"] + pd.Timedelta(hours=18)
    panel.to_parquet(panel_root / "multiview_regime_features.parquet", index=False)
    ledger.to_parquet(ledger_root / "hourly_state_calendar.parquet", index=False)
    _write_signed_manifest(panel_root, "regime_multiview_panel_v1")
    _write_signed_manifest(ledger_root, "regime_episode_ledger_v1")
    return panel_root, ledger_root


def test_materializes_compact_fold_local_panels_with_train_only_availability(tmp_path: Path) -> None:
    panel_root, ledger_root = _roots(tmp_path)
    output = tmp_path / "out"
    report = materialize_fold_local_multiview_selection(
        panel_root=panel_root,
        ledger_root=ledger_root,
        output_dir=output,
        first_evaluation="2024-05-01",
        last_evaluation="2024-07-01",
        minimum_train_months=3,
        frequency="MS",
        config_kwargs={"max_correlation_rows": 100, "max_candidates_per_family_before_redundancy": 12},
    )

    regime = pd.read_parquet(output / "regime_oof_features.parquet")
    transition = pd.read_parquet(output / "transition_oof_features.parquet")
    audit = pd.read_parquet(output / "fold_audit.parquet")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert len(audit) == 3
    assert len(regime) == len(transition) == 92
    assert regime[["source_utc", "calendar_segment_id", "fold_id"]].equals(
        transition[["source_utc", "calendar_segment_id", "fold_id"]]
    )
    assert set(regime.columns).difference({"source_utc", "calendar_segment_id", "fold_id", "evaluation_start_utc", "evaluation_end_exclusive_utc"})
    assert audit["regime_selection_train_only"].all()
    assert audit["transition_selection_train_only"].all()
    assert (audit["train_latest_label_available_utc"] < audit["evaluation_start_utc"]).all()
    contract = manifest["probability_namespace_contract"]
    assert contract["canonical_regime_model_output"] == "regime_state_p__*"
    assert contract["interaction_discovery_regime_input"] == "regime_prob__*"
    assert report["counts"]["regime_oof_rows"] == len(regime)
    assert (output / "manifest.sha256").read_text().split()[0] == sha256(output / "manifest.json")


def test_fold_generation_enforces_label_availability_not_just_source_time() -> None:
    source = pd.date_range("2024-01-01", "2024-05-31", freq="D", tz="UTC")
    labels = pd.DataFrame({"source_utc": source, "target__available_utc": source + pd.Timedelta(hours=13)})
    labels.loc[labels["source_utc"].eq(pd.Timestamp("2024-03-31", tz="UTC")), "target__available_utc"] = pd.Timestamp("2024-05-02", tz="UTC")
    folds = chronological_period_folds(
        labels, first_evaluation="2024-05-01", last_evaluation="2024-05-01", minimum_train_months=3, frequency="MS"
    )

    assert len(folds) == 1
    start, _, train, _ = folds[0]
    assert labels.iloc[train]["target__available_utc"].max() < start
