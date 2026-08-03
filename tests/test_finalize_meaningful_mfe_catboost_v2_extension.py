from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.finalize_meaningful_mfe_catboost_v2_extension import (
    audit_incumbent_overlap,
    finalize,
)

SCORES = ("catboost_hard_ensemble", "catboost_competing_p_favorable")


def _oof(days: int, *, start: str = "2026-05-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=days, freq="D", tz="UTC")
    hard = np.resize(np.array([0.0, 1.0, 0.0, 1.0]), days)
    soft = np.where(hard > 0.5, 0.8, 0.2)
    ensemble = np.linspace(0.1, 0.9, days)
    competing = np.linspace(0.9, 0.1, days)
    return pd.DataFrame(
        {
            "__ts__": ts,
            "__symbol__": np.resize(np.array(["A", "B"]), days),
            "side_name": np.resize(np.array(["long", "short"]), days),
            "candidate_id": [f"candidate-{index}" for index in range(days)],
            "tb_hard_label": hard,
            "tb_soft_label": soft,
            "meaningful_mfe_reached": hard,
            "risk_class": hard.astype(int) * 2,
            "order_ambiguous": False,
            "catboost_hard_ensemble": ensemble,
            "catboost_competing_p_favorable": competing,
        }
    )


def _paired(oof: pd.DataFrame) -> pd.DataFrame:
    paired = oof.copy()
    paired["execution_net_ev_12h"] = np.where(
        paired["tb_hard_label"].gt(0.5),
        0.02,
        -0.01,
    )
    paired["execution_decision_utc"] = paired["__ts__"] + pd.Timedelta(hours=1)
    paired["execution_label_end_utc"] = paired["__ts__"] + pd.Timedelta(hours=13)
    paired["execution_label_available_at"] = paired["execution_label_end_utc"]
    return paired


def test_overlap_audit_rejects_any_prediction_delta() -> None:
    incumbent = _oof(8)
    extension = _oof(10)
    extension.loc[3, "catboost_hard_ensemble"] += 1e-12
    with pytest.raises(ValueError, match="catboost_hard_ensemble"):
        audit_incumbent_overlap(
            incumbent,
            extension,
            score_columns=SCORES,
        )


def test_finalize_rebuilds_reports_hashes_and_ceiling_without_fit(
    tmp_path: Path,
) -> None:
    incumbent_dir = tmp_path / "incumbent"
    extension_dir = tmp_path / "extension"
    incumbent_dir.mkdir()
    extension_dir.mkdir()
    incumbent = _oof(24)
    extension = _oof(28)
    extension.loc[: len(incumbent) - 1, list(SCORES)] = incumbent.loc[
        :, list(SCORES)
    ].to_numpy()
    incumbent_oof = incumbent_dir / "oof_predictions.parquet"
    incumbent_paired = incumbent_dir / "exact_policy_paired.parquet"
    extension_oof = extension_dir / "oof_predictions.parquet"
    extension_paired = extension_dir / "exact_policy_paired.parquet"
    incumbent.to_parquet(incumbent_oof, index=False)
    _paired(incumbent).to_parquet(incumbent_paired, index=False)
    extension.to_parquet(extension_oof, index=False)
    _paired(extension).to_parquet(extension_paired, index=False)
    incumbent_summary = incumbent_dir / "summary.json"
    incumbent_summary.write_text(
        json.dumps(
            {
                "schema": "old-contract",
                "status": "research",
                "reports": {name: {} for name in SCORES},
                "rows": {"valid_labels": len(incumbent)},
                "chronology": {"outer_oof": "causal"},
                "sources": {},
            }
        ),
        encoding="utf-8",
    )
    output = extension_dir / "summary.json"
    summary = finalize(
        incumbent_summary_path=incumbent_summary,
        incumbent_oof_path=incumbent_oof,
        incumbent_paired_path=incumbent_paired,
        extension_oof_path=extension_oof,
        extension_paired_path=extension_paired,
        output_path=output,
    )

    assert summary["recovery_contract"]["model_fit_performed"] is False
    assert summary["overlap_parity"]["global_prediction_max_abs_delta"] == 0.0
    assert summary["overlap_parity"]["extension_only_rows"] == 4
    assert summary["checkpoint_consistency"]["status"] == "passed"
    assert set(summary["reports"]) == set(SCORES)
    ensemble = summary["reports"]["catboost_hard_ensemble"]
    assert ensemble["clean_event"]["rows"] == 28
    assert ensemble["literal_event"]["rows"] == 28
    assert ensemble["exact_policy"]["rows"] == 28
    assert ensemble["post_21d_admission"]["contract"]["fit_days"] == 21
    assert len(ensemble["clean_event_by_side_month"]) == 2
    assert len(ensemble["exact_policy_by_side_month"]) == 2
    assert (
        summary["data_ceiling"]["oof_candidates"]["max"]
        == pd.Timestamp("2026-05-28", tz="UTC")
    )
    assert len(summary["sources"]["extension_oof_checkpoint"]["sha256"]) == 64
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        "finalized_from_completed_checkpoints_no_refit"
    )


def test_finalize_validates_all_sources_before_reading(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="missing required finalization sources"):
        finalize(
            incumbent_summary_path=missing / "summary.json",
            incumbent_oof_path=missing / "old_oof.parquet",
            incumbent_paired_path=missing / "old_paired.parquet",
            extension_oof_path=missing / "new_oof.parquet",
            extension_paired_path=missing / "new_paired.parquet",
            output_path=tmp_path / "summary.json",
        )
