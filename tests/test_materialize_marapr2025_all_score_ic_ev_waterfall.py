from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_marapr2025_all_score_ic_ev_waterfall.py"
)
SPEC = importlib.util.spec_from_file_location("marapr_all_score_waterfall", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ledger() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in ("2025-03", "2025-04"):
        for index in range(2):
            gross = (index + (0 if month == "2025-03" else 1)) / 100.0
            cost = 0.001
            rows.append(
                {
                    "candidate_id": f"{month}-{index}",
                    "side_name": "long" if index == 0 else "short",
                    "__symbol__": f"asset-{index}",
                    "__ts__": pd.Timestamp(f"{month}-01T0{index}:00:00Z"),
                    "execution_label_end_utc": pd.Timestamp(
                        f"{month}-01T0{index}:00:00Z"
                    )
                    + pd.Timedelta(hours=12),
                    "candidate_month": month,
                    "execution_mfe_return_12h": gross + 0.01,
                    "execution_gross_ev_12h": gross,
                    "execution_cost_return": cost,
                    "execution_net_ev_12h": gross - cost,
                    "opportunity_gross_above_cost_0bps": gross - cost > 0.0,
                    "__first_touch_target_soft__": index / 2.0,
                    "score_base_alpha": index / 2.0,
                    "score_base_expected_ev": gross / 2.0,
                    "score_residual_delta_ev": gross / 4.0,
                    "score_residual_expected_ev": gross * 0.75,
                }
            )
    return pd.DataFrame(rows)


def _direct(ledger: pd.DataFrame) -> pd.DataFrame:
    result = ledger.loc[:, list(MODULE.IDENTITY_COLUMNS)].copy()
    result["q25_net_bps"] = [10.0, 20.0, 30.0, 40.0]
    result["execution_net_ev_12h"] = ledger["execution_net_ev_12h"].to_numpy()
    result["label_resolution_utc"] = ledger["execution_label_end_utc"].to_numpy()
    # The input legitimately contains the mapped output, but the new bridge
    # must never read or emit it.
    result["mapped_q25_bps"] = [100.0, 200.0, 300.0, 400.0]
    return result


def _manifests(tmp_path: Path, ledger_path: Path, direct_path: Path) -> tuple[Path, Path]:
    ledger_manifest = tmp_path / "ledger_manifest.json"
    ledger_manifest.write_text(
        json.dumps(
            {
                "schema": "historical_score_economics_conversion_ledgers_v1",
                "ledgers": [
                    {
                        "source_family": "canonical_residual_exact1m_current_spread_cf",
                        "path": str(ledger_path),
                        "sha256": _sha(ledger_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    direct_manifest = tmp_path / "direct_manifest.json"
    direct_manifest.write_text(
        json.dumps(
            {
                "schema": "cross_era_direct_net_quantile_challenger_v1",
                "outputs": {
                    "historical_oof_winner": {
                        "path": str(direct_path),
                        "sha256": _sha(direct_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return ledger_manifest, direct_manifest


def test_build_exactly_intersects_all_four_identity_fields_and_excludes_mapping() -> None:
    ledger = _ledger()
    direct = _direct(ledger)
    result = MODULE.build_all_score_frame(ledger, direct, expected_rows=4)
    assert len(result) == 4
    assert MODULE.DIRECT_BPS_COLUMN in result
    assert MODULE.DIRECT_RETURN_COLUMN in result
    assert result[MODULE.DIRECT_RETURN_COLUMN].iloc[-1] == pytest.approx(0.004)
    assert not any("mapped" in column.lower() for column in result)
    assert MODULE.DIRECT_BPS_COLUMN in MODULE.score_columns(result)
    assert MODULE.DIRECT_RETURN_COLUMN not in MODULE.score_columns(result)


def test_build_fails_closed_on_any_identity_coverage_gap() -> None:
    ledger = _ledger()
    with pytest.raises(ValueError, match="direct March-April rows"):
        MODULE.build_all_score_frame(ledger, _direct(ledger).iloc[:-1], expected_rows=4)


def test_build_fails_closed_when_realized_outcome_or_label_horizon_differs() -> None:
    ledger = _ledger()
    direct = _direct(ledger)
    direct.loc[0, "execution_net_ev_12h"] += 0.001
    with pytest.raises(ValueError, match="different realized net outcomes"):
        MODULE.build_all_score_frame(ledger, direct, expected_rows=4)

    direct = _direct(ledger)
    direct.loc[0, "label_resolution_utc"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="different label horizons"):
        MODULE.build_all_score_frame(ledger, direct, expected_rows=4)


def test_run_hash_binds_inputs_and_emits_every_declared_score_diagnostic(tmp_path: Path) -> None:
    ledger = _ledger()
    direct = _direct(ledger)
    ledger_path = tmp_path / "ledger.parquet"
    direct_path = tmp_path / "direct.parquet"
    ledger.to_parquet(ledger_path, index=False)
    direct.to_parquet(direct_path, index=False)
    ledger_manifest, direct_manifest = _manifests(tmp_path, ledger_path, direct_path)
    report = MODULE.run(
        ledger_path,
        ledger_manifest,
        direct_path,
        direct_manifest,
        tmp_path / "output",
        expected_rows=4,
    )
    assert report["status"] == "DIAGNOSTIC_ONLY_NO_MAPPING_NO_PROMOTION"
    assert report["score_contract"]["mapped_score_forbidden"]
    assert (tmp_path / "output" / "all_score_waterfall.parquet").exists()
    full_ic = pd.read_parquet(tmp_path / "output" / "full_ic.parquet")
    assert set(full_ic.score) == set(report["score_contract"]["declared_diagnostic_scores"])
    stored = pd.read_parquet(tmp_path / "output" / "all_score_waterfall.parquet")
    assert "mapped_q25_bps" not in stored


def test_run_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    ledger = _ledger()
    direct = _direct(ledger)
    ledger_path = tmp_path / "ledger.parquet"
    direct_path = tmp_path / "direct.parquet"
    ledger.to_parquet(ledger_path, index=False)
    direct.to_parquet(direct_path, index=False)
    ledger_manifest, direct_manifest = _manifests(tmp_path, ledger_path, direct_path)
    payload = json.loads(ledger_manifest.read_text())
    payload["ledgers"][0]["sha256"] = "0" * 64
    ledger_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        MODULE.run(
            ledger_path,
            ledger_manifest,
            direct_path,
            direct_manifest,
            tmp_path / "output",
            expected_rows=4,
        )
