from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_mayjul2026_exact_allscore_ic_ev_waterfall.py"
)
SPEC = importlib.util.spec_from_file_location("mayjul_allscore", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sources() -> tuple[pd.DataFrame, ...]:
    labels: list[dict[str, object]] = []
    base: list[dict[str, object]] = []
    residual: list[dict[str, object]] = []
    direct: list[dict[str, object]] = []
    adapter: list[dict[str, object]] = []
    for month in ("2026-05", "2026-06", "2026-07"):
        for index in range(4):
            signal = pd.Timestamp(f"{month}-01T0{index}:00:00Z")
            decision = signal + pd.Timedelta(hours=1)
            end = decision + pd.Timedelta(hours=12)
            symbol = f"ASSET{index}/USD:USD"
            side = "long" if index % 2 == 0 else "short"
            candidate_id = f"{symbol}|{signal.strftime('%Y-%m-%dT%H:%M:%SZ')}|1h|{side}"
            gross = (index - 1) / 100.0
            label = {
                "candidate_id": candidate_id,
                "side_name": side,
                "__symbol__": symbol,
                "__ts__": signal,
                "execution_decision_utc": decision,
                "execution_label_end_utc": end,
                "execution_gross_ev_12h": gross,
                "execution_cost_return": 0.01,
                "execution_net_ev_12h": gross - 0.01,
                "execution_mfe_return_12h": gross + 0.03,
                "execution_mae_return_12h": -0.02,
                "execution_exit_reason": "timeout",
                "execution_exit_hour": 12.0,
            }
            labels.append(label)
            base.append(
                {
                    **{key: label[key] for key in MODULE.IDENTITY_COLUMNS},
                    "prediction": index / 10.0,
                    "__first_touch_target_soft__": index / 4.0,
                    "prediction_source": "outer_oof_fold_model",
                    "base_fold_fit_scope": "strict_prior_resolved_labels_side_local",
                    "validation_start": signal,
                    "train_decision_cutoff": signal - pd.Timedelta(hours=1),
                    "label_resolution_available_at": signal - pd.Timedelta(hours=2),
                    "oos_fold": month,
                }
            )
            residual.append(
                {
                    **{key: label[key] for key in MODULE.IDENTITY_COLUMNS},
                    "base_expected_ev": index / 100.0,
                    "residual_delta_ev": index / 200.0,
                    "residual_expected_ev": index / 80.0,
                    "residual_is_oof": True,
                    "residual_validation_start": signal,
                    "residual_train_decision_cutoff": signal - pd.Timedelta(hours=1),
                    "residual_prediction_available_at": decision,
                    "residual_oof_fold": month,
                    "__label_resolution_ts__": end + pd.Timedelta(hours=12),
                }
            )
            encoded = {
                **{key: label[key] for key in MODULE.IDENTITY_COLUMNS},
                "__symbol__": symbol.replace("/", "_"),
                "q25_net_bps": index * 10.0,
                "q50_net_bps": index * 11.0,
                "execution_net_ev_12h": gross - 0.01,
                "label_resolution_utc": end,
            }
            direct.append(encoded)
            adapter.append(
                {
                    **encoded,
                    "score_parent_bps": index * 9.0,
                    "score_adapter_bps": index * 8.0,
                    "score_reliability_bps": index * 7.0,
                    "score_adapter_reliability_bps": index * 6.0,
                    "fold": month,
                }
            )
    return tuple(
        pd.DataFrame(rows)
        for rows in (labels, base, residual, direct, adapter)
    )


def test_build_exact_allscore_frame_and_repairs_only_direct_symbols() -> None:
    frame, registry = MODULE.build_allscore_frame(
        *_sources(), expected_rows=12
    )
    assert len(frame) == 12
    assert len(registry) == len(MODULE.SCORE_SOURCES)
    assert set(MODULE.score_columns(frame)) == set(MODULE.SCORE_SOURCES)
    assert not any("mapped" in column.lower() for column in frame)
    assert frame["execution_label_end_utc"].equals(
        frame["execution_decision_utc"] + pd.Timedelta(hours=12)
    )


def test_direct_symbol_repair_fails_if_candidate_contract_disagrees() -> None:
    labels, base, residual, direct, adapter = _sources()
    direct.loc[0, "__symbol__"] = "WRONG_USD:USD"
    with pytest.raises(ValueError, match="symbol encoding"):
        MODULE.build_allscore_frame(
            labels, base, residual, direct, adapter, expected_rows=12
        )


def test_build_rejects_wrong_horizon_or_economics() -> None:
    labels, base, residual, direct, adapter = _sources()
    labels.loc[0, "execution_label_end_utc"] += pd.Timedelta(hours=12)
    with pytest.raises(ValueError, match="exact 12h"):
        MODULE.build_allscore_frame(
            labels, base, residual, direct, adapter, expected_rows=12
        )

    labels, base, residual, direct, adapter = _sources()
    direct.loc[0, "execution_net_ev_12h"] += 0.001
    with pytest.raises(ValueError, match="realized net differs"):
        MODULE.build_allscore_frame(
            labels, base, residual, direct, adapter, expected_rows=12
        )


def test_build_rejects_non_oof_base_or_residual() -> None:
    labels, base, residual, direct, adapter = _sources()
    base.loc[0, "prediction_source"] = "final_refit"
    with pytest.raises(ValueError, match="base prediction"):
        MODULE.build_allscore_frame(
            labels, base, residual, direct, adapter, expected_rows=12
        )

    labels, base, residual, direct, adapter = _sources()
    residual.loc[0, "residual_is_oof"] = False
    with pytest.raises(ValueError, match="residual stream"):
        MODULE.build_allscore_frame(
            labels, base, residual, direct, adapter, expected_rows=12
        )
