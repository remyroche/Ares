from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_execution_ev_policy_labels",
    ROOT / "scripts" / "materialize_execution_ev_policy_labels.py",
)
assert SPEC and SPEC.loader
materializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materializer)


def _strategy(
    side: str,
    *,
    scope: str,
    archetype: str = "",
) -> dict[str, object]:
    canonical = (
        f"{side}__parent"
        if scope == "side_parent"
        else f"{side}__policy_archetype_{archetype}"
    )
    return {
        "selected": True,
        "canonical_strategy_id": canonical,
        "side": side,
        "exit_geometry_scope": scope,
        "policy_archetype": archetype,
        "cost_pct_per_side": 0.005,
        "size_power": 1.0,
        "sl_mult": 1.0,
        "trailing_activation_mult": 1.0,
        "trailing_activation_cap_pct": 0.0,
        "fixed_trailing_gap_mult": 0.5,
        "trailing_power": 1.0,
        "trailing_squash_divisor": 2.0,
        "giveback_beta": 0.5,
        "atr_power": 1.0,
        "atr_multiplier": 1.0,
        "policy_pathway_id": "joint_trailing_total_mfe_raw_bayesian_v1",
        "replay_timeframe": "1m",
        "trailing_activation_curve": "total_mfe",
    }


def _policy() -> dict[str, object]:
    local = "policy_archetype_long__long_mixed_wideslow_tentative"
    return {
        "exit_geometry_contract": {
            "replay_timeframe": "1m",
            "horizon_minutes": 3,
            "policy_pathway_id": "joint_trailing_total_mfe_raw_bayesian_v1",
            "trailing_activation_curve": "total_mfe",
        },
        "strategies": [
            _strategy("long", scope="side_parent"),
            _strategy("short", scope="side_parent"),
            _strategy("long", scope="side_archetype", archetype=local),
        ],
    }


def test_window_completeness_requires_every_exact_minute() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": np.arange(5, dtype=float) + 100.0,
            "high": np.arange(5, dtype=float) + 101.0,
            "low": np.arange(5, dtype=float) + 99.0,
            "close": np.arange(5, dtype=float) + 100.5,
        },
        index=index,
    )
    decisions = pd.Series([index[0], index[1]])
    assert materializer._window_completeness(bars, decisions, 3).tolist() == [
        True,
        True,
    ]
    bars.loc[index[2], "close"] = np.nan
    assert materializer._window_completeness(bars, decisions, 3).tolist() == [
        False,
        False,
    ]


def test_geometry_uses_observable_local_key_then_parent_fallback() -> None:
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short"],
            "__raw_policy_archetype__": [
                "long__long_mixed_wideslow_tentative",
                "base_rank_decile_0",
            ],
        }
    )
    resolved, audit = materializer._resolved_geometry(frame, _policy())
    assert resolved["execution_geometry_source"].tolist() == [
        "side_archetype",
        "side_parent_fallback",
    ]
    assert audit["side_archetype_rows"] == 1
    assert audit["side_parent_fallback_rows"] == 1


def test_stage_parser_exposes_audited_subset_flag() -> None:
    parsed = materializer._parser().parse_args(
        [
            "stage",
            "--candidates",
            "candidates.parquet",
            "--context",
            "context.parquet",
            "--path-targets",
            "targets.parquet",
            "--policy-json",
            "policy.json",
            "--output",
            "missing.parquet",
            "--manifest",
            "manifest.json",
            "--coverage-csv",
            "coverage.csv",
            "--allow-subset",
        ]
    )
    assert parsed.allow_subset is True


def test_policy_rejects_unconverted_minute_decay_fields(tmp_path: Path) -> None:
    policy = _policy()
    policy["strategies"][0]["adverse_decay_minutes"] = 30
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="minute-decay"):
        materializer._policy_contract(path)


def test_historical_lineage_binds_all_replay_inputs(tmp_path: Path) -> None:
    paths = {}
    for name in ("candidates", "context", "path_targets"):
        path = tmp_path / f"{name}.parquet"
        pd.DataFrame({"value": [name]}).to_parquet(path, index=False)
        paths[name] = path
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    lineage = {
        "schema": "historical_backcast_exact1m_label_inputs_v1",
        "outputs": {
            name: {"sha256": materializer._sha256(path)}
            for name, path in paths.items()
        },
        "policy_json": {"sha256": materializer._sha256(policy)},
        "evidence_scope": "frozen_backcast_diagnostic_not_oof",
        "lineage": "historical_frozen_backcast_exact1m_research_only",
        "oof_status": "not_oof",
        "execution_parity_claim": False,
        "promotion_eligible": False,
        "economics": "current_frozen_spread_counterfactual",
        "historical_l2_spread_available": False,
        "atr_contract": "diagnostic",
        "decision_to_path": "[signal+1h, signal+1h+12h)",
    }
    manifest = tmp_path / "lineage.json"
    manifest.write_text(json.dumps(lineage))
    resolved = materializer._historical_source_lineage(
        manifest,
        candidates_path=paths["candidates"],
        context_path=paths["context"],
        path_targets_path=paths["path_targets"],
        policy_path=policy,
    )
    assert resolved is not None
    assert resolved["oof_status"] == "not_oof"
    assert resolved["execution_parity_claim"] is False


def test_materialize_writes_signed_exact_policy_labels(
    tmp_path: Path, monkeypatch
) -> None:
    identity = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]),
            "__symbol__": ["BTC/USD:USD", "BTC/USD:USD"],
            "side_name": ["long", "short"],
            "candidate_id": ["long-row", "short-row"],
        }
    )
    candidates = tmp_path / "candidates.parquet"
    context = tmp_path / "context.parquet"
    targets = tmp_path / "targets.parquet"
    policy_path = tmp_path / "policy.json"
    spread = tmp_path / "spread.csv"
    identity.to_parquet(candidates, index=False)
    identity.assign(
        policy_archetype=[
            "long__long_mixed_wideslow_tentative",
            "base_rank_decile_0",
        ]
    ).to_parquet(context, index=False)
    identity.assign(
        __barrier_pct__=[0.01, 0.01],
        __path_auxiliary_atr_fraction__=[0.005, 0.005],
    ).to_parquet(targets, index=False)
    policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
    spread.write_text("symbol,p90_spread_bps\nBTC/USD:USD,20\n", encoding="utf-8")

    def bars(
        _data_root: Path,
        _symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        index = pd.date_range(start, end, freq="min", inclusive="left", tz="UTC")
        price = np.linspace(100.0, 103.0, len(index))
        return pd.DataFrame(
            {
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.5,
            },
            index=index,
        )

    monkeypatch.setattr(materializer, "_load_symbol_bars", bars)
    monkeypatch.setattr(
        materializer,
        "_policy_spread_baseline_audit",
        lambda: {"loaded": True, "source": str(spread.resolve())},
    )
    output = tmp_path / "labels.parquet"
    manifest_path = tmp_path / "manifest.json"
    missing = tmp_path / "missing.csv"
    materializer.materialize(
        SimpleNamespace(
            candidates=candidates,
            context=context,
            path_targets=targets,
            policy_json=policy_path,
            data_root=tmp_path,
            decision_delay_minutes=60,
            output=output,
            manifest=manifest_path,
            missing_csv=missing,
            spread_baseline=spread,
            batch_rows=2,
            allow_subset=False,
        )
    )
    labels = pd.read_parquet(output)
    assert len(labels) == 2
    assert labels["execution_geometry_source"].tolist() == [
        "side_archetype",
        "side_parent_fallback",
    ]
    assert np.allclose(
        labels["execution_gross_ev_12h"] - labels["execution_cost_return"],
        labels["execution_net_ev_12h"],
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == materializer.LABEL_SCHEMA
    assert manifest["exit_policy_contract"]["horizon_minutes"] == 3
    assert manifest["prediction_role_manifest_sha256"] == materializer._canonical_hash(
        manifest
    )
