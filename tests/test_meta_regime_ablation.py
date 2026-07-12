from __future__ import annotations

from pathlib import Path

import pandas as pd

from extreme_price_movements.meta_regime_ablation import (
    FrozenPhaseStateContext,
    SideArchetypeIdentityContext,
    drop_oos_outcome_columns,
)
from scripts.report_train_meta_extended_pool_ablation_metrics import build_report


def test_frozen_phase_state_context_is_backward_only_and_outcome_free(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "states.parquet"
    states = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-04-01T00:00:00Z",
                    "2026-04-01T01:00:00Z",
                    "2026-04-01T00:00:00Z",
                    "2026-04-01T01:00:00Z",
                ],
                utc=True,
            ),
            "side_name": ["long", "long", "short", "short"],
            "state_phase__liquidation_onset": [0.1, 0.9, 0.2, 0.8],
            "state_phase__flush_exhaustion": [0.3, 0.7, 0.4, 0.6],
            # The context adapter must never load or emit this discovery target.
            "target_negative_surprise": [9.0, 9.0, 9.0, 9.0],
        }
    )
    states.to_parquet(state_path, index=False)
    context = FrozenPhaseStateContext(state_path, max_lag_minutes=60)
    rows = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-04-01T00:45:00Z",
                    "2026-04-01T01:00:00Z",
                    "2026-04-01T01:30:00Z",
                ],
                utc=True,
            ),
            "side_name": ["long", "short", "long"],
            "clean_exec": [1.0, 0.0, 1.0],
            "exec_margin": [0.01, -0.02, 0.01],
        }
    )
    oos = context.transform_oos(drop_oos_outcome_columns(rows))
    assert "target_negative_surprise" not in oos.columns
    assert "clean_exec" not in oos.columns
    assert oos["ctx_phase_available"].tolist() == [1.0, 1.0, 1.0]
    assert oos["ctx_phase_age_minutes"].tolist() == [45.0, 0.0, 30.0]
    # 00:45 long gets the 00:00 long state, not the future 01:00 observation.
    assert abs(float(oos.loc[0, "ctx_phase__liquidation_onset"]) - 0.1) < 1e-6
    assert abs(float(oos.loc[1, "ctx_phase__liquidation_onset"]) - 0.8) < 1e-6
    assert abs(float(oos.loc[2, "ctx_phase__liquidation_onset"]) - 0.9) < 1e-6
    assert all(name.startswith("ctx_phase") for name in oos.columns)


def test_side_archetype_identity_context_is_observable_and_outcome_free(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "handoff.parquet"
    source = pd.DataFrame(
        {
            "side_name": ["long", "short", "long"],
            "archetype_label_family": ["mixed", "breakout_impulse", "mixed"],
            "target_soft": [0.9, 0.1, 0.8],
        }
    )
    source.to_parquet(source_path, index=False)
    context = SideArchetypeIdentityContext.from_parquet(source_path)
    rows = pd.DataFrame(
        {
            "side_name": ["long", "short", "long"],
            "archetype_label_family": ["mixed", "breakout_impulse", "unseen"],
            "clean_exec": [1.0, 0.0, 1.0],
        }
    )
    oos = context.transform_oos(drop_oos_outcome_columns(rows))
    assert all(name.startswith("ctx_identity__") for name in oos.columns)
    assert oos.sum(axis=1).tolist() == [1.0, 1.0, 0.0]
    assert "clean_exec" not in oos.columns


def test_ablation_report_includes_base_score_and_monday_start_weeks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "matrix"
    base = root / "baseline_current_full_context"
    phase = root / "causal_phase_state_context"
    for arm_dir, lift in ((base, 0.0), (phase, 0.01)):
        shard_dir = arm_dir / "prediction_shards"
        shard_dir.mkdir(parents=True)
        rows = []
        for day in range(6, 13):
            for idx in range(12):
                rows.append(
                    {
                        "__ts__": f"2026-04-{day:02d}T00:00:00Z",
                        "__symbol__": f"S{idx}",
                        "side_name": "long" if idx % 2 else "short",
                        "archetype_label_family": "mixed",
                        "score_base": 0.5 + 0.02 * idx,
                        "score_meta_base_soft_label": 0.45 + 0.03 * idx,
                        "ev_after_1pct": 0.001 * idx + lift,
                        "exec_margin": 0.001 * idx,
                        "clean_exec": float(idx >= 6),
                        "dirty_positive": float(idx < 3),
                        "first_touch_bad_mae_1r": float(idx < 2),
                        "full_path_bad_mae_1r": float(idx < 3),
                        "timeout": 0.0,
                    }
                )
        pd.DataFrame(rows).to_parquet(
            shard_dir / "predictions_0001_2026-04.parquet", index=False
        )
        (arm_dir / "manifest.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "report"
    manifest = build_report(root_dir=root, out_dir=out, min_group_rows=1)
    metrics = pd.read_csv(manifest["outputs"]["all_metrics"])
    assert "base_score" in set(metrics["selector"])
    assert set(metrics.loc[metrics["scope"].eq("week"), "week_start"].dropna()) == {
        "2026-04-06"
    }
    assert "global_topk_side_archetype_family" in set(metrics["scope"])
    assert set(
        metrics.loc[metrics["scope"].str.startswith("global_topk_"), "selection_basis"]
    ) == {"global_topk"}
    base_delta = pd.read_csv(manifest["outputs"]["delta_vs_base_score"])
    assert not base_delta.empty
    assert "delta_vs_base_score__mean_ev_after_1pct" in base_delta.columns
