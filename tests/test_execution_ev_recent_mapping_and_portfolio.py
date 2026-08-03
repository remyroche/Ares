from __future__ import annotations

import numpy as np
import pandas as pd
from types import SimpleNamespace

from scripts.replay_execution_ev_global_topk_portfolio import build_candidates
from scripts.run_execution_ev_recent_mapping_ablation import causal_mappings, run


def _identities(rows: int) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "__ts__": ts,
            "__symbol__": [f"S{i % 3}" for i in range(rows)],
            "side_name": np.where(np.arange(rows) % 2, "short", "long"),
            "candidate_id": [f"c{i}" for i in range(rows)],
        }
    )


def test_portfolio_candidates_use_exact_pooled_global_top_k() -> None:
    identities = _identities(20)
    score_col = "mapped_score"
    oof = identities.assign(
        mapped_score=np.arange(20, dtype=float),
        mapped_score__is_oof=True,
    )
    handoff = identities.assign(
        execution_decision_utc=identities["__ts__"],
        execution_label_end_utc=identities["__ts__"] + pd.Timedelta(hours=2),
        execution_exit_reason="timeout",
        execution_exit_hour=2.0,
        execution_gross_ev_12h=0.02,
        execution_net_ev_12h=0.01,
    )
    candidates = build_candidates(
        oof, handoff, score_col=score_col, top_k_fraction=0.10
    )
    assert len(candidates) == 2
    assert set(candidates["candidate_id"]) == {"c18", "c19"}
    assert candidates["timestamp"].nunique() == 2


def test_recent_mapping_never_uses_unresolved_or_future_targets() -> None:
    identities = _identities(96)
    decision = identities["__ts__"]
    frame = identities.assign(
        execution_decision_utc=decision,
        execution_label_end_utc=decision + pd.Timedelta(hours=2),
        execution_net_ev_12h=np.linspace(-0.02, 0.03, len(identities)),
        score=np.linspace(-1.0, 1.0, len(identities)),
    )
    first, _ = causal_mappings(
        frame,
        score_col="score",
        window_days=21,
        min_reference_rows=12,
        side_support_target=20.0,
    )
    changed = frame.copy()
    changed.loc[changed.index[-24:], "execution_net_ev_12h"] = 10.0
    second, _ = causal_mappings(
        changed,
        score_col="score",
        window_days=21,
        min_reference_rows=12,
        side_support_target=20.0,
    )
    compare = frame["execution_decision_utc"] < frame[
        "execution_decision_utc"
    ].iloc[-24]
    columns = [
        "causal_recent_percentile",
        "causal_recent_robust_z",
        "causal_recent_isotonic_ev",
        "causal_recent_side_isotonic_ev",
    ]
    for column in columns:
        np.testing.assert_allclose(
            first.loc[compare, column],
            second.loc[compare, column],
            equal_nan=True,
        )


def test_forward_extension_remains_non_oof_and_non_promotable(tmp_path) -> None:
    historical = _identities(72).assign(
        score=np.linspace(-1.0, 1.0, 72),
        score__is_oof=True,
        execution_ev_model_ablation_oof_fold=0,
        execution_decision_utc=lambda x: x["__ts__"],
        execution_net_ev_12h=np.linspace(-0.02, 0.03, 72),
    )
    handoff = historical.loc[
        :,
        [
            "__ts__",
            "__symbol__",
            "side_name",
            "candidate_id",
            "execution_decision_utc",
            "execution_net_ev_12h",
        ],
    ].assign(
        execution_label_end_utc=lambda x: x["execution_decision_utc"]
        + pd.Timedelta(hours=2)
    )
    forward = _identities(24)
    forward["__ts__"] += pd.Timedelta(days=4)
    forward["execution_decision_utc"] = forward["__ts__"]
    forward["execution_label_end_utc"] = (
        forward["execution_decision_utc"] + pd.Timedelta(hours=2)
    )
    forward["execution_net_ev_12h"] = np.linspace(-0.01, 0.02, len(forward))
    forward["forward_score"] = np.linspace(0.0, 1.0, len(forward))
    forward["is_oof"] = False
    forward["promotion_eligible"] = False
    oof_path = tmp_path / "oof.parquet"
    handoff_path = tmp_path / "handoff.parquet"
    forward_path = tmp_path / "forward.parquet"
    historical.to_parquet(oof_path, index=False)
    handoff.to_parquet(handoff_path, index=False)
    forward.to_parquet(forward_path, index=False)
    output = tmp_path / "mapped"
    run(
        SimpleNamespace(
            oof=oof_path,
            handoff=handoff_path,
            score_col="score",
            forward=forward_path,
            forward_score_col="forward_score",
            output_dir=output,
            window_days=21,
            min_reference_rows=2,
            side_support_target=2.0,
            top_k_fraction=0.10,
        )
    )
    mapped = pd.read_parquet(output / "mapped_oof.parquet")
    new = mapped.loc[
        mapped["evaluation_origin"].eq("frozen_final_fit_forward_oos")
    ]
    assert len(new) == len(forward)
    assert not new["promotion_eligible"].astype(bool).any()
    assert not new["causal_recent_isotonic_ev__is_oof"].astype(bool).any()
    assert new["causal_recent_isotonic_ev__is_forward_oos"].astype(bool).any()
