from __future__ import annotations

import pandas as pd
import pytest

from scripts import run_continuous_context_residual_baselines as baselines


def _scores() -> pd.DataFrame:
    rows = []
    for month in range(1, 6):
        for offset in range(8):
            timestamp = pd.Timestamp(year=2024, month=month, day=2, tz="UTC") + pd.Timedelta(hours=offset)
            raw = (offset - 4) * 0.001
            rows.append(
                {
                    "candidate_id": f"{month}-{offset}", "__ts__": timestamp,
                    "__symbol__": "BTCUSDT", "side_name": "long" if offset % 2 else "short",
                    "execution_net_ev_12h": raw + (0.003 if month % 2 else -0.002),
                    "execution_gross_ev_12h": raw + (0.003 if month % 2 else -0.002) + 0.01,
                    "execution_cost_return": 0.01, "score_base_expected_ev": raw * .5,
                    "score_residual_expected_ev": raw, "residual_is_oof": True,
                }
            )
    return pd.DataFrame(rows)


def _context() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", "2024-06-01", freq="h", tz="UTC")
    return pd.DataFrame({
        "source_utc": index,
        "continuous_a": (pd.Series(range(len(index))) % 13).astype(float),
        "continuous_b": (pd.Series(range(len(index))) % 7).astype(float),
    })


def test_continuous_runner_is_prequential_and_p0_is_ranking_invariant(tmp_path) -> None:
    scores = tmp_path / "scores.parquet"
    context = tmp_path / "context.parquet"
    _scores().to_parquet(scores, index=False)
    _context().to_parquet(context, index=False)
    output = baselines.run(
        scores_path=scores, context_path=context, output_dir=tmp_path / "out",
        context_fields=("continuous_a", "continuous_b"), lookback_days=180,
        min_train_rows=5,
    )
    folds = pd.read_csv(output / "causal_folds.csv")
    strict = folds.loc[folds["mode"].eq("strict_prequential")]
    assert not strict.empty
    assert strict["ranking_invariant_p0_within_fold"].all()
    assert (pd.to_datetime(strict["train_label_available_max_utc"], utc=True) < pd.to_datetime(strict["evaluation_start_utc"], utc=True)).all()
    prediction = pd.read_parquet(output / "causal_oof_predictions.parquet")
    assert {"score__P0_location_only", "score__P2_linear_continuous", "score__P3_spline_additive", "score__P4_shallow_tree"}.issubset(prediction.columns)
    assert prediction["score__P0_location_only"].equals(prediction["score_residual_expected_ev"])
    assert (output / "worst_period_metrics.csv").is_file()
    assert (output / "transport_metrics.csv").is_file()


def test_cluster_or_membership_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        baselines._validate_context_fields(("market_regime__state_p_0",))


def test_exact_candidate_context_sidecar_preserves_identity_and_availability(tmp_path) -> None:
    scores = tmp_path / "scores.parquet"
    _scores().to_parquet(scores, index=False)
    sidecar = _scores().loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name"]].copy()
    sidecar["continuous_a"] = range(len(sidecar))
    sidecar["context_available_utc"] = sidecar["__ts__"]
    path = tmp_path / "candidate_sidecar.parquet"
    sidecar.to_parquet(path, index=False)
    panel, coverage = baselines._load_panel(scores_path=scores, context_path=path, context_fields=("continuous_a",))
    assert len(panel) == len(sidecar)
    assert set(coverage["context_join_mode"]) == {"exact_candidate_identity_sidecar"}
    sidecar["context_available_utc"] = sidecar["__ts__"] + pd.Timedelta(minutes=1)
    sidecar.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="looks ahead"):
        baselines._load_panel(scores_path=scores, context_path=path, context_fields=("continuous_a",))
