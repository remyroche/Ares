import numpy as np
import pandas as pd

from extreme_price_movements.scripts import live_backtest_decision_reconciliation as recon


class _NanFeatureStore:
    def latest_values_at(self, key, symbols, ts):
        return pd.Series([np.nan], index=symbols)


def test_training_feature_values_falls_back_to_live_latest_matrix(tmp_path, monkeypatch):
    run_id = "20260101_000000"
    signal_ts = pd.Timestamp("2026-01-01 03:00:00", tz="UTC")
    matrix_dir = tmp_path / "features" / run_id / "_live_latest_matrix"
    matrix_dir.mkdir(parents=True)
    matrix = pd.DataFrame(
        {
            "feature_a": [1.25],
            "feature_b": [-0.75],
        },
        index=pd.Index(["AAA/USD:USD"], name="symbol"),
    )
    matrix.to_parquet(matrix_dir / "matrix_20260101T030000Z.parquet")
    monkeypatch.setattr(recon, "load_features_selected", lambda *args, **kwargs: _NanFeatureStore())

    values = recon._training_feature_values(
        feature_source_run_id=run_id,
        data_root=str(tmp_path),
        signal_ts=signal_ts,
        symbol="AAA/USD:USD",
        feature_keys=["feature_a", "feature_b"],
    )

    assert values["feature_a"] == 1.25
    assert values["feature_b"] == -0.75
