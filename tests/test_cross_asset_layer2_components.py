import unittest
import json
import os
import tempfile
import numpy as np
import pandas as pd

from src.training.steps.labeling.cross_asset_layer2_components import (
    PanelDataProcessor,
    CrossAssetPositionSizer,
    CrossAssetSurprises,
    GatingEngine,
    GatingConfig,
)
from src.training.steps.labeling.label_based_layer_2 import LabelBasedLayer2


class TestCrossAssetLayer2Components(unittest.TestCase):
    def test_panel_namespacing_and_index(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame({
            "close": np.linspace(100, 104, 5),
            "volume": np.arange(5) + 1,
            "foo": np.arange(5) * 2,
        }, index=idx)
        processor = PanelDataProcessor(vol_window=3, dvol_window=3)
        panel = processor.transform_to_panel({"AAA": df})

        self.assertIsInstance(panel.index, pd.MultiIndex)
        self.assertIn("raw__px", panel.columns)
        self.assertIn("y__ret_1", panel.columns)
        self.assertIn("raw__foo", panel.columns)
        self.assertTrue(all(col.startswith(("raw__", "y__")) or col == "ticker" for col in panel.columns))

    def test_closed_left_rolling_window(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        rolling = PanelDataProcessor._rolling_closed_left(series, window=3).mean()
        self.assertAlmostEqual(rolling.iloc[3], 2.0, places=6)
        self.assertTrue(np.isnan(rolling.iloc[0]))

    def test_entropy_filter_and_topk(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=2, freq="h"), ["AAA", "BBB", "CCC"]],
            names=["timestamp", "ticker"],
        )
        scores = pd.DataFrame({"score": [0.9, 0.1, 0.05, 0.8, 0.2, 0.1]}, index=idx)
        sizer = CrossAssetPositionSizer(calibration_window=3, method="isotonic")
        percentiles = sizer.compute_cross_asset_percentiles(scores, labels=None)
        entropy_pass = sizer.apply_entropy_filter(percentiles, threshold=1.0)
        selected = sizer.select_top_k(percentiles, k=2)

        self.assertEqual(entropy_pass.index.name, "timestamp")
        self.assertGreater(len(selected), 0)
        self.assertIn("percentile", percentiles.columns)

    def test_gating_reason_codes(self):
        panel_slice = pd.DataFrame(
            {
                "ca__ect_active": [False, True],
            },
            index=pd.Index(["AAA", "BBB"], name="ticker"),
        )
        portfolio_state = {
            "entropy_pass": False,
            "tail_corr_pass": False,
            "beta_cap_pass": True,
            "max_corr_pass": False,
        }
        gate_engine = GatingEngine()
        # Set persistence_bars=1 to ensure immediate failure triggering
        gate_df = gate_engine.evaluate(panel_slice, portfolio_state, GatingConfig(persistence_bars=1))
        self.assertIn("gate__reason_codes", gate_df.columns)
        self.assertIn("ect_inactive", gate_df.loc["AAA", "gate__reason_codes"])
        self.assertIn("entropy", gate_df.loc["AAA", "gate__reason_codes"])

    def test_ect_activation_output(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="h")
        panel_df = pd.DataFrame(
            {
                "raw__close": np.linspace(100, 110, 30),
                "raw__high": np.linspace(101, 111, 30),
                "raw__low": np.linspace(99, 109, 30),
                "raw__volume": np.linspace(10, 20, 30),
                "raw__px": np.linspace(100, 110, 30),
            },
            index=idx,
        )
        panel_df.index.name = "timestamp"
        panel_df["ticker"] = "AAA"
        panel_df = panel_df.reset_index().set_index(["timestamp", "ticker"])
        ms_features = pd.DataFrame({"ms__pca_0": np.linspace(1, 2, 30)}, index=idx)

        surprises = CrossAssetSurprises(quantiles=[0.5], ect_window=10, ect_half_life_bounds=(0.1, 50.0))
        out = surprises.fit_transform(panel_df, ms_features)
        self.assertIn("ca__ect_active", out.columns)
        self.assertTrue(out["ca__ect_active"].dtype == bool or out["ca__ect_active"].dropna().isin([True, False]).all())

    def test_robust_features_output(self):
        idx = pd.date_range("2024-01-01", periods=200, freq="5min")
        # Ensure positive prices for log
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.001, 200)))
        volume = np.random.lognormal(10, 1, 200)

        slice_df = pd.DataFrame({
            "raw__px": prices,
            "raw__volume": volume
        }, index=idx)

        market_returns = pd.Series(np.random.normal(0, 0.001, 200), index=idx)

        cas = CrossAssetSurprises()
        out = cas._compute_robust_features(slice_df, market_returns)

        expected_cols = [
            "ca__beta_short_w24", "ca__beta_long_w96", "ca__beta_shift",
            "ca__downside_beta_long_w96", "ca__active_ret_z_w48",
            "ca__active_ret_trend", "ca__active_ret_mr", "ca__active_ret_voladj",
            "ca__lead_lag_w48", "ca__lead_lag_sign_persistence",
            "ca__corr_shock", "ca__vol_shock",
            "ca__volume_shock", "ca__delta_volume_shock_12"
        ]

        for col in expected_cols:
            self.assertIn(col, out.columns)

        # Check that we have computed values (non-zero at tail)
        self.assertFalse(out.iloc[-1].isnull().any())
        self.assertNotEqual(out["ca__beta_short_w24"].iloc[-1], 0.0)

    def test_cross_asset_artifacts_written(self):
        idx = pd.date_range("2024-01-01", periods=40, freq="h")
        base = 100 + np.sin(np.linspace(0, 6, len(idx))) * 2
        noise = np.random.normal(0, 0.2, len(idx))

        def make_asset(multiplier: float) -> pd.DataFrame:
            close = base * multiplier + noise
            df = pd.DataFrame(
                {
                    "open": close + np.random.normal(0, 0.1, len(idx)),
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": np.linspace(10, 20, len(idx)),
                },
                index=idx,
            )
            return df

        cross_asset_data = {
            "AAA": make_asset(1.0),
            "BBB": make_asset(1.02),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            layer2 = LabelBasedLayer2(symbol="TEST")
            payload = layer2._run_cross_asset_pipeline(
                cross_asset_data,
                {
                    "cross_asset_run_id": "unit",
                    "dataset_tag": "unit",
                    "cross_asset_artifact_dir": tmpdir,
                    "enable_cross_asset_validation": True,
                    "enable_cross_asset_invariance": True,
                    "persist_market_state": False,
                    "ms_n_components": 2,
                },
            )
            artifacts = payload.get("artifacts", {})
            self.assertIn("leakage_report", artifacts)
            self.assertIn("validation_summary", artifacts)
            self.assertIn("invariance_report", artifacts)
            for key in ("leakage_report", "validation_summary", "invariance_report"):
                self.assertTrue(os.path.exists(artifacts[key]))
            with open(artifacts["leakage_report"], "r") as f:
                report = json.load(f)
            self.assertEqual(report.get("dataset_tag"), "unit")


if __name__ == "__main__":
    unittest.main()
