from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

import scripts.run_meta_v9_ev_mapped_side_residual_ablation as ablation


class _SerializableBooster:
    def __init__(self, text: str, params: dict[str, object]):
        self._text = text
        self.params = params

    def model_to_string(self) -> str:
        return self._text


def _fold_frame(start: str, end: str) -> pd.DataFrame:
    timestamps = pd.date_range(start, end, periods=2, tz="UTC")
    return pd.DataFrame(
        {
            "__ts__": timestamps,
            "__label_path_end_ts__": timestamps + pd.Timedelta(hours=25),
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"],
            "side_name": ["long", "short"],
        }
    )


def test_oos_fold_bundle_persists_complete_exact_state(tmp_path):
    train = _fold_frame("2026-03-01", "2026-03-31")
    test = _fold_frame("2026-04-01", "2026-04-30")
    models = {
        "long": _SerializableBooster("long model", {"seed": 10, "num_leaves": 15}),
        "short": _SerializableBooster("short model", {"seed": 11, "num_leaves": 15}),
    }
    paths = ablation._persist_oos_fold_bundle(
        out_dir=tmp_path,
        fold_id="2026-04-01_2026-05-01",
        oos_fit_mode="expanding_monthly",
        backbone_score="base",
        backbone_score_col="score_base",
        train=train,
        test=test,
        baseline_ev_map={"kind": "baseline"},
        residual_models=models,
        corrected_ev_map={"kind": "corrected"},
        alpha_by_side={"long": 0.6, "short": 0.2},
        features_by_side={"long": ["score_base", "long_feature"], "short": ["score_base", "short_feature"]},
        params_by_side={"long": {"max_depth": 4}, "short": {"max_depth": 3}},
    )

    bundle = joblib.load(paths["bundle_path"])
    manifest = json.loads(Path(paths["manifest_path"]).read_text(encoding="utf-8"))

    assert bundle["schema"] == ablation.OOS_FOLD_BUNDLE_SCHEMA
    assert bundle["baseline_ev_map"] == {"kind": "baseline"}
    assert bundle["corrected_ev_map"] == {"kind": "corrected"}
    assert bundle["alpha_by_side"] == {"long": 0.6, "short": 0.2}
    assert bundle["feature_contract"]["long"] == ["score_base", "long_feature"]
    assert bundle["configured_model_params_by_side"]["short"] == {"max_depth": 3}
    assert bundle["train_boundary"]["signal_timestamp_max"] == "2026-03-31T00:00:00+00:00"
    assert bundle["test_boundary"]["signal_timestamp_min"] == "2026-04-01T00:00:00+00:00"
    assert set(bundle["residual_models"]) == {"long", "short"}
    assert bundle["component_hashes"]["feature_contract_sha256"] == manifest[
        "component_hashes"
    ]["feature_contract_sha256"]
    assert manifest["hashes"]["bundle_sha256"] == ablation._sha256(Path(paths["bundle_path"]))
    assert set(manifest["hashes"]["residual_model_text_sha256"]) == {"long", "short"}


def test_oos_fold_bundle_explicitly_excludes_final_refit(tmp_path):
    frame = _fold_frame("2026-04-01", "2026-04-02")
    paths = ablation._persist_oos_fold_bundle(
        out_dir=tmp_path,
        fold_id="2026-04-01_2026-05-01",
        oos_fit_mode="expanding_monthly",
        backbone_score="base",
        backbone_score_col="score_base",
        train=frame,
        test=frame,
        baseline_ev_map={"kind": "baseline"},
        residual_models={"long": _SerializableBooster("long model", {})},
        corrected_ev_map={"kind": "corrected"},
        alpha_by_side={"long": 0.0, "short": 0.0},
        features_by_side={"long": ["score_base"], "short": ["score_base"]},
        params_by_side={"long": {}, "short": {}},
    )

    bundle = joblib.load(paths["bundle_path"])
    manifest = json.loads(Path(paths["manifest_path"]).read_text(encoding="utf-8"))

    assert bundle["final_refit_included"] is False
    assert manifest["final_refit_included"] is False
    assert "final_refit" not in bundle
