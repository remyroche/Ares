from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reconstruct_weighted_packb_july_fold_score import (
    _append_candidate_base_score_context,
    _append_frozen_base_train_prior_rank,
    _apply_recovered_handoff_base_context,
    _assign_frozen_source_score_tags,
    _load_saved_base_models,
    _score_base,
    _validate_meta_prior_training_contract,
)


def test_frozen_source_tags_use_manifest_edges() -> None:
    candidates = pd.DataFrame(
        {
            "score": [0.15, 0.75, 0.85, 0.95],
            "side_name": ["long", "long", "short", "short"],
        }
    )
    contract = {
        "source_tag_mode": "fallback_side_score_intensity",
        "edges": [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, None],
    }

    out = _assign_frozen_source_score_tags(candidates, source_contract=contract)

    assert out["source_tag"].tolist() == [
        "long__model_candidate_background",
        "long__model_frontier_top30",
        "short__model_frontier_top20",
        "short__model_frontier_top10",
    ]


def test_handoff_base_context_recovers_group_parameters_exactly() -> None:
    train_rows: list[dict[str, object]] = []
    specs = {
        ("long", "long__trend"): (0.40, 0.50, 0.10),
        ("short", "short__reversal"): (0.30, 0.45, 0.05),
    }
    for (side, archetype), (cutoff, mean, std) in specs.items():
        for score in (mean - std, mean, mean + std):
            train_rows.append(
                {
                    "score": score,
                    "side_name": side,
                    "source_tag": archetype,
                    "base_margin_to_cutoff": score - cutoff,
                    "base_margin_to_cutoff_z": (score - cutoff) / std,
                    "base_signal_zscore_within_archetype": (score - mean) / std,
                }
            )
    train = pd.DataFrame(train_rows)
    candidates = pd.DataFrame(
        {
            "score": [0.65, 0.35],
            "side_name": ["long", "short"],
            "source_tag": ["long__trend", "short__reversal"],
        }
    )

    out = _apply_recovered_handoff_base_context(train, candidates)

    np.testing.assert_allclose(out["base_margin_to_cutoff"], [0.25, 0.05])
    np.testing.assert_allclose(out["base_margin_to_cutoff_z"], [2.5, 1.0])
    np.testing.assert_allclose(
        out["base_signal_zscore_within_archetype"], [1.5, -2.0]
    )


def test_candidate_base_context_uses_only_handoff_population() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-07-01T00:00:00Z"] * 4, utc=True
            ),
            "side_name": ["long", "long", "short", "short"],
            "score": [0.1, 0.3, 0.2, 0.8],
            "selected_top30": [False, True, False, True],
        }
    )

    out = _append_candidate_base_score_context(frame.loc[frame["selected_top30"]].copy())

    np.testing.assert_allclose(
        out["base_rank_pct_by_timestamp"], [0.5, 1.0]
    )
    np.testing.assert_allclose(
        out["base_rank_pct_by_timestamp_side"], [1.0, 1.0]
    )
    expected = (0.3 - np.mean([0.3, 0.8])) / np.std([0.3, 0.8], ddof=1)
    np.testing.assert_allclose(out.iloc[0]["base_score_z_by_timestamp"], expected)


def test_frozen_base_train_prior_rank_is_tie_aware() -> None:
    train = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-01T00:00:00Z"] * 4, utc=True
            ),
            "score": [0.1, 0.2, 0.2, 0.9],
        }
    )
    candidates = pd.DataFrame({"score": [0.2, 0.5]})
    out = _append_frozen_base_train_prior_rank(
        train,
        candidates,
        fit_end_exclusive=pd.Timestamp("2026-03-01", tz="UTC"),
    )
    np.testing.assert_allclose(
        out["base_score_rank_pct_train_prior"], [0.5, 0.75]
    )


def test_meta_prior_contract_rejects_missing_supervised_alias() -> None:
    frame = pd.DataFrame(
        {
            "clean_exec": [1.0],
            "dirty_positive": [0.0],
            "full_path_bad_mae_1r": [0.0],
            "first_touch_bad_mae_1r": [0.0],
            "timeout": [0.0],
            "exec_margin": [0.1],
        }
    )

    with pytest.raises(ValueError, match="clean_exec_label"):
        _validate_meta_prior_training_contract(frame)


def test_meta_prior_contract_records_outcome_coverage() -> None:
    frame = pd.DataFrame(
        {
            "clean_exec": [1.0, 0.0],
            "clean_exec_label": [1.0, 0.0],
            "dirty_positive": [0.0, 1.0],
            "full_path_bad_mae_1r": [0.0, 1.0],
            "first_touch_bad_mae_1r": [0.0, 1.0],
            "timeout": [0.0, 0.0],
            "exec_margin": [0.1, -0.1],
        }
    )

    report = _validate_meta_prior_training_contract(frame)

    assert report["finite_coverage"]["clean_exec_label"] == 1.0


class _SharedModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["feature_a"].to_numpy(dtype=np.float32)


def test_load_and_score_shared_base_checkpoint(tmp_path: Path) -> None:
    import joblib
    import json

    (tmp_path / "columns.json").write_text(
        json.dumps(
            {
                "feature_names": ["feature_a", "side"],
                "feature_names_by_side": {"shared": ["feature_a", "side"]},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"seed": 17}), encoding="utf-8"
    )
    joblib.dump(_SharedModel(), tmp_path / "base_model.joblib")

    models, contracts, seed = _load_saved_base_models(tmp_path)
    assert list(models) == ["shared"]
    assert contracts == {"shared": ["feature_a", "side"]}
    assert seed == 17

    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-07-01T00:00:00Z"] * 4, utc=True),
            "__symbol__": ["A", "B", "C", "D"],
            "side_name": ["long", "long", "short", "short"],
            "side": [1, 1, -1, -1],
            "feature_a": [0.8, 0.2, 0.7, 0.1],
        }
    )
    scored = _score_base(
        models=models,
        contracts=contracts,
        frame=frame,
        params={"model_side_scope": "shared"},
    )
    assert scored["base_input_complete"].eq(1).all()
    assert scored.groupby("side_name")["selected_top30"].sum().to_dict() == {
        "long": 1,
        "short": 1,
    }
