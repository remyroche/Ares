from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_stage_c_side_ev_map_global_competition.py"
ROOT = SCRIPT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("stage_c_global_competition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _ledger(*, source: str, start: str, scores: list[float], ev: list[float], side: str) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(scores), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": [f"{source}_{side}_{index}" for index in range(len(scores))],
            "side_name": side,
            "archetype_policy_key": f"{side}_state",
            "score_meta_base_soft_label": scores,
            "ev_after_1pct": ev,
            "first_touch_gross": [value + 0.01 for value in ev],
            "clean_exec": [int(value > 0) for value in ev],
        }
    )


def _prepare(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = frame.copy()
    out["__archetype__"] = MODULE._normalise_archetype(out)
    out["__source__"] = source
    out["__raw_score__"] = out["score_meta_base_soft_label"].astype(float)
    return out


def test_train_only_side_ev_mapping_restores_common_ev_ordering() -> None:
    # Raw long scores are numerically higher, while the short score scale has
    # materially stronger net EV at its top. Raw cross-side competition would
    # select long; a train-only common EV map must select short.
    long_ref = _prepare(
        _ledger(source="long", start="2026-01-01", scores=[0.70, 0.80, 0.90, 0.95], ev=[-0.02, -0.01, 0.01, 0.02], side="long"),
        "long_model",
    )
    short_ref = _prepare(
        _ledger(source="short", start="2026-01-01", scores=[0.10, 0.20, 0.30, 0.40], ev=[-0.03, -0.01, 0.04, 0.06], side="short"),
        "short_model",
    )
    mapping = MODULE.fit_side_archetype_expected_ev_map(
        pd.concat([long_ref, short_ref], ignore_index=True), bins=2, min_group_rows=2, shrink_rows=0
    )
    long_eval = _prepare(_ledger(source="long_eval", start="2026-02-01", scores=[0.95], ev=[0.02], side="long"), "long_model")
    short_eval = _prepare(_ledger(source="short_eval", start="2026-02-01", scores=[0.40], ev=[0.06], side="short"), "short_model")
    raw_winner = pd.concat([long_eval, short_eval], ignore_index=True).sort_values("__raw_score__", ascending=False).iloc[0]
    mapped = pd.concat(
        [
            MODULE.apply_side_archetype_expected_ev_map(long_eval, mapping),
            MODULE.apply_side_archetype_expected_ev_map(short_eval, mapping),
        ],
        ignore_index=True,
    )
    mapped_winner = mapped.sort_values("expected_ev_side_archetype", ascending=False).iloc[0]
    assert raw_winner["side_name"] == "long"
    assert mapped_winner["side_name"] == "short"
    assert mapped_winner["expected_ev_side_archetype"] > 0.04


def test_mapping_reference_must_be_disjoint_and_strictly_prior() -> None:
    evaluation = _prepare(_ledger(source="eval", start="2026-02-01", scores=[0.9], ev=[0.01], side="long"), "model")
    overlap = evaluation.copy()
    with pytest.raises(ValueError, match="overlaps evaluation"):
        MODULE._validate_reference(overlap, evaluation, source_name="model")
    later = _prepare(_ledger(source="later", start="2026-03-01", scores=[0.9], ev=[0.01], side="long"), "model")
    with pytest.raises(ValueError, match="not strictly prior"):
        MODULE._validate_reference(later, evaluation, source_name="model")


def test_reader_rejects_a_non_one_percent_stored_cost_contract(tmp_path: Path) -> None:
    ledger = _ledger(source="bad", start="2026-01-01", scores=[0.5], ev=[0.01], side="long")
    ledger["first_touch_gross"] = 0.015
    path = tmp_path / "bad.parquet"
    ledger.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="exact stored 1%"):
        MODULE._read_ledger(path, source_name="bad", score_col=None)


def test_global_metrics_never_use_timestamp_local_selection() -> None:
    reference = _prepare(
        _ledger(source="ref", start="2026-01-01", scores=[0.1, 0.2, 0.8, 0.9], ev=[-0.02, -0.01, 0.02, 0.04], side="long"),
        "model",
    )
    evaluation = _prepare(
        _ledger(source="eval", start="2026-02-01", scores=[0.1, 0.9, 0.2, 0.8], ev=[-0.01, 0.04, -0.02, 0.02], side="long"),
        "model",
    )
    mapping = MODULE.fit_side_archetype_expected_ev_map(reference, bins=2, min_group_rows=2, shrink_rows=0)
    mapped = MODULE.apply_side_archetype_expected_ev_map(evaluation, mapping)
    metrics, _ = MODULE._metrics_for_variant(mapped, "model")
    top10 = metrics.loc[(metrics["scope"] == "global") & (metrics["top_frac"] == 0.10)].iloc[0]
    assert top10["selection_basis"] == "pooled_global_topk_after_train_only_side_archetype_ev_mapping"
    assert top10["selected_rows"] == 1
