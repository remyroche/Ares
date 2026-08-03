from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

STAGING = Path(__file__).resolve().parents[1] / "scripts"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from materialize_current_lineage_exact_failure_labels import (  # noqa: E402
    MAPPED_SCORE,
    build_current_exact_failure_labels,
)


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    hours = pd.date_range("2026-01-01", periods=40, freq="h", tz="UTC")
    rows = []
    for index, stamp in enumerate(hours):
        for candidate in range(4):
            score = float(candidate + index / 100.0)
            rows.append(
                {
                    "candidate_id": f"{index}_{candidate}",
                    "__ts__": stamp,
                    "__symbol__": f"S{candidate}",
                    "side_name": "long" if candidate % 2 else "short",
                    "execution_decision_utc": stamp + pd.Timedelta(hours=1),
                    "execution_label_end_utc": stamp
                    + pd.Timedelta(hours=13),
                    "execution_gross_ev_12h": score / 100.0,
                    "execution_net_ev_12h": score / 100.0 - 0.01,
                    MAPPED_SCORE: score,
                    f"{MAPPED_SCORE}__is_oof": True,
                    f"{MAPPED_SCORE}__is_forward_oos": False,
                }
            )
    overlay = pd.DataFrame(rows)
    health = pd.DataFrame(
        {
            "source_utc": hours,
            "execution_decision_utc": hours + pd.Timedelta(hours=1),
            "health__placeholder": 1.0,
        }
    )
    return overlay, health


def test_current_labels_select_one_global_mapped_book() -> None:
    overlay, health = _fixture()
    labelled, _, selected = build_current_exact_failure_labels(
        overlay, health, top_k_fraction=0.10
    )
    assert len(selected) == 16
    assert selected[MAPPED_SCORE].min() >= overlay[MAPPED_SCORE].nlargest(16).min()
    assert selected["__ts__"].nunique() < health["source_utc"].nunique()
    assert "label_window_complete" in labelled


def test_current_labels_reject_forward_rows() -> None:
    overlay, health = _fixture()
    overlay.loc[0, f"{MAPPED_SCORE}__is_oof"] = False
    overlay.loc[0, f"{MAPPED_SCORE}__is_forward_oos"] = True
    try:
        build_current_exact_failure_labels(
            overlay, health, top_k_fraction=0.10
        )
    except ValueError as error:
        assert "strict OOF" in str(error)
    else:
        raise AssertionError("forward rows must not enter strict current labels")


def test_current_labels_allow_explicit_retired_strict_forward_history() -> None:
    overlay, health = _fixture()
    overlay.loc[0, f"{MAPPED_SCORE}__is_oof"] = False
    overlay.loc[0, f"{MAPPED_SCORE}__is_forward_oos"] = True
    overlay["failure_first_score_is_strict_model_oos"] = True
    labelled, _, _ = build_current_exact_failure_labels(
        overlay,
        health,
        top_k_fraction=0.10,
        allow_resolved_forward=True,
    )
    assert len(labelled) == len(health)
