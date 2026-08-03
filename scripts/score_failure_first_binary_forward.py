#!/usr/bin/env python3
"""Score the frozen historical binary detector on resolved current forward OOS.

The July 11--19 cohort is never added to discovery, HPO, feature selection, or
fitting.  It remains a cross-model forward-transfer diagnostic because the
detector was trained on a historical model generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.unsupervised_regime_learning.failure_first_health import (  # noqa: E402
    FailureHealthConfig,
    build_causal_decision_health,
    group_failure_bins_into_episodes,
)
from extreme_price_movements.unsupervised_regime_learning.failure_first_pipeline import (  # noqa: E402
    build_hourly_observable_state,
)
from scripts.run_failure_first_binary_ablation import (  # noqa: E402
    _classification_report,
    _economics,
    _prepare_panel,
)


DEFAULT_BUNDLE = Path(
    "data_perp/artifacts/failure_first_binary_ablation_20260726_v1/"
    "historical_binary_failure_detector_frozen.joblib"
)
DEFAULT_LEDGER = Path(
    "data_perp/artifacts/"
    "execution_ev_context_clean_recent_mapping_forward_july19_20260726_v1/"
    "mapped_oof.parquet"
)
DEFAULT_STATE = Path(
    "data_perp/artifacts/raw_market_state_backward_recurrence_20260726_v1/"
    "weekly_raw_state_diagnostic_rows.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/failure_first_binary_forward_july19_20260726_v1"
)
FORWARD_FLAG = "causal_recent_side_isotonic_ev__is_forward_oos"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--state-source", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    bundle = joblib.load(args.bundle)
    ledger = pd.read_parquet(args.ledger)
    state = pd.read_parquet(args.state_source)
    if FORWARD_FLAG not in ledger:
        raise KeyError(f"forward ledger missing {FORWARD_FLAG}")
    forward = ledger.loc[ledger[FORWARD_FLAG].fillna(False).astype(bool)].copy()
    if forward.empty:
        raise ValueError("no frozen forward-OOS rows")
    required_raw = [
        "causal_recent_side_isotonic_ev",
        "catboost__residual__without_hpo__all_features",
        "existing_alpha_ev",
        "base_oof_score",
        "base_margin_to_cutoff_z",
        "mkt_state__volatility_of_volatility_48__h0",
        "mkt_state__market_breadth_4h__h0",
    ]
    additions = [
        name
        for name in required_raw
        if name not in forward and name in state
    ]
    missing_ids = ~forward["candidate_id"].isin(state["candidate_id"])
    if missing_ids.any():
        raise ValueError(
            f"forward raw-H0 state missing {int(missing_ids.sum())} rows"
        )
    joined = forward.merge(
        state.loc[:, ["candidate_id", *additions]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    missing = [name for name in required_raw if name not in joined]
    if missing:
        raise ValueError(
            "forward binary source missing: " + ", ".join(missing)
        )
    if joined[required_raw].isna().any().any():
        raise ValueError("forward binary source has incomplete features")
    hourly_state, state_features = build_hourly_observable_state(
        joined, feature_columns=required_raw
    )
    frozen_base = [
        name
        for name in bundle.feature_columns
        if name.startswith("state__")
    ]
    if not set(frozen_base).issubset(state_features):
        raise ValueError("forward hourly base feature contract is incomplete")
    health_config = FailureHealthConfig(score_oof_col=FORWARD_FLAG)
    health, membership = build_causal_decision_health(
        ledger, config=health_config
    )
    episodes, _ = group_failure_bins_into_episodes(
        health, membership, config=health_config
    )
    panel, _ = _prepare_panel(
        health=health,
        hourly_state=hourly_state,
        frozen_base_features=state_features,
    )
    missing_features = sorted(
        set(bundle.feature_columns).difference(panel)
    )
    if missing_features:
        raise ValueError(
            "forward panel missing frozen detector features: "
            + ", ".join(missing_features)
        )
    predictions = panel.loc[
        :,
        ["execution_decision_utc", "side_name", "evaluation_origin"],
    ].copy()
    scored = bundle.score(panel)
    for name in scored:
        predictions[name] = scored[name].to_numpy()
    for name in (
        "target__failure_onset_within_3h",
        "target__failure_active_or_within_3h",
        "binary_failure_label_available_at",
    ):
        predictions[name] = panel[name].to_numpy()
    classification = _classification_report(predictions)
    overlay, economics = _economics(
        ledger, predictions, eligibility_flag=FORWARD_FLAG
    )
    predictions.to_parquet(
        output / "forward_predictions.parquet", index=False
    )
    overlay.to_parquet(
        output / "forward_candidate_overlay.parquet", index=False
    )
    health.to_parquet(output / "forward_health_6h.parquet", index=False)
    episodes.to_parquet(output / "forward_failure_episodes.parquet", index=False)
    report = {
        "schema": "failure_first_binary_forward_v1",
        "status": "CROSS_MODEL_FORWARD_TRANSFER_DIAGNOSTIC",
        "forward_rows": int(len(forward)),
        "forward_hours_scored": int(len(predictions)),
        "forward_start": forward["execution_decision_utc"].min(),
        "forward_end": forward["execution_decision_utc"].max(),
        "failure_bins": int(
            health["model_failure_bin"].fillna(False).sum()
        ),
        "failure_episodes": int(len(episodes)),
        "classification": classification,
        "economics": economics,
        "promotion_allowed": False,
        "promotion_reason": (
            "The detector is a historical-generation model; this resolved "
            "forward cohort is not current-model chronological detector OOF."
        ),
        "bundle_sha256": _sha256(Path(args.bundle)),
        "source_hashes": {
            str(Path(args.ledger)): _sha256(Path(args.ledger)),
            str(Path(args.state_source)): _sha256(Path(args.state_source)),
        },
    }
    _write_json(output / "report.json", report)
    return {
        "status": report["status"],
        "forward_rows": int(len(forward)),
        "forward_hours_scored": int(len(predictions)),
        "failure_episodes": int(len(episodes)),
        "promotion_allowed": False,
        "output_dir": str(output),
    }


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2))


if __name__ == "__main__":
    main()
