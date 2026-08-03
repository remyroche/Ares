#!/usr/bin/env python3
"""Score a frozen historical failure detector on the current OOF state panel.

This is a cross-model transfer diagnostic, never current-model detector OOF.
It tests whether a failure-state detector learned on the historical comparator
recognizes the two current-model failures and improves the unchanged pooled
global top-10% allocation.
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

from extreme_price_movements.unsupervised_regime_learning.failure_first_detector import (  # noqa: E402
    add_causal_bocpd_features,
)
from scripts.run_failure_first_regime_pipeline import (  # noqa: E402
    _detector_economics_report,
)


DEFAULT_BUNDLE = Path(
    "data_perp/artifacts/"
    "failure_first_regime_pipeline_historical_20260726_v12/"
    "failure_detector_latest_oof_fold.joblib"
)
DEFAULT_STATE = Path(
    "data_perp/artifacts/failure_first_regime_pipeline_20260726_v4/"
    "hourly_observable_state.parquet"
)
DEFAULT_EPISODES = Path(
    "data_perp/artifacts/failure_first_regime_pipeline_20260726_v4/"
    "failure_episodes.parquet"
)
DEFAULT_LEDGER = Path(
    "data_perp/artifacts/"
    "execution_ev_context_clean_recent_mapping_forward_july19_20260726_v1/"
    "mapped_oof.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/failure_first_detector_current_transfer_20260726_v6"
)


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
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _episode_detection(
    predictions: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    lookup = predictions.set_index("execution_decision_utc")
    risk = predictions["p_failure_destination_3h"]
    rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        onset = pd.Timestamp(episode.episode_onset_decision_utc)
        row: dict[str, Any] = {
            "episode_id": episode.episode_id,
            "episode_onset_decision_utc": onset,
            "severity_tier": episode.severity_tier,
        }
        for offset in (-6, -3, -1, 0):
            timestamp = onset + pd.Timedelta(hours=offset)
            if timestamp in lookup.index:
                local = lookup.loc[timestamp]
                if isinstance(local, pd.DataFrame):
                    local = local.iloc[0]
                value = float(local["p_failure_destination_3h"])
                transition = float(local["p_transition_within_3h"])
            else:
                value = transition = np.nan
            row[f"p_failure_destination_h{offset:+d}"] = value
            row[f"p_transition_h{offset:+d}"] = transition
            row[f"failure_risk_percentile_h{offset:+d}"] = (
                float(risk.le(value).mean()) if np.isfinite(value) else np.nan
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _monthly_economics(covered: pd.DataFrame) -> dict[str, Any]:
    from scripts.run_failure_first_regime_pipeline import _one_global_top_decile

    rows: dict[str, Any] = {}
    score_columns = (
        "failure_trust_adjusted_score",
        "failure_transition_adjusted_score",
        "failure_combined_adjusted_score",
    )
    for month, local in covered.groupby("evaluation_month", sort=True):
        base = _one_global_top_decile(
            local, score_col="causal_recent_side_isotonic_ev"
        )
        row: dict[str, Any] = {"mapped_score": base}
        for score_column in score_columns:
            adjusted = _one_global_top_decile(local, score_col=score_column)
            row[score_column] = adjusted
            row[f"{score_column}__incremental_bps"] = (
                adjusted.get("mean_net_ev_bps", np.nan)
                - base.get("mean_net_ev_bps", np.nan)
            )
        rows[str(month)] = row
    return rows


def _aggregate_transfer_economics(covered: pd.DataFrame) -> dict[str, Any]:
    from scripts.run_failure_first_regime_pipeline import _one_global_top_decile

    base = _one_global_top_decile(
        covered, score_col="causal_recent_side_isotonic_ev"
    )
    report: dict[str, Any] = {
        "mapped_score": base,
        "selection_contract": (
            "one pooled global top 10 percent across timestamps and sides"
        ),
    }
    for score_column in (
        "failure_trust_adjusted_score",
        "failure_transition_adjusted_score",
        "failure_combined_adjusted_score",
    ):
        adjusted = _one_global_top_decile(covered, score_col=score_column)
        report[score_column] = adjusted
        report[f"{score_column}__incremental_bps"] = (
            adjusted.get("mean_net_ev_bps", np.nan)
            - base.get("mean_net_ev_bps", np.nan)
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--hourly-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    bundle = joblib.load(args.bundle)
    state = pd.read_parquet(args.hourly_state)
    episodes = pd.read_parquet(args.episodes)
    ledger = pd.read_parquet(args.ledger)
    state["execution_decision_utc"] = pd.to_datetime(
        state["execution_decision_utc"], utc=True, errors="raise"
    )
    origins = (
        ledger.loc[
            ledger[
                "causal_recent_side_isotonic_ev__is_oof"
            ].fillna(False).astype(bool),
            ["execution_decision_utc", "evaluation_origin"],
        ]
        .drop_duplicates()
    )
    if origins["execution_decision_utc"].duplicated().any():
        raise ValueError("current hour spans multiple evaluation origins")
    panel = state.merge(
        origins,
        on="execution_decision_utc",
        how="inner",
        validate="one_to_one",
    )
    base_features = [
        name
        for name in bundle.feature_columns
        if not name.startswith("failure_bocpd_")
    ]
    missing = sorted(set(base_features).difference(panel.columns))
    if missing:
        raise ValueError(
            "current transfer panel is missing frozen features: "
            + ", ".join(missing)
        )
    signals: list[str] = []
    for token in (
        "volatility_of_volatility",
        "market_breadth_4h",
        "base_margin_to_cutoff_z",
        "catboost_entropy",
    ):
        match = next((name for name in base_features if token in name), None)
        if match is not None and match not in signals:
            signals.append(match)
    panel = add_causal_bocpd_features(
        panel,
        signal_columns=signals,
        timestamp_col="execution_decision_utc",
        group_columns=("side_name", "evaluation_origin"),
    )
    predictions = panel.loc[
        :, ["execution_decision_utc", "side_name", "evaluation_origin"]
    ].copy()
    scored = bundle.score(panel)
    for name in scored:
        predictions[name] = scored[name].to_numpy()
    detections = _episode_detection(predictions, episodes)
    covered, _ = _detector_economics_report(
        ledger, predictions
    )
    transition = predictions.loc[
        :,
        [
            "execution_decision_utc",
            "evaluation_origin",
            "p_transition_within_3h",
        ],
    ].drop_duplicates(["execution_decision_utc", "evaluation_origin"])
    covered = covered.merge(
        transition,
        on=["execution_decision_utc", "evaluation_origin"],
        how="left",
        validate="many_to_one",
    )
    mapped_score = covered["causal_recent_side_isotonic_ev"].astype(float)
    covered["failure_transition_adjusted_score"] = (
        mapped_score
        - covered["p_transition_within_3h"] * mapped_score.abs()
    )
    covered["failure_combined_adjusted_score"] = (
        mapped_score
        - covered[
            ["p_failure_destination_3h", "p_transition_within_3h"]
        ].max(axis=1)
        * mapped_score.abs()
    )
    aggregate_economics = _aggregate_transfer_economics(covered)
    monthly = _monthly_economics(covered)
    predictions.to_parquet(output / "current_transfer_predictions.parquet", index=False)
    detections.to_parquet(output / "current_failure_episode_detection.parquet", index=False)
    covered.to_parquet(output / "candidate_overlay.parquet", index=False)
    _write_json(output / "aggregate_economics.json", aggregate_economics)
    _write_json(output / "monthly_economics.json", monthly)
    report = {
        "schema": "failure_first_detector_current_transfer_v1",
        "status": "cross_model_transfer_diagnostic_not_current_detector_oof",
        "historical_bundle": str(Path(args.bundle).resolve()),
        "historical_bundle_sha256": _sha256(Path(args.bundle)),
        "current_rows": int(len(predictions)),
        "current_start": predictions["execution_decision_utc"].min(),
        "current_end": predictions["execution_decision_utc"].max(),
        "current_failure_episodes": int(len(detections)),
        "feature_count": int(len(bundle.feature_columns)),
        "feature_contract_exact": True,
        "episode_detection": detections.to_dict(orient="records"),
        "aggregate_economics": aggregate_economics,
        "monthly_economics": monthly,
        "promotion_allowed": False,
        "promotion_reason": (
            "Cross-model historical comparator transfer is diagnostic only; "
            "current-model detector OOF and fresh forward recurrence are absent."
        ),
    }
    _write_json(output / "report.json", report)
    return {
        "output_dir": output,
        "rows": int(len(predictions)),
        "episodes": int(len(detections)),
        "promotion_allowed": False,
    }


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
