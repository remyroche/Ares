#!/usr/bin/env python3
"""Ablate a direct binary failure detector without a regime taxonomy.

The historical strict-OOF comparator is used for chronological model/feature
research.  Its frozen winner is then scored without refitting on the current
strict-OOF panel as a cross-model transfer diagnostic.  Neither route is
promotion evidence for a current-model detector.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.unsupervised_regime_learning.failure_first_binary import (  # noqa: E402
    BinaryFailureDetectorConfig,
    add_causal_transition_deltas,
    build_hourly_binary_failure_targets,
    chronological_binary_failure_oof,
    fit_binary_failure_detector,
)
from extreme_price_movements.unsupervised_regime_learning.failure_first_detector import (  # noqa: E402
    add_causal_bocpd_features,
)
from scripts.run_failure_first_regime_pipeline import (  # noqa: E402
    _one_global_top_decile,
)


HISTORICAL_PIPELINE = Path(
    "data_perp/artifacts/"
    "failure_first_regime_pipeline_historical_20260726_v12"
)
CURRENT_PIPELINE = Path(
    "data_perp/artifacts/failure_first_regime_pipeline_20260726_v4"
)
HISTORICAL_LEDGER = Path(
    "data_perp/artifacts/failure_first_historical_backfill_20260726_v3/"
    "mapped_strict_oof_ledger.parquet"
)
CURRENT_LEDGER = Path(
    "data_perp/artifacts/"
    "execution_ev_context_clean_recent_mapping_forward_july19_20260726_v1/"
    "mapped_oof.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/failure_first_binary_ablation_20260726_v1"
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


def _signal_columns(state_features: list[str]) -> list[str]:
    signals: list[str] = []
    for token in (
        "volatility_of_volatility",
        "market_breadth_4h",
        "base_margin_to_cutoff_z",
    ):
        match = next(
            (name for name in state_features if token in name), None
        )
        if match is not None and match not in signals:
            signals.append(match)
    return signals


def _prepare_panel(
    *,
    health: pd.DataFrame,
    hourly_state: pd.DataFrame,
    frozen_base_features: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    targets = build_hourly_binary_failure_targets(health)
    state = hourly_state.copy()
    state["execution_decision_utc"] = pd.to_datetime(
        state["execution_decision_utc"], utc=True, errors="raise"
    )
    available_state = [
        name for name in state if name.startswith("state__")
    ]
    if frozen_base_features is None:
        market = [
            name for name in available_state if "state__mkt_state_" in name
        ][:20]
        health_features = [
            name for name in available_state if name not in market
        ][:5]
        base = [*market, *health_features]
    else:
        base = list(frozen_base_features)
        missing = sorted(set(base).difference(state))
        if missing:
            raise ValueError(
                "frozen binary detector features missing: "
                + ", ".join(missing)
            )
        market = [
            name for name in base if "state__mkt_state_" in name
        ]
        health_features = [name for name in base if name not in market]
    panel = targets.merge(
        state.drop(columns=["side_name"]),
        on="execution_decision_utc",
        how="inner",
        validate="one_to_one",
    )
    delta_signals = [
        *_signal_columns(base),
        "candidate_rows",
        "side_long_share",
        "asset_hhi",
    ]
    delta_signals = [
        name for name in dict.fromkeys(delta_signals) if name in panel
    ]
    panel, delta_features = add_causal_transition_deltas(
        panel,
        signal_columns=delta_signals,
        timestamp_col="execution_decision_utc",
        group_columns=("side_name", "evaluation_origin"),
    )
    bocpd_signals = [
        name for name in _signal_columns(base) if name in panel
    ]
    if "asset_hhi" in panel:
        bocpd_signals.append("asset_hhi")
    panel = add_causal_bocpd_features(
        panel,
        signal_columns=bocpd_signals,
        timestamp_col="execution_decision_utc",
        group_columns=("side_name", "evaluation_origin"),
    )
    bocpd_features = [
        "failure_bocpd_probability_max",
        "failure_bocpd_break_count",
        "failure_bocpd_break_intensity",
    ]
    blocks = {
        "market": market,
        "model_health": health_features,
        "transition_deltas": delta_features,
        "bocpd": bocpd_features,
        "base": base,
        "transition": [*delta_features, *bocpd_features],
        "full": [*base, *delta_features, *bocpd_features],
    }
    if len(blocks["full"]) > 40:
        raise ValueError("binary detector full feature contract exceeds 40")
    return panel, blocks


def _binary_metrics(
    frame: pd.DataFrame, target: str, probability: str
) -> dict[str, Any]:
    local = frame.loc[
        frame[target].notna() & frame[probability].notna()
    ].copy()
    if local.empty:
        return {"rows": 0, "positive_rows": 0}
    labels = pd.to_numeric(local[target], errors="raise").astype(int)
    score = pd.to_numeric(
        local[probability], errors="raise"
    ).clip(1e-6, 1 - 1e-6)
    return {
        "rows": int(len(local)),
        "positive_rows": int(labels.sum()),
        "roc_auc": (
            float(roc_auc_score(labels, score))
            if labels.nunique() == 2
            else None
        ),
        "brier": float(brier_score_loss(labels, score)),
        "log_loss": float(log_loss(labels, score, labels=[0, 1])),
    }


def _classification_report(predictions: pd.DataFrame) -> dict[str, Any]:
    timestamp = pd.to_datetime(
        predictions["execution_decision_utc"], utc=True, errors="raise"
    )
    latest_month = timestamp.dt.strftime("%Y-%m").max()
    scopes = {
        "aggregate": predictions,
        f"latest_month::{latest_month}": predictions.loc[
            timestamp.dt.strftime("%Y-%m").eq(latest_month)
        ],
    }
    report: dict[str, Any] = {}
    for scope, local in scopes.items():
        report[scope] = {
            "onset": _binary_metrics(
                local,
                "target__failure_onset_within_3h",
                "p_failure_onset_within_3h",
            ),
            "risk": _binary_metrics(
                local,
                "target__failure_active_or_within_3h",
                "p_failure_active_or_within_3h",
            ),
        }
    return report


def _economics(
    ledger: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    eligibility_flag: str = "causal_recent_side_isotonic_ev__is_oof",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    probability = predictions.loc[
        :,
        [
            "execution_decision_utc",
            "evaluation_origin",
            "p_failure_onset_within_3h",
            "p_failure_active_or_within_3h",
        ],
    ].drop_duplicates(["execution_decision_utc", "evaluation_origin"])
    strict = ledger.loc[
        ledger[eligibility_flag]
        .fillna(False)
        .astype(bool)
    ].copy()
    covered = strict.merge(
        probability,
        on=["execution_decision_utc", "evaluation_origin"],
        how="inner",
        validate="many_to_one",
    )
    mapped = pd.to_numeric(
        covered["causal_recent_side_isotonic_ev"], errors="raise"
    )
    covered["binary_onset_adjusted_score"] = (
        mapped - covered["p_failure_onset_within_3h"] * mapped.abs()
    )
    covered["binary_risk_adjusted_score"] = (
        mapped
        - covered["p_failure_active_or_within_3h"] * mapped.abs()
    )
    covered["evaluation_month"] = pd.to_datetime(
        covered["execution_decision_utc"], utc=True
    ).dt.strftime("%Y-%m")
    scopes = {"aggregate": covered}
    if len(covered):
        latest = covered["evaluation_month"].max()
        scopes[f"latest_month::{latest}"] = covered.loc[
            covered["evaluation_month"].eq(latest)
        ]
    report: dict[str, Any] = {}
    for scope, local in scopes.items():
        base = _one_global_top_decile(
            local, score_col="causal_recent_side_isotonic_ev"
        )
        row: dict[str, Any] = {
            "mapped_score": base,
            "selection_contract": (
                "one pooled global top 10 percent across timestamps and sides"
            ),
        }
        for score in (
            "binary_onset_adjusted_score",
            "binary_risk_adjusted_score",
        ):
            adjusted = _one_global_top_decile(local, score_col=score)
            row[score] = adjusted
            row[f"{score}__incremental_bps"] = (
                adjusted.get("mean_net_ev_bps", np.nan)
                - base.get("mean_net_ev_bps", np.nan)
            )
        report[scope] = row
    return covered, report


def _summary_row(
    *,
    name: str,
    stage: str,
    features: list[str],
    config: BinaryFailureDetectorConfig,
    predictions: pd.DataFrame,
    classification: dict[str, Any],
    economics: dict[str, Any],
) -> dict[str, Any]:
    latest_class = next(
        value
        for key, value in classification.items()
        if key.startswith("latest_month::")
    )
    latest_econ = next(
        value
        for key, value in economics.items()
        if key.startswith("latest_month::")
    )
    aggregate = classification["aggregate"]
    aggregate_econ = economics["aggregate"]
    auc_values = [
        aggregate["onset"].get("roc_auc"),
        aggregate["risk"].get("roc_auc"),
        latest_class["onset"].get("roc_auc"),
        latest_class["risk"].get("roc_auc"),
    ]
    finite_auc = [
        float(value) for value in auc_values if value is not None
    ]
    worst_auc = min(finite_auc) if len(finite_auc) == 4 else -np.inf
    aggregate_delta = aggregate_econ[
        "binary_risk_adjusted_score__incremental_bps"
    ]
    latest_delta = latest_econ[
        "binary_risk_adjusted_score__incremental_bps"
    ]
    return {
        "arm": name,
        "stage": stage,
        "feature_count": int(len(features)),
        "features": features,
        "config": asdict(config),
        "oof_rows": int(len(predictions)),
        "aggregate_onset_auc": aggregate["onset"].get("roc_auc"),
        "aggregate_risk_auc": aggregate["risk"].get("roc_auc"),
        "latest_onset_auc": latest_class["onset"].get("roc_auc"),
        "latest_risk_auc": latest_class["risk"].get("roc_auc"),
        "latest_onset_positives": latest_class["onset"].get(
            "positive_rows", 0
        ),
        "latest_risk_positives": latest_class["risk"].get(
            "positive_rows", 0
        ),
        "worst_auc": worst_auc,
        "aggregate_risk_delta_bps": aggregate_delta,
        "latest_risk_delta_bps": latest_delta,
        "aggregate_risk_net_bps": aggregate_econ[
            "binary_risk_adjusted_score"
        ].get("mean_net_ev_bps"),
        "latest_risk_net_bps": latest_econ[
            "binary_risk_adjusted_score"
        ].get("mean_net_ev_bps"),
        "promotion_gate_pass": bool(
            worst_auc > 0.5
            and aggregate_econ["binary_risk_adjusted_score"].get(
                "mean_net_ev_bps", -np.inf
            )
            > 0
            and latest_econ["binary_risk_adjusted_score"].get(
                "mean_net_ev_bps", -np.inf
            )
            > 0
            and latest_class["onset"].get("positive_rows", 0) >= 12
            and latest_class["risk"].get("positive_rows", 0) >= 12
        ),
    }


def _research_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = [row for row in rows if row["promotion_gate_pass"]]
    pool = promoted if promoted else rows
    return max(
        pool,
        key=lambda row: (
            float(row["worst_auc"]),
            min(
                float(row["aggregate_risk_delta_bps"]),
                float(row["latest_risk_delta_bps"]),
            ),
            float(row["aggregate_risk_net_bps"]),
            -int(row["feature_count"]),
            str(row["arm"]),
        ),
    )


def _run_arm(
    *,
    name: str,
    stage: str,
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    features: list[str],
    config: BinaryFailureDetectorConfig,
) -> tuple[dict[str, Any], pd.DataFrame, list[Any]]:
    predictions, bundles = chronological_binary_failure_oof(
        panel, feature_columns=features, config=config
    )
    classification = _classification_report(predictions)
    _, economics = _economics(ledger, predictions)
    row = _summary_row(
        name=name,
        stage=stage,
        features=features,
        config=config,
        predictions=predictions,
        classification=classification,
        economics=economics,
    )
    return row, predictions, bundles


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-pipeline", type=Path, default=HISTORICAL_PIPELINE
    )
    parser.add_argument(
        "--current-pipeline", type=Path, default=CURRENT_PIPELINE
    )
    parser.add_argument(
        "--historical-ledger", type=Path, default=HISTORICAL_LEDGER
    )
    parser.add_argument(
        "--current-ledger", type=Path, default=CURRENT_LEDGER
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    historical_health_path = (
        Path(args.historical_pipeline) / "decision_health_6h.parquet"
    )
    historical_state_path = (
        Path(args.historical_pipeline) / "hourly_observable_state.parquet"
    )
    current_health_path = (
        Path(args.current_pipeline) / "decision_health_6h.parquet"
    )
    current_state_path = (
        Path(args.current_pipeline) / "hourly_observable_state.parquet"
    )
    historical_health = pd.read_parquet(historical_health_path)
    historical_state = pd.read_parquet(historical_state_path)
    historical_ledger = pd.read_parquet(args.historical_ledger)
    panel, blocks = _prepare_panel(
        health=historical_health, hourly_state=historical_state
    )
    first_eval = (
        pd.to_datetime(
            panel["execution_decision_utc"], utc=True, errors="raise"
        ).min()
        + pd.Timedelta(days=90)
    )
    base_config = BinaryFailureDetectorConfig(
        first_eval_time=first_eval.isoformat()
    )
    hpo_configs = {
        "unweighted_d5": replace(
            base_config,
            auto_class_weights=None,
            depth=5,
            max_iter=120,
            learning_rate=0.05,
            l2_regularization=5.0,
        ),
        "balanced_d4": replace(
            base_config,
            depth=4,
            max_iter=120,
            learning_rate=0.05,
            l2_regularization=5.0,
        ),
        "balanced_d5": base_config,
        "balanced_d6_slow": replace(
            base_config,
            depth=6,
            max_iter=160,
            learning_rate=0.04,
            l2_regularization=8.0,
        ),
    }
    summaries: list[dict[str, Any]] = []
    arm_outputs: dict[str, tuple[pd.DataFrame, list[Any]]] = {}
    for name, config in hpo_configs.items():
        row, predictions, bundles = _run_arm(
            name=name,
            stage="hpo_full",
            panel=panel,
            ledger=historical_ledger,
            features=blocks["full"],
            config=config,
        )
        summaries.append(row)
        arm_outputs[name] = (predictions, bundles)
    hpo_winner = _research_winner(summaries)
    winner_config = BinaryFailureDetectorConfig(**hpo_winner["config"])
    feature_arms = {
        "market_only": blocks["market"],
        "model_health_only": blocks["model_health"],
        "market_model_health": blocks["base"],
        "market_plus_transition": [
            *blocks["market"],
            *blocks["transition"],
        ],
        "model_health_plus_transition": [
            *blocks["model_health"],
            *blocks["transition"],
        ],
        "full_without_bocpd": [
            *blocks["base"],
            *blocks["transition_deltas"],
        ],
        "full_without_deltas": [
            *blocks["base"],
            *blocks["bocpd"],
        ],
        "full": blocks["full"],
    }
    feature_rows: list[dict[str, Any]] = []
    for name, features in feature_arms.items():
        arm_name = f"feature__{name}"
        if (
            name == "full"
            and hpo_winner["arm"] in arm_outputs
        ):
            predictions, bundles = arm_outputs[hpo_winner["arm"]]
            classification = _classification_report(predictions)
            _, economics = _economics(historical_ledger, predictions)
            row = _summary_row(
                name=arm_name,
                stage="feature_block",
                features=features,
                config=winner_config,
                predictions=predictions,
                classification=classification,
                economics=economics,
            )
        else:
            row, predictions, bundles = _run_arm(
                name=arm_name,
                stage="feature_block",
                panel=panel,
                ledger=historical_ledger,
                features=features,
                config=winner_config,
            )
        summaries.append(row)
        feature_rows.append(row)
        arm_outputs[arm_name] = (predictions, bundles)
    feature_winner = _research_winner(feature_rows)
    winner_features = list(feature_winner["features"])
    winner_predictions, _ = arm_outputs[feature_winner["arm"]]
    winner_classification = _classification_report(winner_predictions)
    historical_overlay, winner_economics = _economics(
        historical_ledger, winner_predictions
    )
    final_boundary = pd.to_datetime(
        panel["binary_failure_label_available_at"],
        utc=True,
        errors="coerce",
    ).max() + pd.Timedelta("1ns")
    frozen = fit_binary_failure_detector(
        panel,
        feature_columns=winner_features,
        train_end_exclusive=final_boundary,
        config=winner_config,
    )
    bundle_path = output / "historical_binary_failure_detector_frozen.joblib"
    joblib.dump(frozen, bundle_path)

    current_health = pd.read_parquet(current_health_path)
    current_state = pd.read_parquet(current_state_path)
    current_ledger = pd.read_parquet(args.current_ledger)
    current_panel, current_blocks = _prepare_panel(
        health=current_health,
        hourly_state=current_state,
        frozen_base_features=blocks["base"],
    )
    missing = sorted(set(winner_features).difference(current_panel))
    if missing:
        raise ValueError(
            "current transfer missing winner features: " + ", ".join(missing)
        )
    current_predictions = current_panel.loc[
        :,
        ["execution_decision_utc", "side_name", "evaluation_origin"],
    ].copy()
    current_scored = frozen.score(current_panel)
    for name in current_scored:
        current_predictions[name] = current_scored[name].to_numpy()
    for name in (
        "target__failure_onset_within_3h",
        "target__failure_active_or_within_3h",
        "binary_failure_label_available_at",
    ):
        current_predictions[name] = current_panel[name].to_numpy()
    current_classification = _classification_report(current_predictions)
    current_overlay, current_economics = _economics(
        current_ledger, current_predictions
    )

    pd.DataFrame(summaries).drop(columns=["features", "config"]).to_csv(
        output / "ablation_summary.csv", index=False
    )
    winner_predictions.to_parquet(
        output / "historical_winner_oof.parquet", index=False
    )
    historical_overlay.to_parquet(
        output / "historical_winner_candidate_overlay.parquet", index=False
    )
    current_predictions.to_parquet(
        output / "current_transfer_predictions.parquet", index=False
    )
    current_overlay.to_parquet(
        output / "current_transfer_candidate_overlay.parquet", index=False
    )
    _write_json(
        output / "ablation_manifest.json",
        {
            "schema": "failure_first_binary_ablation_v1",
            "status": (
                "RESEARCH_PROMOTION_GATE_PASS"
                if feature_winner["promotion_gate_pass"]
                else "RESEARCH_COMPLETE_PROMOTION_REJECT"
            ),
            "historical_hpo_winner": hpo_winner,
            "historical_feature_winner": feature_winner,
            "all_arms": summaries,
            "historical_classification": winner_classification,
            "historical_economics": winner_economics,
            "current_transfer_classification": current_classification,
            "current_transfer_economics": current_economics,
            "promotion_allowed": False,
            "promotion_reason": (
                "Historical-comparator HPO and cross-model current transfer "
                "are diagnostics; current-model chronological OOF and a sealed "
                "later forward recurrence are absent."
            ),
            "feature_blocks": blocks,
            "current_feature_blocks": current_blocks,
            "frozen_bundle_sha256": _sha256(bundle_path),
            "source_hashes": {
                str(path): _sha256(path)
                for path in (
                    historical_health_path,
                    historical_state_path,
                    Path(args.historical_ledger),
                    current_health_path,
                    current_state_path,
                    Path(args.current_ledger),
                )
            },
        },
    )
    return {
        "status": (
            "RESEARCH_PROMOTION_GATE_PASS"
            if feature_winner["promotion_gate_pass"]
            else "RESEARCH_COMPLETE_PROMOTION_REJECT"
        ),
        "historical_hpo_winner": hpo_winner["arm"],
        "historical_feature_winner": feature_winner["arm"],
        "historical_oof_rows": int(len(winner_predictions)),
        "current_transfer_rows": int(len(current_predictions)),
        "promotion_allowed": False,
        "output_dir": str(output),
    }


def main() -> None:
    result = run(_parser().parse_args())
    print(json.dumps(_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
