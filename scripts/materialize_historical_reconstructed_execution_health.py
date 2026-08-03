#!/usr/bin/env python3
"""Materialize historical current-architecture execution-health reconstruction.

This is not the frozen 2026 current model backcast.  It joins the completed
March inner-OOF and untouched April outer execution-EV reconstruction,
recomputes the same causal 21-day score-to-EV mapping, and materializes the
29-field health schema where exact semantics exist.  Missing current-only
fields remain missing and are excluded from the common cross-era catalog.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_transition_current_model_health import (  # noqa: E402
    CURRENT_MODEL_HEALTH_COLUMNS,
    build_hourly_current_model_health,
)
from materialize_historical_exact_model_health import (  # noqa: E402
    _safe,
    _sha256,
    _write_json,
)
from run_execution_ev_recent_mapping_ablation import causal_mappings  # noqa: E402


DEFAULT_GATE = ROOT / (
    "data_perp/artifacts/historical_execution_ev_add_drop_gate_20260729_v6/"
    "base_residual"
)
DEFAULT_RESIDUAL = ROOT / (
    "data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1/"
    "oof_predictions.parquet"
)
DEFAULT_CONTEXT = ROOT / (
    "data_perp/artifacts/febapr2025_execution_ev_current_spread_two_layer_oof_"
    "20260727_v2/two_layer_direct_ev_strict_oof.parquet"
)
DEFAULT_SIX_LONG = ROOT / (
    "data_perp/artifacts/febapr2025_historical_six_class_catboost_"
    "20260729_v3/long/oof.parquet"
)
DEFAULT_SIX_SHORT = ROOT / (
    "data_perp/artifacts/febapr2025_historical_six_class_catboost_"
    "20260729_v3/short/oof.parquet"
)

IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
SIX_PROBABILITIES = (
    "prob_immediate_adverse_path",
    "prob_fast_realization_winner",
    "prob_late_breakout",
    "prob_slow_grinder",
    "prob_mfe_reversal_or_timeout",
    "prob_dead_timeout",
)
NONCOMPARABLE_HEALTH_FIELDS = {
    # Historical reconstruction has no exact alpha-uncertainty output.
    "health__alpha_uncertainty_mean",
    # Historical CatBoost has six classes versus seven in the current stack;
    # raw entropy scales are therefore not directly comparable.
    "health__catboost_entropy_mean",
}
COMMON_HEALTH_COLUMNS = tuple(
    column
    for column in CURRENT_MODEL_HEALTH_COLUMNS
    if column not in NONCOMPARABLE_HEALTH_FIELDS
)


def _normalise_identity(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} lacks {missing}")
    result = frame.copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["__symbol__"] = result["__symbol__"].astype(str)
    if result.duplicated(list(IDENTITY)).any():
        raise ValueError(f"{source} contains duplicate identities")
    return result


def _catboost_entropy(six: pd.DataFrame) -> pd.DataFrame:
    local = _normalise_identity(six, source="historical six-class OOF")
    missing = sorted(set(SIX_PROBABILITIES).difference(local.columns))
    if missing:
        raise ValueError(f"historical six-class OOF lacks {missing}")
    probability = local.loc[:, SIX_PROBABILITIES].apply(
        pd.to_numeric, errors="coerce"
    )
    if probability.isna().any().any():
        raise ValueError("historical six-class probabilities contain missing values")
    values = probability.to_numpy(float)
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("historical six-class probabilities do not sum to one")
    entropy = -(values * np.log(np.clip(values, 1e-12, 1.0))).sum(axis=1)
    return local.loc[:, IDENTITY].assign(catboost_entropy=entropy)


def build_historical_reconstructed_sources(
    march: pd.DataFrame,
    april: pd.DataFrame,
    residual: pd.DataFrame,
    context: pd.DataFrame,
    six: pd.DataFrame,
    *,
    window_days: int,
    minimum_reference_rows: int,
    side_support_target: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    march = _normalise_identity(march, source="March execution OOF")
    april = _normalise_identity(april, source="April execution outer")
    required_march = {
        "score",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        "execution_net_ev_12h",
    }
    required_april = {
        "raw_score",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        "execution_net_ev_12h",
    }
    if missing := sorted(required_march.difference(march.columns)):
        raise ValueError(f"March execution OOF lacks {missing}")
    if missing := sorted(required_april.difference(april.columns)):
        raise ValueError(f"April execution outer lacks {missing}")
    march_ledger = march.loc[
        :,
        [
            *IDENTITY,
            "score",
            "execution_label_end_utc",
            "execution_gross_ev_12h",
            "execution_net_ev_12h",
        ],
    ].rename(columns={"score": "raw_execution_ev_score"})
    march_ledger["evaluation_origin"] = "march_inner_oof"
    april_ledger = april.loc[
        :,
        [
            *IDENTITY,
            "raw_score",
            "execution_label_end_utc",
            "execution_gross_ev_12h",
            "execution_net_ev_12h",
        ],
    ].rename(columns={"raw_score": "raw_execution_ev_score"})
    april_ledger["evaluation_origin"] = "april_outer_oos"
    ledger = pd.concat([march_ledger, april_ledger], ignore_index=True)
    if ledger.duplicated(list(IDENTITY)).any():
        raise ValueError("March and April execution ledgers overlap")
    ledger["execution_decision_utc"] = (
        ledger["__ts__"] + pd.Timedelta(hours=1)
    )
    ledger["execution_label_end_utc"] = pd.to_datetime(
        ledger["execution_label_end_utc"], utc=True, errors="raise"
    )
    ledger = ledger.sort_values(
        ["execution_decision_utc", "__symbol__", "side_name", "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)
    mapped, audit = causal_mappings(
        ledger,
        score_col="raw_execution_ev_score",
        window_days=int(window_days),
        min_reference_rows=int(minimum_reference_rows),
        side_support_target=float(side_support_target),
    )
    available = mapped["causal_recent_side_isotonic_ev"].notna()
    for column in (
        "causal_recent_percentile",
        "causal_recent_robust_z",
        "causal_recent_isotonic_ev",
        "causal_recent_side_isotonic_ev",
    ):
        mapped[f"{column}__is_oof"] = mapped[column].notna()
        mapped[f"{column}__is_forward_oos"] = False
    mapped["catboost__residual__without_hpo__all_features"] = mapped[
        "raw_execution_ev_score"
    ]

    residual = _normalise_identity(residual, source="historical residual OOF")
    context = _normalise_identity(context, source="historical context")
    entropy = _catboost_entropy(six)
    rich = mapped.loc[:, IDENTITY].merge(
        residual.loc[
            :,
            [*IDENTITY, "base_oof_score", "residual_is_oof"],
        ],
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    rich = rich.merge(
        context.loc[:, [*IDENTITY, "base_margin_to_cutoff_z"]],
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    rich = rich.merge(
        entropy,
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    if rich[
        ["base_oof_score", "base_margin_to_cutoff_z", "catboost_entropy"]
    ].isna().any().any():
        raise ValueError("historical reconstructed health lacks exact rich context")
    if not rich["residual_is_oof"].fillna(False).astype(bool).all():
        raise ValueError("historical reconstructed health joins non-OOF residuals")
    rich["execution_decision_utc"] = rich["__ts__"] + pd.Timedelta(hours=1)
    rich["alpha_prediction_uncertainty"] = np.nan
    rich = rich.loc[
        :,
        [
            "candidate_id",
            "__ts__",
            "execution_decision_utc",
            "base_oof_score",
            "base_margin_to_cutoff_z",
            "catboost_entropy",
            "alpha_prediction_uncertainty",
        ],
    ]
    return mapped, rich, audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    gate = Path(args.gate_dir)
    residual_path = Path(args.residual)
    context_path = Path(args.context)
    six_paths = [Path(args.six_long), Path(args.six_short)]
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    six = pd.concat(
        [pd.read_parquet(path) for path in six_paths], ignore_index=True
    )
    ledger, rich, audit = build_historical_reconstructed_sources(
        pd.read_parquet(gate / "march_inner_oof_scores.parquet"),
        pd.read_parquet(gate / "april_outer_predictions.parquet"),
        pd.read_parquet(residual_path),
        pd.read_parquet(context_path),
        six,
        window_days=int(args.window_days),
        minimum_reference_rows=int(args.minimum_reference_rows),
        side_support_target=float(args.side_support_target),
    )
    health, health_report = build_hourly_current_model_health(ledger, rich)
    output.mkdir(parents=True, exist_ok=False)
    ledger_path = output / "reconstructed_execution_ledger.parquet"
    rich_path = output / "reconstructed_rich_handoff.parquet"
    health_path = output / "hourly_model_health.parquet"
    catalog_path = output / "common_health_catalog.csv"
    audit_path = output / "mapping_audit.csv"
    ledger.to_parquet(ledger_path, index=False, compression="zstd")
    rich.to_parquet(rich_path, index=False, compression="zstd")
    health.to_parquet(health_path, index=False, compression="zstd")
    pd.DataFrame(
        {
            "feature": CURRENT_MODEL_HEALTH_COLUMNS,
            "cross_era_common": [
                feature in COMMON_HEALTH_COLUMNS
                for feature in CURRENT_MODEL_HEALTH_COLUMNS
            ],
        }
    ).to_csv(catalog_path, index=False)
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    mapped = ledger["causal_recent_side_isotonic_ev"].notna()
    manifest = {
        "schema": "historical_reconstructed_execution_health_v1",
        "status": "RESEARCH_ONLY_HISTORICAL_CURRENT_ARCHITECTURE_RECONSTRUCTION",
        "exact_current_lineage": False,
        "lineage_disclosure": (
            "March inner-OOF and April untouched outer execution-EV "
            "reconstruction; not a frozen 2026 model backcast"
        ),
        "selection_readiness": (
            "one pooled global top-k is allowed only after the recomputed "
            "causal recent side-EV mapping"
        ),
        "candidate_rows": int(len(ledger)),
        "mapped_candidate_rows": int(mapped.sum()),
        "mapped_candidate_fraction": float(mapped.mean()),
        "health_rows": int(len(health)),
        "start_utc": health["source_utc"].min(),
        "end_utc": health["source_utc"].max(),
        "health_feature_count": len(CURRENT_MODEL_HEALTH_COLUMNS),
        "cross_era_common_feature_count": len(COMMON_HEALTH_COLUMNS),
        "excluded_cross_era_fields": sorted(NONCOMPARABLE_HEALTH_FIELDS),
        "health_report": health_report,
        "mapping_contract": {
            "window_days": int(args.window_days),
            "minimum_reference_rows": int(args.minimum_reference_rows),
            "side_support_target": float(args.side_support_target),
            "reference_resolution": "strictly before each UTC-day snapshot",
        },
        "sources": {
            "gate_manifest": {
                "path": str(gate.parent / "manifest.json"),
                "sha256": _sha256(gate.parent / "manifest.json"),
            },
            "residual": {
                "path": str(residual_path),
                "sha256": _sha256(residual_path),
            },
            "context": {
                "path": str(context_path),
                "sha256": _sha256(context_path),
            },
            **{
                f"six_{index}": {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for index, path in enumerate(six_paths)
            },
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in {
                "ledger": ledger_path,
                "rich": rich_path,
                "health": health_path,
                "catalog": catalog_path,
                "mapping_audit": audit_path,
            }.items()
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--residual", type=Path, default=DEFAULT_RESIDUAL)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--six-long", type=Path, default=DEFAULT_SIX_LONG)
    parser.add_argument("--six-short", type=Path, default=DEFAULT_SIX_SHORT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-days", type=int, default=21)
    parser.add_argument("--minimum-reference-rows", type=int, default=500)
    parser.add_argument("--side-support-target", type=float, default=500.0)
    return parser


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
