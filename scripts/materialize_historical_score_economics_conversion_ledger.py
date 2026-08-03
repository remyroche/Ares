#!/usr/bin/env python3
"""Materialize source-separated score-to-economics conversion ledgers.

This runner standardizes frozen/OOF score streams and realized 12-hour
execution outcomes without pooling evidence tiers.  Each output parquet is a
separate source family with one unique candidate identity and explicit path,
cost, OOF and promotion contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
ECONOMICS = (
    "execution_gross_ev_12h",
    "execution_net_ev_12h",
    "execution_cost_return",
    "execution_exit_reason",
    "execution_mfe_return_12h",
    "execution_mae_return_12h",
    "execution_soft_positive_12h",
)
STANDARD_REQUIRED = (
    *IDENTITY,
    "execution_decision_utc",
    "execution_label_end_utc",
    "candidate_month",
    *ECONOMICS,
    "execution_exit_minute",
    "execution_exit_class",
)
CANONICAL_PROMOTION_FAMILIES = {
    "canonical_base_exact1m_current_spread_cf",
    "canonical_residual_exact1m_current_spread_cf",
}

DEFAULT_CANONICAL_BASE = ROOT / (
    "data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1/"
    "oof_predictions.parquet"
)
DEFAULT_CANONICAL_RESIDUAL = ROOT / (
    "data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1/"
    "oof_predictions.parquet"
)
DEFAULT_CANONICAL_POPULATION = ROOT / (
    "data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v2/"
    "population.parquet"
)
DEFAULT_RECONSTRUCTED_EXACT_SCORES = ROOT / (
    "data_perp/artifacts/janapr2025_execution_ev_exact1m_two_layer_oof_20260727_v1/"
    "two_layer_direct_ev_strict_oof.parquet"
)
DEFAULT_RECONSTRUCTED_EXACT_LABELS = ROOT / (
    "data_perp/artifacts/janapr2025_execution_ev_exact1m_two_layer_oof_20260727_v1/"
    "exact_1m_execution_ev_12h_labels.parquet"
)
DEFAULT_LATE2024_SCORES = ROOT / (
    "data_perp/artifacts/late2024_execution_ev_hourly_comparator_20260727_v2/"
    "hourly_two_layer_execution_ev_strict_oof.parquet"
)
DEFAULT_LATE2024_LABELS = ROOT / (
    "data_perp/artifacts/late2024_execution_ev_hourly_comparator_20260727_v2/"
    "hourly_execution_ev_12h_labels.parquet"
)
DEFAULT_HISTORICAL_HOURLY = ROOT / (
    "data_perp/artifacts/historical_comparable_execution_ev_12h_oof_20260726_v4/"
    "historical_direct_ev_strict_oof.parquet"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/historical_score_economics_conversion_ledgers_20260729_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _normalise_identity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="raise")
    out["__symbol__"] = out["__symbol__"].astype(str).str.upper()
    out["side_name"] = out["side_name"].astype(str).str.lower()
    out["candidate_id"] = out["candidate_id"].astype(str)
    return out


def _verified_join(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """One-to-one join scores to labels and verify duplicated score outcomes."""

    score_frame = _normalise_identity(scores)
    label_frame = _normalise_identity(labels)
    if score_frame.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("score source contains duplicate identities")
    if label_frame.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("label source contains duplicate identities")
    overlap = sorted(
        set(score_frame.columns)
        .intersection(label_frame.columns)
        .difference(IDENTITY)
    )
    renamed = {column: f"{column}__score_source" for column in overlap}
    joined = score_frame.rename(columns=renamed).merge(
        label_frame,
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = int(joined["_merge"].ne("both").sum())
        raise ValueError(f"label source is missing {missing} score identities")
    joined = joined.drop(columns="_merge")
    for column in overlap:
        score_column = f"{column}__score_source"
        if column.endswith("_utc"):
            score_values = pd.to_datetime(joined[score_column], utc=True, errors="raise")
            label_values = pd.to_datetime(joined[column], utc=True, errors="raise")
            if not score_values.equals(label_values):
                raise ValueError(f"score/label timestamp mismatch for {column}")
        elif (
            pd.api.types.is_numeric_dtype(joined[score_column])
            and pd.api.types.is_numeric_dtype(joined[column])
        ):
            score_values = pd.to_numeric(joined[score_column], errors="raise")
            label_values = pd.to_numeric(joined[column], errors="raise")
            if not np.allclose(
                score_values.to_numpy(float),
                label_values.to_numpy(float),
                rtol=0.0,
                atol=1e-7,
            ):
                raise ValueError(f"score/label economics mismatch for {column}")
        else:
            score_values = joined[score_column].astype(str)
            label_values = joined[column].astype(str)
            if not score_values.equals(label_values):
                raise ValueError(f"score/label field mismatch for {column}")
        joined = joined.drop(columns=score_column)
    return joined


def _standardise(
    frame: pd.DataFrame,
    *,
    source_family: str,
    evidence_tier: str,
    path_frequency: str,
    cost_contract: str,
    promotion_eligible: bool,
    exact_policy_parity: bool,
    score_columns: Mapping[str, str],
    exit_duration_column: str = "execution_exit_minute",
    exit_duration_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Return one validated standard ledger for a declared source family."""

    work = _normalise_identity(frame)
    for column in ("execution_decision_utc", "execution_label_end_utc"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    if "candidate_month" not in work:
        work["candidate_month"] = work["execution_decision_utc"].dt.strftime("%Y-%m")
    work["candidate_month"] = work["candidate_month"].astype(str)
    work["execution_exit_minute"] = (
        pd.to_numeric(work[exit_duration_column], errors="raise")
        * float(exit_duration_multiplier)
    )
    work["execution_exit_class"] = (
        work["execution_exit_reason"]
        .astype(str)
        .replace({"full_sl": "full_stop"})
    )
    keep = [*STANDARD_REQUIRED]
    optional = [
        "__first_touch_target_soft__",
        "__first_touch_capture_net__",
        "base_oof_fold_start_utc",
        "base_oof_train_cutoff_utc",
        "fold_id",
        "fold_validation_start_utc",
        "fold_validation_end_utc",
        "base_label_resolution_utc",
        "effective_label_resolution_utc",
        "direct_oof_fold_start_utc",
        "direct_oof_train_cutoff_utc",
        "execution_ev_oof_fold_start_utc",
        "execution_ev_oof_train_cutoff_utc",
        "oof_train_cutoff_utc",
        "oof_fold_month",
        "residual_fold",
    ]
    keep.extend(column for column in optional if column in work)
    for standard_name, source_name in score_columns.items():
        if source_name not in work:
            raise ValueError(
                f"{source_family} is missing score source {source_name!r}"
            )
        work[standard_name] = pd.to_numeric(work[source_name], errors="raise")
        keep.append(standard_name)
    out = work.loc[:, list(dict.fromkeys(keep))].copy()
    out["source_family"] = str(source_family)
    out["evidence_tier"] = str(evidence_tier)
    out["path_frequency"] = str(path_frequency)
    out["label_horizon_hours"] = np.int16(12)
    out["cost_contract"] = str(cost_contract)
    out["promotion_eligible"] = bool(promotion_eligible)
    out["diagnostic_only"] = not bool(promotion_eligible)
    out["exact_policy_parity"] = bool(exact_policy_parity)
    out["historical_observed_spread"] = False
    out["opportunity_gross_above_cost_0bps"] = (
        pd.to_numeric(out["execution_gross_ev_12h"], errors="raise")
        > pd.to_numeric(out["execution_cost_return"], errors="raise")
    )
    out["opportunity_gross_above_cost_25bps"] = (
        pd.to_numeric(out["execution_gross_ev_12h"], errors="raise")
        > pd.to_numeric(out["execution_cost_return"], errors="raise") + 0.0025
    )
    out["positive_net_12h"] = (
        pd.to_numeric(out["execution_net_ev_12h"], errors="raise") > 0.0
    )
    exit_class = out["execution_exit_class"].astype(str)
    out["exit_is_trailing"] = exit_class.eq("trailing")
    out["exit_is_timeout"] = exit_class.eq("timeout")
    out["exit_is_full_stop"] = exit_class.eq("full_stop")
    out["exit_is_adverse"] = exit_class.isin(("full_stop", "adverse_exit"))
    out["exit_is_adverse_exit"] = exit_class.eq("adverse_exit")
    return validate_ledger(out)


def validate_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail closed on identity, causal horizon, economics and source mixing."""

    required = {
        *STANDARD_REQUIRED,
        "source_family",
        "evidence_tier",
        "path_frequency",
        "label_horizon_hours",
        "cost_contract",
        "promotion_eligible",
        "diagnostic_only",
        "exact_policy_parity",
        "historical_observed_spread",
        "execution_exit_class",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"conversion ledger missing fields: {missing}")
    if frame.empty or frame.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("conversion ledger identities must be nonempty and unique")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("candidate_id must be globally unique within a ledger")
    for column in ("source_family", "evidence_tier", "path_frequency", "cost_contract"):
        if frame[column].nunique(dropna=False) != 1:
            raise ValueError(f"conversion ledger mixes {column}")
    signal = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    decision = pd.to_datetime(
        frame["execution_decision_utc"], utc=True, errors="raise"
    )
    resolution = pd.to_datetime(
        frame["execution_label_end_utc"], utc=True, errors="raise"
    )
    if not decision.equals(signal + pd.Timedelta(hours=1)):
        raise ValueError("decision timestamp must equal signal timestamp + one hour")
    if not resolution.equals(decision + pd.Timedelta(hours=12)):
        raise ValueError("label resolution must equal decision timestamp + twelve hours")
    if not frame["candidate_month"].astype(str).equals(signal.dt.strftime("%Y-%m")):
        raise ValueError("candidate_month does not match the UTC signal month")
    if not pd.to_numeric(
        frame["label_horizon_hours"], errors="raise"
    ).eq(12).all():
        raise ValueError("label_horizon_hours must be 12 on every row")
    if not frame["diagnostic_only"].astype(bool).eq(
        ~frame["promotion_eligible"].astype(bool)
    ).all():
        raise ValueError("diagnostic_only must invert promotion_eligible")
    if frame["historical_observed_spread"].astype(bool).any():
        raise ValueError("historical observed spread must not be claimed")
    numeric = frame.loc[
        :,
        [
            "execution_gross_ev_12h",
            "execution_net_ev_12h",
            "execution_cost_return",
            "execution_exit_minute",
            "execution_mfe_return_12h",
            "execution_mae_return_12h",
            "execution_soft_positive_12h",
        ],
    ].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("conversion economics must be finite")
    if not np.allclose(
        numeric["execution_gross_ev_12h"].to_numpy(float)
        - numeric["execution_cost_return"].to_numpy(float),
        numeric["execution_net_ev_12h"].to_numpy(float),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("gross minus cost does not reconcile to net")
    soft = numeric["execution_soft_positive_12h"].to_numpy(float)
    if ((soft < 0.0) | (soft > 1.0)).any():
        raise ValueError("execution_soft_positive_12h must be in [0,1]")
    score_columns = [column for column in frame if column.startswith("score_")]
    if not score_columns:
        raise ValueError("conversion ledger has no standardized score")
    if not np.isfinite(
        frame.loc[:, score_columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    ).all():
        raise ValueError("standardized scores must be finite")
    if not frame["side_name"].isin(("long", "short")).all():
        raise ValueError("conversion ledger contains an unknown side")
    exit_class = frame["execution_exit_class"].astype(str)
    allowed_exit_classes = {"trailing", "timeout", "full_stop", "adverse_exit"}
    if not exit_class.isin(allowed_exit_classes).all():
        unknown = sorted(set(exit_class).difference(allowed_exit_classes))
        raise ValueError(f"conversion ledger contains unknown exit classes: {unknown}")
    exit_flags = frame.loc[
        :,
        [
            "exit_is_trailing",
            "exit_is_timeout",
            "exit_is_full_stop",
            "exit_is_adverse_exit",
        ],
    ].astype(bool)
    if not exit_flags.sum(axis=1).eq(1).all():
        raise ValueError("exactly one canonical exit-class flag is required")
    if not frame["exit_is_adverse"].astype(bool).equals(
        frame["exit_is_full_stop"].astype(bool)
        | frame["exit_is_adverse_exit"].astype(bool)
    ):
        raise ValueError("exit_is_adverse does not match stop/adverse-exit flags")
    promotion = frame["promotion_eligible"].astype(bool)
    if promotion.any():
        family = str(frame["source_family"].iloc[0])
        if family not in CANONICAL_PROMOTION_FAMILIES:
            raise ValueError("noncanonical source family is promotion eligible")
        if not frame["path_frequency"].eq("exact_1m").all():
            raise ValueError("promotion-eligible rows require exact one-minute paths")
        if not frame["exact_policy_parity"].astype(bool).all():
            raise ValueError("promotion-eligible rows require exact-policy parity")
    return frame.sort_values(
        ["execution_decision_utc", "candidate_id"], kind="stable"
    ).reset_index(drop=True)


def _canonical_base(args: argparse.Namespace) -> pd.DataFrame:
    scores = pd.read_parquet(args.canonical_base)
    labels = pd.read_parquet(args.canonical_population)
    joined = _verified_join(scores, labels)
    return _standardise(
        joined,
        source_family="canonical_base_exact1m_current_spread_cf",
        evidence_tier="canonical_frozen_base_exact_1m",
        path_frequency="exact_1m",
        cost_contract="current_asset_spread_counterfactual_plus_frozen_policy_cost",
        promotion_eligible=True,
        exact_policy_parity=True,
        score_columns={"score_base_alpha": "base_oof_score"},
    )


def _canonical_residual(args: argparse.Namespace) -> pd.DataFrame:
    scores = pd.read_parquet(args.canonical_residual)
    scores = scores.loc[scores["residual_is_oof"].astype(bool)].copy()
    labels = pd.read_parquet(args.canonical_population)
    joined = _verified_join(scores, labels)
    return _standardise(
        joined,
        source_family="canonical_residual_exact1m_current_spread_cf",
        evidence_tier="canonical_frozen_residual_exact_1m",
        path_frequency="exact_1m",
        cost_contract="current_asset_spread_counterfactual_plus_frozen_policy_cost",
        promotion_eligible=True,
        exact_policy_parity=True,
        score_columns={
            "score_base_alpha": "base_oof_score",
            "score_base_expected_ev": "base_expected_ev",
            "score_residual_expected_ev": "residual_expected_ev",
            "score_residual_delta_ev": "residual_delta_ev",
        },
    )


def _reconstructed_exact(args: argparse.Namespace) -> pd.DataFrame:
    scores = pd.read_parquet(args.reconstructed_exact_scores)
    labels = pd.read_parquet(args.reconstructed_exact_labels)
    joined = _verified_join(scores, labels)
    return _standardise(
        joined,
        source_family="reconstructed_exact1m_janapr_fee_only",
        evidence_tier="diagnostic_reconstructed_two_layer_exact_1m",
        path_frequency="exact_1m",
        cost_contract="current_side_parent_round_trip_fee_once_no_historical_spread",
        promotion_eligible=False,
        exact_policy_parity=False,
        score_columns={
            "score_base_alpha": "historical_base_soft_oof",
            "score_direct_execution_ev": "historical_direct_ev_oof",
        },
    )


def _late2024_hourly(args: argparse.Namespace) -> pd.DataFrame:
    scores = pd.read_parquet(args.late2024_scores)
    labels = pd.read_parquet(args.late2024_labels)
    joined = _verified_join(scores, labels)
    return _standardise(
        joined,
        source_family="reconstructed_hourly_late2024_fee_only",
        evidence_tier="diagnostic_reconstructed_two_layer_hourly",
        path_frequency="hourly",
        cost_contract="current_side_parent_round_trip_fee_once_no_historical_spread",
        promotion_eligible=False,
        exact_policy_parity=False,
        score_columns={
            "score_base_alpha": "hourly_base_soft_oof",
            "score_direct_execution_ev": "hourly_execution_ev_oof",
        },
        exit_duration_column="execution_exit_hour",
        exit_duration_multiplier=60.0,
    )


def _historical_hourly(args: argparse.Namespace) -> pd.DataFrame:
    source = pd.read_parquet(args.historical_hourly)
    net = pd.to_numeric(source["execution_net_ev_12h"], errors="raise").to_numpy(
        dtype=float
    )
    source["execution_soft_positive_12h"] = 1.0 / (
        1.0 + np.exp(-np.clip(net / 0.01, -60.0, 60.0))
    )
    return _standardise(
        source,
        source_family="historical_hourly_old55_may25_apr26",
        evidence_tier="diagnostic_old55_hourly_recurrence",
        path_frequency="hourly",
        cost_contract="fixed_1pct_round_trip_fee_once_no_historical_spread",
        promotion_eligible=False,
        exact_policy_parity=False,
        score_columns={
            "score_base_alpha": "score_base",
            "score_direct_execution_ev": "historical_direct_ev_oof",
        },
        exit_duration_column="execution_exit_hour",
        exit_duration_multiplier=60.0,
    )


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    inputs = {
        "canonical_base": args.canonical_base,
        "canonical_residual": args.canonical_residual,
        "canonical_population": args.canonical_population,
        "reconstructed_exact_scores": args.reconstructed_exact_scores,
        "reconstructed_exact_labels": args.reconstructed_exact_labels,
        "late2024_scores": args.late2024_scores,
        "late2024_labels": args.late2024_labels,
        "historical_hourly": args.historical_hourly,
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} source does not exist: {path}")
    builders = (
        _canonical_base,
        _canonical_residual,
        _reconstructed_exact,
        _late2024_hourly,
        _historical_hourly,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ledgers_dir = args.output_dir / "ledgers"
    ledgers_dir.mkdir()
    outputs: dict[str, Path] = {}
    catalog: list[dict[str, Any]] = []
    for builder in builders:
        ledger = builder(args)
        family = str(ledger["source_family"].iloc[0])
        path = ledgers_dir / f"{family}.parquet"
        ledger.to_parquet(path, index=False, compression="zstd")
        outputs[family] = path
        score_columns = [column for column in ledger if column.startswith("score_")]
        months = (
            ledger.groupby(["candidate_month", "side_name"], sort=True)
            .size()
            .rename("rows")
            .reset_index()
            .to_dict(orient="records")
        )
        catalog.append(
            {
                "source_family": family,
                "path": str(path),
                "sha256": _sha256(path),
                "rows": int(len(ledger)),
                "minimum_decision_utc": ledger["execution_decision_utc"].min(),
                "maximum_decision_utc": ledger["execution_decision_utc"].max(),
                "score_columns": score_columns,
                "evidence_tier": ledger["evidence_tier"].iloc[0],
                "path_frequency": ledger["path_frequency"].iloc[0],
                "cost_contract": ledger["cost_contract"].iloc[0],
                "promotion_eligible": bool(ledger["promotion_eligible"].iloc[0]),
                "exact_policy_parity": bool(ledger["exact_policy_parity"].iloc[0]),
                "historical_observed_spread": False,
                "side_month_rows": months,
            }
        )
    manifest = {
        "schema": "historical_score_economics_conversion_ledgers_v1",
        "status": "SOURCE_SEPARATED_CAUSAL_CONVERSION_INPUTS_MATERIALIZED",
        "identity": list(IDENTITY),
        "label_contract": {
            "decision": "signal + 1h",
            "resolution": "decision + 12h",
            "economics": "gross - cost = net",
            "exit_classes": ["trailing", "timeout", "full_stop", "adverse_exit"],
            "historical_observed_spread_available": False,
        },
        "selection_contract": (
            "Evaluate one pooled global top-k within each source family with "
            "candidate_id tie-breaking; never pool source families."
        ),
        "promotion_contract": (
            "Only canonical exact-1m current-spread-counterfactual families "
            "may support promotion; older and reconstructed families are diagnostic."
        ),
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "ledgers": catalog,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    outputs["manifest"] = manifest_path
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-base", type=Path, default=DEFAULT_CANONICAL_BASE)
    parser.add_argument(
        "--canonical-residual", type=Path, default=DEFAULT_CANONICAL_RESIDUAL
    )
    parser.add_argument(
        "--canonical-population", type=Path, default=DEFAULT_CANONICAL_POPULATION
    )
    parser.add_argument(
        "--reconstructed-exact-scores",
        type=Path,
        default=DEFAULT_RECONSTRUCTED_EXACT_SCORES,
    )
    parser.add_argument(
        "--reconstructed-exact-labels",
        type=Path,
        default=DEFAULT_RECONSTRUCTED_EXACT_LABELS,
    )
    parser.add_argument(
        "--late2024-scores", type=Path, default=DEFAULT_LATE2024_SCORES
    )
    parser.add_argument(
        "--late2024-labels", type=Path, default=DEFAULT_LATE2024_LABELS
    )
    parser.add_argument(
        "--historical-hourly", type=Path, default=DEFAULT_HISTORICAL_HOURLY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    outputs = run(_parser().parse_args())
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
