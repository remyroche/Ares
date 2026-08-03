#!/usr/bin/env python3
"""Materialize outcome-only economic conversion transition labels.

This consumes the immutable canonical opportunity/payoff/trust panel and
produces global-hour, side, and causal base-score-decile cohort outcomes.  It
does not fit a model and deliberately has no feature surface: every emitted
metric is a label whose availability is after the relevant future window has
fully resolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2"
DEFAULT_OUTPUT = (
    ROOT / "data_perp/artifacts/canonical_economic_conversion_transition_labels_20260729_v1"
)
EXIT_CLASSES = ("trailing", "timeout", "full_stop", "adverse_exit")
HORIZONS = ((12, "primary"), (3, "auxiliary"))
DECILES = tuple(range(10))
POST_WINDOW_PUBLICATION_LAG = pd.Timedelta(hours=1)
REQUIRED_COLUMNS = (
    "candidate_id",
    "side_name",
    "__symbol__",
    "__ts__",
    "base_oof_score",
    "execution_label_end_utc",
    "execution_net_ev_12h",
    "execution_exit_class",
    "opportunity_gross_above_cost_0bps",
    "opportunity_gross_above_cost_25bps",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
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


def _normalise_source(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"canonical source lacks required columns: {missing}")
    output = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    output["candidate_id"] = output["candidate_id"].astype(str)
    output["side_name"] = output["side_name"].astype(str).str.lower()
    output["__symbol__"] = output["__symbol__"].astype(str)
    if not output["side_name"].isin(("long", "short")).all():
        raise ValueError("canonical source has non-canonical sides")
    for column in ("__ts__", "execution_label_end_utc"):
        output[column] = pd.to_datetime(output[column], utc=True, errors="raise")
        if not output[column].dt.floor("h").eq(output[column]).all():
            raise ValueError(f"{column} must be UTC-aligned to the hour")
    for column in (
        "base_oof_score",
        "execution_net_ev_12h",
        "opportunity_gross_above_cost_0bps",
        "opportunity_gross_above_cost_25bps",
    ):
        output[column] = pd.to_numeric(output[column], errors="coerce")
        if not np.isfinite(output[column]).all():
            raise ValueError(f"{column} must be finite")
    output["execution_exit_class"] = output["execution_exit_class"].astype(str)
    if not output["execution_exit_class"].isin(EXIT_CLASSES).all():
        raise ValueError("source exit classes do not match the canonical four-class contract")
    if output["candidate_id"].duplicated().any():
        raise ValueError("candidate identity must be unique")
    return output


def add_frozen_causal_score_deciles(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic score-only deciles at each timestamp and side.

    The stable order is score descending then symbol then candidate ID.  No
    realised economic, opportunity, or exit field participates in this step.
    Decile zero is the highest score cohort; this remains deterministic even
    when scores tie.
    """

    required = {"__ts__", "side_name", "__symbol__", "candidate_id", "base_oof_score"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"score-decile input lacks: {missing}")
    output = frame.copy().sort_values(
        ["__ts__", "side_name", "base_oof_score", "__symbol__", "candidate_id"],
        ascending=[True, True, False, True, True],
        kind="stable",
    )
    groups = output.groupby(["__ts__", "side_name"], sort=False, observed=True)
    rank_zero = groups.cumcount().to_numpy(dtype=np.int64)
    size = groups["base_oof_score"].transform("size").to_numpy(dtype=np.int64)
    output["frozen_base_score_decile"] = ((rank_zero * 10) // size).clip(0, 9).astype(np.int8)
    output["frozen_base_score_decile_group_rows"] = size.astype(np.int32)
    return output.sort_values(["__ts__", "side_name", "__symbol__", "candidate_id"], kind="stable").reset_index(drop=True)


def robust_mean(values: Iterable[float], *, lower: float = 0.05, upper: float = 0.95) -> float:
    """Return a deterministic within-window winsorized mean (p05/p95)."""

    numeric = np.asarray(list(values), dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if not len(numeric):
        return float("nan")
    low, high = np.quantile(numeric, (lower, upper))
    return float(np.clip(numeric, low, high).mean())


def _window_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    """Compute one cohort-window's strictly outcome-only targets and support."""

    support = int(len(rows))
    result: dict[str, Any] = {
        "candidate_support": support,
        "window_missing_support_flag": support == 0,
        "opportunity_support": support,
        "direct_mean_net": float("nan"),
        "opportunity_probability_0bps": float("nan"),
        "opportunity_probability_25bps": float("nan"),
        "favorable_net_support": 0,
        "conditional_favorable_net_robust_mean": float("nan"),
        "conditional_favorable_net_q50": float("nan"),
        "favorable_net_missing_support_flag": True,
        "adverse_loss_support": 0,
        "conditional_adverse_loss_robust_mean": float("nan"),
        "conditional_adverse_loss_q80": float("nan"),
        "adverse_loss_missing_support_flag": True,
        "target_available_utc": pd.NaT,
    }
    for exit_class in EXIT_CLASSES:
        result[f"exit_{exit_class}_support"] = 0
        result[f"p_exit_{exit_class}"] = float("nan")
        result[f"conditional_net_{exit_class}"] = float("nan")
        result[f"exit_{exit_class}_missing_support_flag"] = True
    if not support:
        result["exit_mixture_expected_net"] = float("nan")
        result["exit_mixture_reconciles_direct_mean_flag"] = False
        return result

    net = rows["execution_net_ev_12h"].to_numpy(dtype=float)
    result["direct_mean_net"] = float(net.mean())
    result["opportunity_probability_0bps"] = float(
        rows["opportunity_gross_above_cost_0bps"].mean()
    )
    result["opportunity_probability_25bps"] = float(
        rows["opportunity_gross_above_cost_25bps"].mean()
    )
    favorable = net[net > 0.0]
    adverse_loss = -net[net <= 0.0]
    result["favorable_net_support"] = int(len(favorable))
    result["favorable_net_missing_support_flag"] = not len(favorable)
    if len(favorable):
        result["conditional_favorable_net_robust_mean"] = robust_mean(favorable)
        result["conditional_favorable_net_q50"] = float(np.quantile(favorable, 0.50))
    result["adverse_loss_support"] = int(len(adverse_loss))
    result["adverse_loss_missing_support_flag"] = not len(adverse_loss)
    if len(adverse_loss):
        result["conditional_adverse_loss_robust_mean"] = robust_mean(adverse_loss)
        result["conditional_adverse_loss_q80"] = float(np.quantile(adverse_loss, 0.80))

    mixture = 0.0
    for exit_class in EXIT_CLASSES:
        mask = rows["execution_exit_class"].eq(exit_class).to_numpy()
        count = int(mask.sum())
        probability = count / support
        result[f"exit_{exit_class}_support"] = count
        result[f"p_exit_{exit_class}"] = float(probability)
        result[f"exit_{exit_class}_missing_support_flag"] = count == 0
        conditional = float(net[mask].mean()) if count else float("nan")
        result[f"conditional_net_{exit_class}"] = conditional
        if count:
            mixture += probability * conditional
    result["exit_mixture_expected_net"] = float(mixture)
    result["exit_mixture_reconciles_direct_mean_flag"] = bool(
        np.isclose(mixture, result["direct_mean_net"], rtol=0.0, atol=1e-12)
    )
    latest_end = rows["execution_label_end_utc"].max()
    result["target_available_utc"] = latest_end + POST_WINDOW_PUBLICATION_LAG
    return result


def _global_hour_completeness(hours: pd.DatetimeIndex, anchor: pd.Timestamp, horizon_hours: int) -> tuple[int, bool, int, bool]:
    """Return observed hour counts and exact half-open completeness flags."""

    before = pd.date_range(anchor - pd.Timedelta(hours=horizon_hours), periods=horizon_hours, freq="h", tz="UTC")
    after = pd.date_range(anchor, periods=horizon_hours, freq="h", tz="UTC")
    observed = set(hours)
    before_count = sum(stamp in observed for stamp in before)
    after_count = sum(stamp in observed for stamp in after)
    return before_count, before_count == horizon_hours, after_count, after_count == horizon_hours


def materialize_transition_labels(source: pd.DataFrame) -> pd.DataFrame:
    """Build global-hour × side × causal-score-decile before/after labels."""

    rows = add_frozen_causal_score_deciles(_normalise_source(source))
    global_hours = pd.DatetimeIndex(sorted(rows["__ts__"].unique()))
    if not len(global_hours):
        raise ValueError("canonical source is empty")
    side_values = ("long", "short")
    records: list[dict[str, Any]] = []
    for horizon_hours, horizon_role in HORIZONS:
        horizon = pd.Timedelta(hours=horizon_hours)
        for side in side_values:
            for decile in DECILES:
                cohort = rows.loc[
                    rows["side_name"].eq(side)
                    & rows["frozen_base_score_decile"].eq(decile)
                ].sort_values("__ts__", kind="stable")
                stamps = cohort["__ts__"].to_numpy(dtype="datetime64[ns]")
                for anchor in global_hours:
                    # searchsorted gives the exact half-open [start,end) windows.
                    before_left = np.searchsorted(stamps, (anchor - horizon).to_datetime64(), side="left")
                    before_right = np.searchsorted(stamps, anchor.to_datetime64(), side="left")
                    after_left = before_right
                    after_right = np.searchsorted(stamps, (anchor + horizon).to_datetime64(), side="left")
                    before = _window_metrics(cohort.iloc[before_left:before_right])
                    after = _window_metrics(cohort.iloc[after_left:after_right])
                    before_hours, before_complete, after_hours, after_complete = _global_hour_completeness(global_hours, anchor, horizon_hours)
                    record: dict[str, Any] = {
                        "cohort_anchor_utc": anchor,
                        "side_name": side,
                        "frozen_base_score_decile": decile,
                        "horizon_hours": horizon_hours,
                        "horizon_role": horizon_role,
                        "before_window_start_utc": anchor - horizon,
                        "before_window_end_utc": anchor,
                        "after_window_start_utc": anchor,
                        "after_window_end_utc": anchor + horizon,
                        "before_global_hour_support": before_hours,
                        "after_global_hour_support": after_hours,
                        "before_global_hour_complete_flag": before_complete,
                        "after_global_hour_complete_flag": after_complete,
                        "outcome_only_not_model_feature": True,
                    }
                    record.update({f"before_{name}": value for name, value in before.items()})
                    record.update({f"after_{name}": value for name, value in after.items()})
                    for metric in (
                        "direct_mean_net",
                        "opportunity_probability_0bps",
                        "opportunity_probability_25bps",
                        "conditional_favorable_net_robust_mean",
                        "conditional_favorable_net_q50",
                        "conditional_adverse_loss_robust_mean",
                        "conditional_adverse_loss_q80",
                        "exit_mixture_expected_net",
                        *[f"p_exit_{exit_class}" for exit_class in EXIT_CLASSES],
                        *[
                            f"conditional_net_{exit_class}"
                            for exit_class in EXIT_CLASSES
                        ],
                    ):
                        record[f"delta_{metric}"] = after[metric] - before[metric]
                    records.append(record)
    output = pd.DataFrame.from_records(records)
    output = output.sort_values(
        ["horizon_hours", "cohort_anchor_utc", "side_name", "frozen_base_score_decile"], kind="stable"
    ).reset_index(drop=True)
    return output


def row_support_summary(labels: pd.DataFrame) -> pd.DataFrame:
    return (
        labels.groupby(["horizon_hours", "horizon_role", "side_name", "frozen_base_score_decile"], observed=True, sort=True)
        .agg(
            rows=("cohort_anchor_utc", "size"),
            after_nonempty_windows=("after_window_missing_support_flag", lambda value: int((~value).sum())),
            before_nonempty_windows=("before_window_missing_support_flag", lambda value: int((~value).sum())),
            after_complete_global_windows=("after_global_hour_complete_flag", "sum"),
            before_complete_global_windows=("before_global_hour_complete_flag", "sum"),
            after_candidate_support_sum=("after_candidate_support", "sum"),
            before_candidate_support_sum=("before_candidate_support", "sum"),
            after_exit_reconciliation_failures=("after_exit_mixture_reconciles_direct_mean_flag", lambda value: int((~value).sum())),
        )
        .reset_index()
    )


def _source_hashes(source: Path) -> dict[str, str]:
    paths = [source / "panel.parquet", source / "manifest.json", source / "manifest.sha256"]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"immutable canonical source is incomplete: {missing}")
    return {str(path): sha256(path) for path in paths}


def plan(source: Path, output: Path) -> dict[str, Any]:
    hashes = _source_hashes(source)
    return {
        "action": "PLAN_ONLY_NO_MATERIALIZATION",
        "source": str(source),
        "output": str(output),
        "source_sha256": hashes,
        "schema": "canonical_economic_conversion_transition_labels_v1",
        "cohorts": "global-hour × side × frozen causal base-score-decile",
        "windows": {"primary": "H=12h", "auxiliary": "H=3h", "before": "[s-H,s)", "after": "[s,s+H)"},
        "availability": "max(actual execution_label_end_utc in after window) + 1h; a complete hourly H window is s+H+13h",
        "feature_surface": [],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source)
    output = Path(args.output_dir)
    if args.plan_only:
        return plan(source, output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    source_hashes = _source_hashes(source)
    labels = materialize_transition_labels(pd.read_parquet(source / "panel.parquet", columns=list(REQUIRED_COLUMNS)))
    summary = row_support_summary(labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    labels.to_parquet(temporary / "cohort_transition_labels.parquet", index=False, compression="zstd")
    summary.to_parquet(temporary / "row_support_summary.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "canonical_economic_conversion_transition_labels_v1",
        "status": "IMMUTABLE_OUTCOME_ONLY_LABEL_ARTIFACT",
        "source": str(source),
        "source_sha256": source_hashes,
        "rows": int(len(labels)),
        "label_columns_not_model_features": [name for name in labels.columns if name not in {"cohort_anchor_utc", "side_name", "frozen_base_score_decile", "horizon_hours", "horizon_role", "before_window_start_utc", "before_window_end_utc", "after_window_start_utc", "after_window_end_utc"}],
        "contracts": {
            "utc": "all stored timestamps are timezone-aware UTC",
            "score_decile": "at each timestamp/side, base_oof_score descending then symbol/candidate_id tie-break; outcome fields never participate",
            "windows": "before [s-H,s), after [s,s+H), evaluated on actual candidate timestamps",
            "availability": "after_target_available_utc is max actual execution_label_end_utc + 1h; complete hourly H windows resolve at s+H+13h",
            "conditional_payoffs": "favorable net conditions on net>0; adverse loss severity conditions on net<=0; p05/p95 winsorized mean and requested quantile",
            "exit_reconciliation": "sum P(exit_class)*E(net|exit_class) equals direct mean net within 1e-12",
            "feature_usage": "all columns in this artifact are outcomes/audit metadata only and are prohibited model features",
        },
        "support_summary": summary.to_dict(orient="records"),
        "outputs_sha256": {path.name: sha256(path) for path in sorted(temporary.glob("*.parquet"))},
        "checksum_convention": "manifest.json is verified by detached manifest.sha256",
    }
    (temporary / "manifest.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (temporary / "manifest.sha256").write_text(f"{sha256(temporary / 'manifest.json')}  manifest.json\n", encoding="utf-8")
    os.replace(temporary, output)
    return {"output": str(output), "rows": int(len(labels)), "source_sha256": source_hashes}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", type=Path, default=SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--plan-only", action="store_true", help="Validate immutable inputs and print the materialization contract without reading or writing label rows.")
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
