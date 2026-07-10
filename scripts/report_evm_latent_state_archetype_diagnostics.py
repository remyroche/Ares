#!/usr/bin/env python3
"""Discover OOS-favorable and unfavorable latent states for the EV calibrator.

This is a diagnostic/reporting tool for the per ``side x archetype`` EVM
calibration layer.  It does not fit GMMs and it excludes AE/GMM columns by
default.  It fits train-only feature thresholds, assigns OOS rows with those
frozen thresholds, and reports which observable states improve or degrade the
top-k trading stream for each archetype.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.evm_latent_state_discovery import (  # noqa: E402
    DEFAULT_AEGMM_HINTS,
    DEFAULT_EVM_FEATURE_HINTS,
    DEFAULT_EVM_PRIORITY_FEATURE_HINTS,
    DEFAULT_SCORE_COL,
    EvmLatentStateConfig,
    discover_evm_latent_states,
    downcast_numeric,
    evm_feature_priority_score,
    is_market_context_shock_entropy_feature,
    select_evm_state_feature_columns,
)

try:  # Reuse the current calibration defaults without importing at module import time in tests.
    from scripts.report_meta_oos_regime_calibration import (  # noqa: E402
        ARCH_COL,
        BASE_SCORE_COL,
        DEFAULT_HANDOFF,
        DEFAULT_META_RUN,
        KEYS,
        OUTCOME_COLS,
        SCORE_COL,
    )
except Exception:  # pragma: no cover
    DEFAULT_META_RUN = Path(".")
    DEFAULT_HANDOFF = Path(".")
    SCORE_COL = DEFAULT_SCORE_COL
    BASE_SCORE_COL = "score_base"
    ARCH_COL = "archetype_policy_key"
    OUTCOME_COLS = ["ev_after_1pct", "clean_exec", "dirty_positive", "full_path_bad_mae_1r", "timeout"]
    KEYS = ["__ts__", "__symbol__", "side_name"]


DEFAULT_OUT = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260706/"
    "evm_latent_state_archetype_diagnostics_20260710"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _schema_cols(path: Path) -> list[str]:
    return pq.read_schema(path).names


def _prediction_shard(meta_run: Path, month: str) -> Path:
    matches = sorted((meta_run / "prediction_shards").glob(f"*{month}.parquet"))
    if not matches:
        raise FileNotFoundError(f"No prediction shard for {month} under {meta_run / 'prediction_shards'}")
    return matches[-1]


def _read_parquet(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if columns is None:
        return pq.read_table(path).to_pandas()
    schema = set(_schema_cols(path))
    cols = [col for col in dict.fromkeys(str(c) for c in columns) if col in schema]
    if not cols:
        raise ValueError(f"No readable columns resolved for {path}")
    return pq.read_table(path, columns=cols).to_pandas()


def _load_predictions(meta_run: Path, months: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month in months:
        frames.append(_read_parquet(_prediction_shard(meta_run, month)))
    pred = pd.concat(frames, ignore_index=True, copy=False)
    pred["__ts__"] = pd.to_datetime(pred["__ts__"], utc=True, errors="coerce")
    pred["month"] = pred["__ts__"].dt.to_period("M").astype(str)
    pred["week_start"] = pred["__ts__"].dt.to_period("W-MON").apply(lambda p: p.start_time.date().isoformat())
    pred["__symbol__"] = pred["__symbol__"].astype(str)
    pred["side_name"] = pred["side_name"].astype(str)
    if ARCH_COL not in pred.columns:
        pred[ARCH_COL] = pred.get("__archetype_policy_key__", pred.get("policy_archetype", "missing"))
    pred[ARCH_COL] = pred[ARCH_COL].astype(str).replace({"nan": "missing", "None": "missing"})
    return downcast_numeric(pred)


def _derive_prediction_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame
    if SCORE_COL in out.columns and BASE_SCORE_COL in out.columns:
        out["__derived_score_dispersion__"] = (
            pd.to_numeric(out[SCORE_COL], errors="coerce")
            - pd.to_numeric(out[BASE_SCORE_COL], errors="coerce")
        ).abs().astype("float32")
    if SCORE_COL in out.columns:
        p = pd.to_numeric(out[SCORE_COL], errors="coerce").clip(1e-6, 1.0 - 1e-6)
        out["__derived_meta_uncertainty__"] = (
            -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2.0)
        ).astype("float32")
        out["__derived_score_rank_by_timestamp_side__"] = (
            p.groupby([out["__ts__"], out["side_name"]], sort=False)
            .rank(pct=True, method="first")
            .astype("float32")
        )
    if SCORE_COL in out.columns and ARCH_COL in out.columns:
        out["__derived_score_rank_by_month_side_archetype__"] = (
            pd.to_numeric(out[SCORE_COL], errors="coerce")
            .groupby([out["month"], out["side_name"], out[ARCH_COL]], sort=False)
            .rank(pct=True, method="first")
            .astype("float32")
        )
    return out


def _name_based_candidate_cols(
    cols: Iterable[str],
    *,
    include_aegmm: bool,
    required_cols: set[str],
) -> list[str]:
    selected: list[str] = []
    lower_hints = tuple(h.lower() for h in DEFAULT_EVM_FEATURE_HINTS)
    aegmm_hints = tuple(h.lower() for h in DEFAULT_AEGMM_HINTS)
    target_hints = (
        "target",
        "label",
        "future",
        "oracle",
        "realized",
        "ret_net",
        "net_return",
        "gross_return",
        "pnl",
        "profit",
        "loss",
        "first_touch",
        "full_path",
        "timeout",
        "stop",
        "bad_mae",
        "exec_margin",
        "clean_exec",
        "dirty_positive",
        "adverse",
        "diagnostic_only",
        "outcome",
        "exit_",
        "position_",
    )
    for col in cols:
        text = str(col)
        lower = text.lower()
        if text in required_cols:
            continue
        if any(h in lower for h in target_hints):
            continue
        if not include_aegmm and any(h in lower for h in aegmm_hints):
            continue
        if not is_market_context_shock_entropy_feature(text):
            continue
        if any(h in lower for h in lower_hints):
            selected.append(text)
    return selected


def _is_derivable_raw_shock_entropy_col(name: str, *, include_aegmm: bool, required_cols: set[str]) -> bool:
    text = str(name)
    lower = text.lower()
    if text in required_cols:
        return False
    if "shock" not in lower and "entropy" not in lower:
        return False
    if is_market_context_shock_entropy_feature(text):
        return False
    if not include_aegmm and any(h in lower for h in tuple(h.lower() for h in DEFAULT_AEGMM_HINTS)):
        return False
    target_hints = (
        "target",
        "label",
        "future",
        "oracle",
        "realized",
        "ret_net",
        "net_return",
        "gross_return",
        "pnl",
        "profit",
        "loss",
        "first_touch",
        "full_path",
        "timeout",
        "stop",
        "bad_mae",
        "exec_margin",
        "clean_exec",
        "dirty_positive",
        "adverse",
        "diagnostic_only",
        "outcome",
        "exit_",
        "position_",
    )
    return not any(h in lower for h in target_hints)


def _prioritize_candidate_cols(cols: list[str], max_feature_cols: int) -> list[str]:
    if not max_feature_cols or len(cols) <= int(max_feature_cols):
        return cols
    indexed = list(enumerate(cols))
    indexed.sort(key=lambda item: (evm_feature_priority_score(item[1]), -item[0]), reverse=True)
    return [col for _, col in indexed[: int(max_feature_cols)]]


def _load_feature_handoff(
    handoff: Path,
    pred: pd.DataFrame,
    *,
    include_aegmm: bool,
    max_feature_cols: int,
) -> tuple[pd.DataFrame, list[str]]:
    handoff_schema_cols = _schema_cols(handoff)
    handoff_cols = set(handoff_schema_cols)
    pred_cols = set(pred.columns)
    required = set(KEYS)
    derived_cols = [
        "__derived_score_dispersion__",
        "__derived_meta_uncertainty__",
        "__derived_score_rank_by_timestamp_side__",
        "__derived_score_rank_by_month_side_archetype__",
    ]
    candidate_source_cols = list(dict.fromkeys([*pred.columns, *handoff_schema_cols, *derived_cols]))
    candidates = _name_based_candidate_cols(
        candidate_source_cols,
        include_aegmm=include_aegmm,
        required_cols=required | set(OUTCOME_COLS) | {SCORE_COL, ARCH_COL},
    )
    handoff_feature_cols = [col for col in candidates if col in handoff_cols and col not in pred_cols]
    derivable_context_cols = [
        col
        for col in handoff_schema_cols
        if col in handoff_cols
        and col not in pred_cols
        and _is_derivable_raw_shock_entropy_col(
            col,
            include_aegmm=include_aegmm,
            required_cols=required | set(OUTCOME_COLS) | {SCORE_COL, ARCH_COL},
        )
    ]
    if max_feature_cols and len(handoff_feature_cols) > int(max_feature_cols):
        handoff_feature_cols = _prioritize_candidate_cols(handoff_feature_cols, int(max_feature_cols))
    read_cols = [col for col in [*KEYS, *handoff_feature_cols, *derivable_context_cols] if col in handoff_cols]
    if not read_cols or set(read_cols).issubset(set(KEYS)):
        return pd.DataFrame(columns=KEYS), [col for col in candidates if col in pred_cols]
    features = _read_parquet(handoff, read_cols)
    features["__ts__"] = pd.to_datetime(features["__ts__"], utc=True, errors="coerce")
    features["month"] = features["__ts__"].dt.to_period("M").astype(str)
    features = features.loc[features["month"].isin(sorted(pred["month"].unique()))].copy()
    features["__symbol__"] = features["__symbol__"].astype(str)
    features["side_name"] = features["side_name"].astype(str)
    features = features.drop(columns=["month"]).drop_duplicates(KEYS)
    return downcast_numeric(features), candidates


def _safe_feature_token(name: str) -> str:
    token = str(name)
    for ch in "/: ()[]{}+-*.,%":
        token = token.replace(ch, "_")
    return "_".join(part for part in token.split("_") if part)


def _derive_market_shock_entropy_context(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Derive market-wide shock/entropy context from raw columns.

    Raw per-asset shock/entropy columns are deliberately not selected directly.
    They are only used here to create timestamp-level market state features.
    """

    raw_cols = [
        col
        for col in frame.columns
        if _is_derivable_raw_shock_entropy_col(
            col,
            include_aegmm=True,
            required_cols=set(KEYS) | set(OUTCOME_COLS) | {SCORE_COL, ARCH_COL},
        )
    ]
    if not raw_cols or "__ts__" not in frame.columns:
        return frame, []
    derived: dict[str, pd.Series] = {}
    grouped = frame.groupby("__ts__", sort=False, observed=True)
    for col in raw_cols:
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite_share = float(values.notna().mean())
        if finite_share < 0.20 or int(values.nunique(dropna=True)) < 8:
            continue
        token = _safe_feature_token(col)
        median = grouped[col].transform("median")
        q75 = grouped[col].transform(lambda s: pd.to_numeric(s, errors="coerce").quantile(0.75))
        q25 = grouped[col].transform(lambda s: pd.to_numeric(s, errors="coerce").quantile(0.25))
        q90 = grouped[col].transform(lambda s: pd.to_numeric(s, errors="coerce").quantile(0.90))
        derived[f"market_{token}_median"] = pd.to_numeric(median, errors="coerce").astype("float32")
        derived[f"market_{token}_iqr"] = (pd.to_numeric(q75, errors="coerce") - pd.to_numeric(q25, errors="coerce")).astype("float32")
        derived[f"market_{token}_p90"] = pd.to_numeric(q90, errors="coerce").astype("float32")
    if not derived:
        return frame, []
    derived_frame = pd.DataFrame(derived, index=frame.index)
    out = pd.concat([frame, derived_frame], axis=1, copy=False)
    return downcast_numeric(out), list(derived)


def _prepare_panel(
    *,
    meta_run: Path,
    handoff: Path,
    months: list[str],
    include_aegmm: bool,
    max_feature_cols: int,
) -> tuple[pd.DataFrame, list[str]]:
    pred = _derive_prediction_features(_load_predictions(meta_run, months))
    features, candidate_names = _load_feature_handoff(
        handoff,
        pred,
        include_aegmm=include_aegmm,
        max_feature_cols=max_feature_cols,
    )
    merged = pred.merge(features, on=KEYS, how="left", validate="many_to_one") if len(features) else pred
    merged = downcast_numeric(merged)
    merged, derived_market_context_cols = _derive_market_shock_entropy_context(merged)
    required = [*KEYS, "month", "week_start", SCORE_COL, ARCH_COL, *OUTCOME_COLS]
    feature_cols = select_evm_state_feature_columns(
        merged,
        include_aegmm=include_aegmm,
        required_columns=required,
        max_columns=max_feature_cols,
    )
    # Keep the name-based handoff candidates if they survived dtype pruning.
    if candidate_names:
        allowed = set(candidate_names).union(derived_market_context_cols)
        feature_cols = [col for col in feature_cols if col in allowed or col.startswith("__derived_")]
    return merged, feature_cols


def _write_result(
    out_dir: Path,
    *,
    eval_month: str,
    result: Any,
) -> None:
    prefix = eval_month.replace("-", "")
    result.feature_state_metrics.to_csv(out_dir / f"{prefix}_feature_state_oos_metrics.csv", index=False)
    result.pair_state_metrics.to_csv(out_dir / f"{prefix}_pair_state_oos_metrics.csv", index=False)
    result.baselines.to_csv(out_dir / f"{prefix}_group_baselines.csv", index=False)
    result.thresholds.to_csv(out_dir / f"{prefix}_train_state_thresholds.csv", index=False)
    result.catalog.to_csv(out_dir / f"{prefix}_latent_state_catalog.csv", index=False)
    (out_dir / f"{prefix}_manifest.json").write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-run", type=Path, default=DEFAULT_META_RUN)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--all-months", nargs="+", default=["2026-04", "2026-05", "2026-06"])
    parser.add_argument("--eval-months", nargs="+", default=["2026-05", "2026-06"])
    parser.add_argument("--include-aegmm", action="store_true", help="Allow AE/GMM columns as candidate state features.")
    parser.add_argument("--max-feature-cols", type=int, default=120)
    parser.add_argument("--min-group-rows", type=int, default=160)
    parser.add_argument("--min-state-rows", type=int, default=30)
    parser.add_argument("--max-features-per-group", type=int, default=24)
    parser.add_argument("--top-features-for-pairs", type=int, default=6)
    parser.add_argument("--no-pair-states", action="store_true")
    parser.add_argument("--min-oos-objective-delta", type=float, default=0.0002)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel, feature_cols = _prepare_panel(
        meta_run=args.meta_run,
        handoff=args.handoff,
        months=list(args.all_months),
        include_aegmm=bool(args.include_aegmm),
        max_feature_cols=int(args.max_feature_cols),
    )
    required = [SCORE_COL, ARCH_COL, "ev_after_1pct"]
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise RuntimeError(f"Missing required panel columns: {missing}")
    panel = panel.rename(columns={ARCH_COL: "archetype_policy_key"}) if ARCH_COL != "archetype_policy_key" else panel
    if SCORE_COL != DEFAULT_SCORE_COL and SCORE_COL in panel.columns:
        panel[DEFAULT_SCORE_COL] = panel[SCORE_COL]
    if "stop_or_adverse" not in panel.columns:
        if "full_stop_loss" in panel.columns:
            panel["stop_or_adverse"] = pd.to_numeric(panel["full_stop_loss"], errors="coerce").fillna(0.0).astype("int8")
        else:
            panel["stop_or_adverse"] = 0
    config = EvmLatentStateConfig(
        min_group_rows=int(args.min_group_rows),
        min_state_rows=int(args.min_state_rows),
        max_features_per_group=int(args.max_features_per_group),
        top_features_for_pairs=int(args.top_features_for_pairs),
        include_pair_states=not bool(args.no_pair_states),
        min_oos_objective_delta=float(args.min_oos_objective_delta),
    )
    all_catalog_parts: list[pd.DataFrame] = []
    all_feature_parts: list[pd.DataFrame] = []
    all_pair_parts: list[pd.DataFrame] = []
    all_baseline_parts: list[pd.DataFrame] = []
    all_threshold_parts: list[pd.DataFrame] = []
    fold_manifests: dict[str, Any] = {}
    for eval_month in args.eval_months:
        train = panel.loc[panel["month"].lt(eval_month)].copy()
        oos = panel.loc[panel["month"].eq(eval_month)].copy()
        if train.empty or oos.empty:
            continue
        result = discover_evm_latent_states(
            train,
            oos,
            feature_cols,
            config=config,
            eval_label=eval_month,
        )
        _write_result(args.output_dir, eval_month=eval_month, result=result)
        fold_manifests[eval_month] = result.manifest
        for frame, parts in (
            (result.catalog, all_catalog_parts),
            (result.feature_state_metrics, all_feature_parts),
            (result.pair_state_metrics, all_pair_parts),
            (result.baselines, all_baseline_parts),
            (result.thresholds, all_threshold_parts),
        ):
            if not frame.empty:
                tagged = frame.copy()
                tagged["eval_month"] = eval_month
                parts.append(tagged)

    combined_outputs: dict[str, pd.DataFrame] = {}
    for name, parts in (
        ("latent_state_catalog.csv", all_catalog_parts),
        ("feature_state_oos_metrics.csv", all_feature_parts),
        ("pair_state_oos_metrics.csv", all_pair_parts),
        ("group_baselines.csv", all_baseline_parts),
        ("train_state_thresholds.csv", all_threshold_parts),
    ):
        if parts:
            combined = pd.concat(parts, ignore_index=True, copy=False)
            combined_outputs[name] = combined
            combined.to_csv(args.output_dir / name, index=False)
        else:
            combined_outputs[name] = pd.DataFrame()
            pd.DataFrame().to_csv(args.output_dir / name, index=False)

    catalog_combined = combined_outputs["latent_state_catalog.csv"]
    if not catalog_combined.empty:
        side_summary = (
            catalog_combined.groupby(
                ["eval_month", "side_name", "archetype_policy_key", "direction", "scope"],
                as_index=False,
                observed=True,
            )
            .agg(
                states=("state_name", "count"),
                mean_oos_objective_delta=("oos_objective_delta", "mean"),
                max_abs_oos_objective_delta=(
                    "oos_objective_delta",
                    lambda s: pd.to_numeric(s, errors="coerce").abs().max(),
                ),
                mean_oos_ev_after_1pct=("oos_mean_ev_after_1pct", "mean"),
                median_oos_rows=("oos_rows", "median"),
            )
            .sort_values(["eval_month", "side_name", "archetype_policy_key", "scope", "direction"])
        )
        feature_summary = (
            catalog_combined.groupby(["feature", "direction"], as_index=False, observed=True)
            .agg(
                states=("state_name", "count"),
                months=("eval_month", "nunique"),
                archetypes=("archetype_policy_key", "nunique"),
                mean_oos_objective_delta=("oos_objective_delta", "mean"),
                max_abs_oos_objective_delta=(
                    "oos_objective_delta",
                    lambda s: pd.to_numeric(s, errors="coerce").abs().max(),
                ),
                min_oos_rows=("oos_rows", "min"),
            )
            .sort_values(["months", "states", "max_abs_oos_objective_delta"], ascending=[False, False, False])
        )
    else:
        side_summary = pd.DataFrame()
        feature_summary = pd.DataFrame()
    side_summary.to_csv(args.output_dir / "side_archetype_state_summary.csv", index=False)
    feature_summary.to_csv(args.output_dir / "feature_state_summary.csv", index=False)

    state_definitions: list[dict[str, Any]] = []
    if not catalog_combined.empty:
        catalog = catalog_combined
        for _, row in catalog.sort_values("oos_objective_delta", ascending=False).head(500).iterrows():
            state_definitions.append(
                {
                    "side_name": str(row.get("side_name", "")),
                    "archetype_policy_key": str(row.get("archetype_policy_key", "")),
                    "state_name": str(row.get("state_name", "")),
                    "state_kind": str(row.get("state_kind", "")),
                    "direction": str(row.get("direction", "")),
                    "scope": str(row.get("scope", "")),
                    "feature": str(row.get("feature", "")),
                    "feature_bin": str(row.get("feature_bin", "")),
                    "q_low": float(row.get("q_low", np.nan)),
                    "q_high": float(row.get("q_high", np.nan)),
                    "feature_b": str(row.get("feature_b", "")),
                    "feature_b_bin": str(row.get("feature_b_bin", "")),
                    "q_low_b": float(row.get("q_low_b", np.nan)),
                    "q_high_b": float(row.get("q_high_b", np.nan)),
                    "oos_objective_delta": float(row.get("oos_objective_delta", np.nan)),
                    "oos_mean_ev_after_1pct": float(row.get("oos_mean_ev_after_1pct", np.nan)),
                    "oos_rows": int(row.get("oos_rows", 0) or 0),
                }
            )
    (args.output_dir / "latent_state_candidate_definitions.json").write_text(
        json.dumps(
            {
                "artifact_type": "evm_latent_state_candidate_definitions",
                "note": "diagnostic states for later EVM calibration; not promoted live calibration",
                "include_aegmm": bool(args.include_aegmm),
                "states": state_definitions,
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    manifest = {
        "generated_by": "report_evm_latent_state_archetype_diagnostics.py",
        "meta_run": str(args.meta_run),
        "handoff": str(args.handoff),
        "output_dir": str(args.output_dir),
        "all_months": list(args.all_months),
        "eval_months": list(args.eval_months),
        "include_aegmm": bool(args.include_aegmm),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "folds": fold_manifests,
        "outputs": {
            "latent_state_catalog": str(args.output_dir / "latent_state_catalog.csv"),
            "feature_state_oos_metrics": str(args.output_dir / "feature_state_oos_metrics.csv"),
            "pair_state_oos_metrics": str(args.output_dir / "pair_state_oos_metrics.csv"),
            "group_baselines": str(args.output_dir / "group_baselines.csv"),
            "train_state_thresholds": str(args.output_dir / "train_state_thresholds.csv"),
            "side_archetype_state_summary": str(args.output_dir / "side_archetype_state_summary.csv"),
            "feature_state_summary": str(args.output_dir / "feature_state_summary.csv"),
            "latent_state_candidate_definitions": str(args.output_dir / "latent_state_candidate_definitions.json"),
        },
        "leakage_contract": {
            "regime_features": "pre-entry/meta/context columns only; target/outcome columns excluded",
            "aegmm_default": "AE/GMM columns excluded unless --include-aegmm is passed",
            "fit": "state thresholds are fit on rows before the eval month",
            "transform": "eval rows receive frozen train thresholds only",
            "assessment": "outcomes are used only for metrics after OOS state assignment",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps({"event": "evm_latent_state_diagnostics_done", **manifest}, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
