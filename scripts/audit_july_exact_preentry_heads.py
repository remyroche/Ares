#!/usr/bin/env python3
"""Research-only exact audit of frozen July pre-entry and execution-EV heads.

The runner joins the frozen Pack-B alpha layers, pre-entry auxiliary heads,
retrospective execution-EV scores, exact 1m policy labels, and decision-time
ATR/barrier geometry by exact candidate identity.  It never refits or promotes
models.  Seven-class path truth is intentionally not inferred from the
coarser exact policy outcomes; path probabilities receive economic and
explicitly-labelled proxy diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


SCHEMA = "july_exact_preentry_head_audit_v1"
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
SIDES = ("long", "short")
PATH_CLASSES = (
    "immediate_adverse_path",
    "early_mfe_full_reversal",
    "fast_realization_winner",
    "late_breakout",
    "slow_grinder",
    "noisy_timeout_usable_mfe",
    "dead_timeout",
)
MIN_USABLE_MFE_ATR = 1.5
MIN_USABLE_MFE_RETURN = 0.015
PEAK_MFE_ATR_CLIP = 10.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _nested(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"manifest key missing: {'.'.join(keys)}")
        value = value[key]
    return value


def validate_manifest_hash(
    data_path: Path,
    manifest_path: Path,
    hash_keys: Sequence[str],
    *,
    expected_schema: str,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != expected_schema:
        raise ValueError(
            f"unexpected manifest schema for {data_path}: {manifest.get('schema')!r}"
        )
    expected = str(_nested(manifest, hash_keys))
    observed = sha256(data_path)
    if observed != expected:
        raise ValueError(f"input hash mismatch for {data_path}")
    return {
        "path": str(data_path),
        "sha256": observed,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "manifest_schema": expected_schema,
    }


def _validate_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} identity columns missing: {missing}")
    work = frame.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    if work.duplicated(list(IDENTITY)).any():
        raise ValueError(f"{name} contains duplicate identities")
    if set(work["side_name"].astype(str)) != set(SIDES):
        raise ValueError(f"{name} must contain both sides")
    return work


def prepare_joined(
    packb: pd.DataFrame,
    preentry: pd.DataFrame,
    scored: pd.DataFrame,
    labels: pd.DataFrame,
    geometry: pd.DataFrame,
) -> pd.DataFrame:
    """Exact, fail-closed join and canonical target derivation."""

    frames = {
        "packb": _validate_frame(packb, "packb"),
        "preentry": _validate_frame(preentry, "preentry"),
        "scored": _validate_frame(scored, "scored"),
        "labels": _validate_frame(labels, "labels"),
        "geometry": _validate_frame(geometry, "geometry"),
    }
    key_sets = {
        name: set(map(tuple, frame.loc[:, IDENTITY].itertuples(index=False, name=None)))
        for name, frame in frames.items()
    }
    reference = key_sets["packb"]
    for name, keys in key_sets.items():
        if keys != reference:
            raise ValueError(
                f"{name} identities differ from Pack-B "
                f"(missing={len(reference - keys)}, extra={len(keys - reference)})"
            )

    packb_columns = [
        *IDENTITY,
        "execution_decision_utc",
        "base_prediction",
        "base_alpha_ev",
        "residual_delta_ev",
        "existing_alpha_ev",
    ]
    preentry_columns = [
        *IDENTITY,
        "oof_clean_favorable_probability",
        "pred_peak_MFE_12h_ATR",
        "catboost_archetype",
        *[f"catboost_p_{index}" for index in range(7)],
    ]
    scored_columns = [
        *IDENTITY,
        "final_direct_net_raw",
        "final_capture_probability",
        "mapped_execution_ev",
    ]
    label_columns = [
        *IDENTITY,
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_exit_reason",
        "execution_exit_hour",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        "execution_label_end_utc",
    ]
    geometry_columns = [
        *IDENTITY,
        "__barrier_pct__",
        "__path_auxiliary_atr_fraction__",
    ]
    selections = {
        "packb": packb_columns,
        "preentry": preentry_columns,
        "scored": scored_columns,
        "labels": label_columns,
        "geometry": geometry_columns,
    }
    for name, columns in selections.items():
        missing = sorted(set(columns).difference(frames[name].columns))
        if missing:
            raise ValueError(f"{name} audit columns missing: {missing}")

    output = frames["packb"].loc[:, packb_columns].copy()
    for name in ("preentry", "scored", "labels", "geometry"):
        output = output.merge(
            frames[name].loc[:, selections[name]],
            on=list(IDENTITY),
            how="inner",
            validate="one_to_one",
        )
    if len(output) != len(reference):
        raise AssertionError("exact audit join changed row count")

    numeric = [
        "base_prediction",
        "base_alpha_ev",
        "residual_delta_ev",
        "existing_alpha_ev",
        "oof_clean_favorable_probability",
        "pred_peak_MFE_12h_ATR",
        *[f"catboost_p_{index}" for index in range(7)],
        "final_direct_net_raw",
        "final_capture_probability",
        "mapped_execution_ev",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_exit_hour",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        "__barrier_pct__",
        "__path_auxiliary_atr_fraction__",
    ]
    for column in numeric:
        output[column] = pd.to_numeric(output[column], errors="raise")
    if not np.isfinite(output[numeric].to_numpy(dtype=float)).all():
        raise ValueError("audit inputs contain non-finite numeric values")
    if (output["__path_auxiliary_atr_fraction__"] <= 0.0).any():
        raise ValueError("decision-time ATR fraction must be positive")
    if (output["__barrier_pct__"] <= 0.0).any():
        raise ValueError("exact policy barrier must be positive")
    probabilities = output[[f"catboost_p_{index}" for index in range(7)]].to_numpy()
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("path CatBoost probabilities do not sum to one")
    if (
        (output[["oof_clean_favorable_probability", "final_capture_probability"]] < 0)
        | (output[["oof_clean_favorable_probability", "final_capture_probability"]] > 1)
    ).any().any():
        raise ValueError("probability head output is outside [0, 1]")

    atr = output["__path_auxiliary_atr_fraction__"].to_numpy(float)
    raw_peak_atr = output["execution_mfe_return_12h"].to_numpy(float) / atr
    raw_mae_atr = output["execution_mae_return_12h"].to_numpy(float) / atr
    threshold_return = np.maximum(MIN_USABLE_MFE_ATR * atr, MIN_USABLE_MFE_RETURN)
    meaningful = output["execution_mfe_return_12h"].to_numpy(float) >= threshold_return
    output["exact_peak_mfe_atr_raw"] = np.clip(raw_peak_atr, 0.0, PEAK_MFE_ATR_CLIP)
    output["exact_peak_mfe_atr_canonical"] = np.where(
        meaningful, output["exact_peak_mfe_atr_raw"], 0.0
    )
    output["exact_mae_atr"] = np.clip(raw_mae_atr, 0.0, PEAK_MFE_ATR_CLIP)
    output["meaningful_mfe_threshold_return"] = threshold_return
    output["meaningful_mfe_reached"] = meaningful.astype(np.int8)
    output["exact_net_positive"] = (
        output["execution_net_ev_12h"].to_numpy(float) > 0.0
    ).astype(np.int8)
    output["adverse_barrier_reached"] = (
        output["execution_mae_return_12h"].to_numpy(float)
        >= output["__barrier_pct__"].to_numpy(float)
    ).astype(np.int8)
    output["meaningful_but_net_nonpositive"] = (
        meaningful & (output["execution_net_ev_12h"].to_numpy(float) <= 0.0)
    ).astype(np.int8)
    output["timeout_exit"] = output["execution_exit_reason"].astype(str).str.contains(
        "timeout", case=False, regex=False
    ).astype(np.int8)
    p_hit = output["oof_clean_favorable_probability"].to_numpy(float)
    output["pred_peak_mfe_if_hit_atr"] = np.divide(
        output["pred_peak_MFE_12h_ATR"].to_numpy(float),
        p_hit,
        out=np.full(len(output), np.nan),
        where=p_hit > 1e-8,
    )
    output["path_favorable_probability"] = output[
        ["catboost_p_2", "catboost_p_3", "catboost_p_4"]
    ].sum(axis=1)
    output["path_adverse_probability"] = output[
        ["catboost_p_0", "catboost_p_1", "catboost_p_6"]
    ].sum(axis=1)
    output["path_usable_timeout_probability"] = output["catboost_p_5"]
    output["path_favorable_minus_adverse"] = (
        output["path_favorable_probability"] - output["path_adverse_probability"]
    )
    output["audit_day_utc"] = pd.to_datetime(
        output["execution_decision_utc"], utc=True, errors="raise"
    ).dt.strftime("%Y-%m-%d")
    return output.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)


def _rank_ic(score: pd.Series, target: pd.Series) -> float:
    x = pd.to_numeric(score, errors="coerce").to_numpy(float)
    y = pd.to_numeric(target, errors="coerce").to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.unique(x[finite]).size < 2 or np.unique(y[finite]).size < 2:
        return float("nan")
    return float(spearmanr(x[finite], y[finite]).statistic)


def _binary_metrics(target: pd.Series, probability: pd.Series) -> dict[str, float]:
    y = pd.to_numeric(target, errors="coerce").to_numpy(float)
    p = pd.to_numeric(probability, errors="coerce").to_numpy(float)
    finite = np.isfinite(y) & np.isfinite(p)
    y = y[finite].astype(int)
    p = np.clip(p[finite], 0.0, 1.0)
    result = {
        "rows": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "mean_probability": float(p.mean()) if len(p) else float("nan"),
        "auc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float("nan"),
        "ece_10": float("nan"),
    }
    if len(y) == 0:
        return result
    result["brier"] = float(brier_score_loss(y, p))
    bins = np.minimum((p * 10).astype(int), 9)
    result["ece_10"] = float(
        sum(
            (bins == index).mean()
            * abs(float(p[bins == index].mean()) - float(y[bins == index].mean()))
            for index in np.unique(bins)
        )
    )
    if np.unique(y).size == 2:
        result["auc"] = float(roc_auc_score(y, p))
        result["pr_auc"] = float(average_precision_score(y, p))
    return result


def _top_fraction(frame: pd.DataFrame, score: str, fraction: float = 0.10) -> dict[str, Any]:
    ordered = frame.sort_values(
        [score, "candidate_id"], ascending=[False, True], kind="stable"
    )
    count = max(1, int(math.ceil(len(ordered) * fraction)))
    selected = ordered.iloc[:count]
    net = selected["execution_net_ev_12h"]
    return {
        "selection_scope": "one_global_pool_within_reported_scope",
        "fraction": fraction,
        "rows": int(count),
        "net_ev_bps": float(net.mean() * 1e4),
        "gross_ev_bps": float(selected["execution_gross_ev_12h"].mean() * 1e4),
        "cost_bps": float(selected["execution_cost_return"].mean() * 1e4),
        "positive_net_fraction": float((net > 0.0).mean()),
        "meaningful_mfe_fraction": float(selected["meaningful_mfe_reached"].mean()),
        "adverse_barrier_fraction": float(selected["adverse_barrier_reached"].mean()),
        "mean_peak_mfe_atr": float(selected["exact_peak_mfe_atr_raw"].mean()),
        "mean_mae_atr": float(selected["exact_mae_atr"].mean()),
    }


def _deciles(frame: pd.DataFrame, head: str, score: str, scope: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = frame.copy()
    work["score_decile"] = pd.qcut(
        work[score].rank(method="first"), 10, labels=False
    ).astype(int) + 1
    table = (
        work.groupby("score_decile", sort=True)
        .agg(
            rows=("candidate_id", "size"),
            score_mean=(score, "mean"),
            net_ev_bps=("execution_net_ev_12h", lambda value: float(value.mean() * 1e4)),
            gross_ev_bps=("execution_gross_ev_12h", lambda value: float(value.mean() * 1e4)),
            positive_net_fraction=("exact_net_positive", "mean"),
            meaningful_mfe_fraction=("meaningful_mfe_reached", "mean"),
            adverse_barrier_fraction=("adverse_barrier_reached", "mean"),
            peak_mfe_atr=("exact_peak_mfe_atr_raw", "mean"),
            mae_atr=("exact_mae_atr", "mean"),
        )
        .reset_index()
    )
    table.insert(0, "scope", scope)
    table.insert(0, "head", head)
    differences = np.diff(table["net_ev_bps"].to_numpy(float))
    summary = {
        "decile_net_spearman": _rank_ic(table["score_decile"], table["net_ev_bps"]),
        "adjacent_net_violations": int((differences < 0.0).sum()),
        "top_minus_bottom_net_bps": float(
            table.iloc[-1]["net_ev_bps"] - table.iloc[0]["net_ev_bps"]
        ),
    }
    return table, summary


def head_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"head": "base_raw_alpha_score", "score": "base_prediction", "kind": "continuous"},
        {"head": "base_mapped_alpha_ev", "score": "base_alpha_ev", "kind": "continuous"},
        {"head": "residual_delta_ev", "score": "residual_delta_ev", "kind": "continuous"},
        {"head": "existing_alpha_ev", "score": "existing_alpha_ev", "kind": "continuous"},
        {
            "head": "clean_meaningful_event_probability",
            "score": "oof_clean_favorable_probability",
            "kind": "probability",
            "binary_target": "meaningful_mfe_reached",
            "target_fidelity": "exact canonical meaningful-MFE incidence",
        },
        {
            "head": "peak_mfe_unconditional",
            "score": "pred_peak_MFE_12h_ATR",
            "kind": "continuous",
        },
        {
            "head": "peak_mfe_conditional_magnitude",
            "score": "pred_peak_mfe_if_hit_atr",
            "kind": "conditional_continuous",
        },
        {
            "head": "path_favorable_probability_mass",
            "score": "path_favorable_probability",
            "kind": "proxy_probability",
            "binary_target": "meaningful_mfe_reached",
            "target_fidelity": "proxy only; exact seven-class path truth unavailable",
        },
        {
            "head": "path_adverse_probability_mass",
            "score": "path_adverse_probability",
            "kind": "proxy_probability",
            "binary_target": "adverse_barrier_reached",
            "higher_is_favorable": False,
            "target_fidelity": "proxy only; exact seven-class path truth unavailable",
        },
        {
            "head": "path_usable_timeout_probability",
            "score": "path_usable_timeout_probability",
            "kind": "proxy_probability",
            "binary_target": "timeout_exit",
            "target_fidelity": "proxy only; usable-MFE path class is not exact exit reason",
        },
        {
            "head": "path_favorable_minus_adverse",
            "score": "path_favorable_minus_adverse",
            "kind": "continuous",
        },
        {"head": "direct_execution_ev", "score": "final_direct_net_raw", "kind": "continuous"},
        {
            "head": "capture_probability",
            "score": "final_capture_probability",
            "kind": "probability",
            "binary_target": "exact_net_positive",
            "target_fidelity": "exact deployed-policy net-positive event",
        },
        {"head": "mapped_execution_ev", "score": "mapped_execution_ev", "kind": "continuous"},
    ]
    for index, name in enumerate(PATH_CLASSES):
        specs.append(
            {
                "head": f"path_probability__{name}",
                "score": f"catboost_p_{index}",
                "kind": "class_probability_without_exact_class_truth",
                "higher_is_favorable": name
                not in {
                    "immediate_adverse_path",
                    "early_mfe_full_reversal",
                    "dead_timeout",
                },
                "target_fidelity": "economic audit only; exact seven-class path truth unavailable",
            }
        )
    return specs


def evaluate(joined: pd.DataFrame) -> dict[str, pd.DataFrame | dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    decile_parts: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for spec in head_specs():
        head, score = spec["head"], spec["score"]
        higher_is_favorable = bool(spec.get("higher_is_favorable", True))
        for scope in ("pooled", *SIDES):
            local = joined if scope == "pooled" else joined.loc[joined["side_name"].eq(scope)]
            target_local = local
            if spec["kind"] == "conditional_continuous":
                target_local = local.loc[local["meaningful_mfe_reached"].eq(1)]
            deciles, monotonicity = _deciles(local, head, score, scope)
            decile_parts.append(deciles)
            top = _top_fraction(local, score)
            binary: dict[str, Any] = {}
            if "binary_target" in spec:
                binary = _binary_metrics(local[spec["binary_target"]], local[score])
            row = {
                "head": head,
                "score_column": score,
                "kind": spec["kind"],
                "higher_is_favorable": higher_is_favorable,
                "scope": scope,
                "rows": int(len(local)),
                "target_fidelity": spec.get("target_fidelity", "exact economic outcomes"),
                "rank_ic_net": _rank_ic(local[score], local["execution_net_ev_12h"]),
                "rank_ic_gross": _rank_ic(local[score], local["execution_gross_ev_12h"]),
                "rank_ic_peak_mfe_atr_raw": _rank_ic(
                    target_local[score], target_local["exact_peak_mfe_atr_raw"]
                ),
                "rank_ic_peak_mfe_atr_canonical": _rank_ic(
                    target_local[score], target_local["exact_peak_mfe_atr_canonical"]
                ),
                "rank_ic_mae_atr_favorable_orientation": _rank_ic(
                    local[score], -local["exact_mae_atr"]
                ),
                "binary_target": spec.get("binary_target"),
                "auc": binary.get("auc"),
                "pr_auc": binary.get("pr_auc"),
                "brier": binary.get("brier"),
                "ece_10": binary.get("ece_10"),
                "event_prevalence": binary.get("prevalence"),
                "mean_probability": binary.get("mean_probability"),
                **{f"top10_{key}": value for key, value in top.items()},
                **monotonicity,
            }
            metric_rows.append(row)

            for day, day_frame in local.groupby("audit_day_utc", sort=True):
                day_top = _top_fraction(day_frame, score)
                daily_rows.append(
                    {
                        "head": head,
                        "scope": scope,
                        "day_utc": day,
                        "rows": int(len(day_frame)),
                        "rank_ic_net": _rank_ic(
                            day_frame[score], day_frame["execution_net_ev_12h"]
                        ),
                        "rank_ic_peak_mfe_atr_raw": _rank_ic(
                            day_frame[score], day_frame["exact_peak_mfe_atr_raw"]
                        ),
                        "top10_net_ev_bps": day_top["net_ev_bps"],
                        "top10_positive_net_fraction": day_top["positive_net_fraction"],
                        "top10_meaningful_mfe_fraction": day_top[
                            "meaningful_mfe_fraction"
                        ],
                        "top10_adverse_barrier_fraction": day_top[
                            "adverse_barrier_fraction"
                        ],
                    }
                )

            meaningful_local = local.loc[local["meaningful_mfe_reached"].eq(1)]
            oriented_rank = local[score].rank(pct=True)
            if not higher_is_favorable:
                oriented_rank = 1.0 - oriented_rank
            incidence = _binary_metrics(
                local["meaningful_mfe_reached"],
                oriented_rank,
            )
            adverse = _binary_metrics(
                local["adverse_barrier_reached"],
                1.0 - oriented_rank,
            )
            conditional_payoff_ic = _rank_ic(
                (
                    meaningful_local[score]
                    if higher_is_favorable
                    else -meaningful_local[score]
                ),
                meaningful_local["execution_net_ev_12h"],
            )
            diagnosis = {
                "head": head,
                "scope": scope,
                "higher_is_favorable": higher_is_favorable,
                "meaningful_incidence_auc_rank_scaled": incidence["auc"],
                "conditional_on_meaningful_net_rank_ic": conditional_payoff_ic,
                "adverse_barrier_auc_inverse_rank": adverse["auc"],
                "incidence_learned_threshold": (
                    bool(incidence["auc"] >= 0.53)
                    if np.isfinite(incidence["auc"])
                    else False
                ),
                "conditional_payoff_learned_threshold": (
                    bool(conditional_payoff_ic >= 0.05)
                    if np.isfinite(conditional_payoff_ic)
                    else False
                ),
                "adverse_tail_learned_threshold": (
                    bool(adverse["auc"] >= 0.53)
                    if np.isfinite(adverse["auc"])
                    else False
                ),
            }
            diagnosis["diagnosis"] = (
                "incidence_signal_but_payoff_magnitude_failure"
                if diagnosis["incidence_learned_threshold"]
                and not diagnosis["conditional_payoff_learned_threshold"]
                else "incidence_and_payoff_signal"
                if diagnosis["incidence_learned_threshold"]
                and diagnosis["conditional_payoff_learned_threshold"]
                else "no_reliable_incidence_signal"
            )
            if not diagnosis["adverse_tail_learned_threshold"]:
                diagnosis["diagnosis"] += "__adverse_tail_not_separated"
            diagnostic_rows.append(diagnosis)

    archetype = (
        joined.groupby(["side_name", "catboost_archetype"], sort=True)
        .agg(
            rows=("candidate_id", "size"),
            net_ev_bps=("execution_net_ev_12h", lambda value: float(value.mean() * 1e4)),
            positive_net_fraction=("exact_net_positive", "mean"),
            meaningful_mfe_fraction=("meaningful_mfe_reached", "mean"),
            adverse_barrier_fraction=("adverse_barrier_reached", "mean"),
            peak_mfe_atr=("exact_peak_mfe_atr_raw", "mean"),
            mae_atr=("exact_mae_atr", "mean"),
        )
        .reset_index()
    )
    head_metrics = pd.DataFrame(metric_rows)
    daily = pd.DataFrame(daily_rows)
    stability = (
        daily.groupby(["head", "scope"], sort=True)
        .agg(
            days=("day_utc", "nunique"),
            positive_rank_ic_days=("rank_ic_net", lambda value: int((value > 0).sum())),
            positive_top10_net_days=(
                "top10_net_ev_bps",
                lambda value: int((value > 0).sum()),
            ),
            worst_day_top10_net_bps=("top10_net_ev_bps", "min"),
            best_day_top10_net_bps=("top10_net_ev_bps", "max"),
            mean_daily_rank_ic_net=("rank_ic_net", "mean"),
            std_daily_rank_ic_net=("rank_ic_net", "std"),
        )
        .reset_index()
    )
    return {
        "head_metrics": head_metrics,
        "deciles": pd.concat(decile_parts, ignore_index=True),
        "daily": daily,
        "daily_stability": stability,
        "archetype_economics": archetype,
        "diagnostics": pd.DataFrame(diagnostic_rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    bindings = {
        "packb": validate_manifest_hash(
            args.packb,
            args.packb_manifest,
            ("output", "sha256"),
            expected_schema="packb_final_refits_forward_v1",
        ),
        "preentry": validate_manifest_hash(
            args.preentry,
            args.preentry_manifest,
            ("output", "sha256"),
            expected_schema="execution_ev_forward_preentry_v1",
        ),
        "scored": validate_manifest_hash(
            args.scored,
            args.scored_manifest,
            ("outputs", "scored_population", "sha256"),
            expected_schema="execution_ev_retrospective_scored_population_v1",
        ),
        "labels": validate_manifest_hash(
            args.labels,
            args.labels_manifest,
            ("output", "sha256"),
            expected_schema="execution_ev_deployed_policy_1m_labels_v1",
        ),
        "geometry": validate_manifest_hash(
            args.geometry,
            args.geometry_manifest,
            ("outputs", "path_targets", "sha256"),
            expected_schema="execution_ev_retrospective_causal_geometry_v1",
        ),
    }
    label_manifest = json.loads(args.labels_manifest.read_text(encoding="utf-8"))
    if label_manifest.get("coverage", {}).get("overall", {}).get("coverage") != 1.0:
        raise ValueError("exact policy label coverage must be 100%")
    if label_manifest.get("exit_policy_contract", {}).get("horizon_minutes") != 720:
        raise ValueError("audit requires the exact 12h policy-label artifact")
    if label_manifest.get("accounting", {}).get("portfolio_concurrency_applied") is not False:
        raise ValueError("per-head audit requires candidate-local labels before portfolio constraints")

    joined = prepare_joined(
        pd.read_parquet(args.packb),
        pd.read_parquet(args.preentry),
        pd.read_parquet(args.scored),
        pd.read_parquet(args.labels),
        pd.read_parquet(args.geometry),
    )
    results = evaluate(joined)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = args.output_dir / "exact_head_audit_ledger.parquet"
    joined.to_parquet(ledger_path, index=False)
    outputs: dict[str, Any] = {
        "exact_head_audit_ledger": {
            "path": str(ledger_path),
            "rows": int(len(joined)),
            "sha256": sha256(ledger_path),
        }
    }
    for name, table in results.items():
        assert isinstance(table, pd.DataFrame)
        path = args.output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs[name] = {
            "path": str(path),
            "rows": int(len(table)),
            "sha256": sha256(path),
        }

    pooled = results["head_metrics"]
    assert isinstance(pooled, pd.DataFrame)
    pooled = pooled.loc[pooled["scope"].eq("pooled")].copy()
    diagnostics = results["diagnostics"]
    assert isinstance(diagnostics, pd.DataFrame)
    report = {
        "schema": SCHEMA,
        "status": "research_only_retrospective_nonpromotable",
        "promotion_eligible": False,
        "rows": int(len(joined)),
        "coverage": {
            "pooled": int(len(joined)),
            "by_side": joined["side_name"].value_counts().sort_index().to_dict(),
            "days": sorted(joined["audit_day_utc"].unique().tolist()),
        },
        "target_contract": {
            "policy_label": "exact deployed simple-policy 1m replay, 12h timeout-only ablation",
            "atr_normalization": "exact decision-time raw Wilder ATR14 / signal close from v2 geometry",
            "meaningful_mfe": (
                "execution_mfe_return_12h >= max(1.5 * "
                "__path_auxiliary_atr_fraction__, 0.015)"
            ),
            "canonical_peak_mfe_atr": (
                "clip(MFE / decision-time ATR, 0, 10), set to zero below "
                "the meaningful-MFE threshold"
            ),
            "net_positive": "execution_net_ev_12h > 0",
            "adverse_barrier": "execution_mae_return_12h >= exact __barrier_pct__",
        },
        "metric_contract": {
            "ranking": "one pooled global ordering within each reported scope; never per timestamp",
            "top10": "ceil(10% * scope rows), deterministic score-descending/candidate-id tie break",
            "daily": "same global-within-day ordering; diagnostic stability only",
            "diagnosis_thresholds": {
                "incidence_auc": 0.53,
                "conditional_payoff_rank_ic": 0.05,
                "inverse_score_adverse_barrier_auc": 0.53,
            },
        },
        "path_class_limit": {
            "exact_seven_class_truth_available": False,
            "reason": (
                "the exact 1m policy-label artifact contains payoff/MFE/MAE/exit "
                "outcomes but no materialized first-touch/path-archetype class labels"
            ),
            "treatment": (
                "individual class probabilities and predicted archetypes receive "
                "economic grouping only; favorable/adverse masses receive explicitly "
                "labelled proxy-event diagnostics, not seven-class accuracy"
            ),
        },
        "pooled_head_summary": pooled.to_dict("records"),
        "pooled_diagnostics": diagnostics.loc[
            diagnostics["scope"].eq("pooled")
        ].to_dict("records"),
        "inputs": bindings,
        "outputs": outputs,
    }
    report_path = args.output_dir / "report.json"
    _write_json(report_path, report)
    manifest = {
        "schema": SCHEMA,
        "status": "research_only_retrospective_nonpromotable",
        "promotion_eligible": False,
        "outcomes_used_for_scoring": True,
        "models_refit": False,
        "inputs": bindings,
        "outputs": {
            **outputs,
            "report": {"path": str(report_path), "sha256": sha256(report_path)},
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return report


def parser() -> argparse.ArgumentParser:
    root = Path(
        "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2"
    )
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--packb", type=Path, default=root / "packb/packb_forward_context.parquet")
    value.add_argument("--packb-manifest", type=Path, default=root / "packb/manifest.json")
    value.add_argument("--preentry", type=Path, default=root / "preentry/preentry.parquet")
    value.add_argument("--preentry-manifest", type=Path, default=root / "preentry/manifest.json")
    value.add_argument("--scored", type=Path, default=root / "scored/scored_population.parquet")
    value.add_argument("--scored-manifest", type=Path, default=root / "scored/manifest.json")
    value.add_argument(
        "--labels", type=Path, default=root / "labels_12h/execution_ev_policy_labels.parquet"
    )
    value.add_argument("--labels-manifest", type=Path, default=root / "labels_12h/manifest.json")
    value.add_argument("--geometry", type=Path, default=root / "geometry/path_targets.parquet")
    value.add_argument("--geometry-manifest", type=Path, default=root / "geometry/manifest.json")
    value.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data_perp/artifacts/july_exact_preentry_head_audit_20260730_v2"
        ),
    )
    return value


if __name__ == "__main__":
    run(parser().parse_args())
