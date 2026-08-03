#!/usr/bin/env python3
"""Strict-OOF pairwise entry-action value relative to enter-now."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_meta import chronological_purged_splits

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
FEATURES = (
    "frozen_alpha",
    "frozen_aux_mae",
    "frozen_aux_peak",
    "frozen_aux_slope",
    "frozen_aux_time",
    "frozen_aux_turn",
    "frozen_entropy",
    "frozen_execution_ev",
    "frozen_p_0",
    "frozen_p_1",
    "frozen_p_2",
    "frozen_p_3",
    "frozen_p_4",
    "frozen_p_5",
    "frozen_p_6",
    "frozen_residual",
    "frozen_side_is_long",
    "frozen_side_is_short",
)
ACTIONS = (
    "wait_market_60m",
    "adverse_limit_180m_0.2500atr",
)
ENTER_NOW = "enter_now"


class _Constant:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.value, dtype=np.float64)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        value = float(np.clip(self.value, 0.0, 1.0))
        return np.column_stack(
            [np.full(len(x), 1.0 - value), np.full(len(x), value)]
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
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


def _lgbm_regressor(x: pd.DataFrame, y: np.ndarray, *, seed: int) -> Any:
    target = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(target)
    if int(finite.sum()) < 20 or float(np.nanstd(target[finite])) <= 1e-9:
        return _Constant(float(np.nanmean(target[finite])) if finite.any() else 0.0)
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=40,
        reg_alpha=0.12,
        reg_lambda=5.0,
        colsample_bytree=0.85,
        subsample=0.85,
        subsample_freq=1,
        max_bin=127,
        random_state=int(seed),
        deterministic=True,
        force_col_wise=True,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(x.iloc[np.flatnonzero(finite)], target[finite])
    return model


def _lgbm_classifier(x: pd.DataFrame, y: np.ndarray, *, seed: int) -> Any:
    target = np.asarray(y, dtype=np.int8)
    if len(np.unique(target)) < 2:
        return _Constant(float(target[0]) if len(target) else 0.0)
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=40,
        reg_alpha=0.12,
        reg_lambda=5.0,
        colsample_bytree=0.85,
        subsample=0.85,
        subsample_freq=1,
        max_bin=127,
        random_state=int(seed),
        deterministic=True,
        force_col_wise=True,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(x, target)
    return model


def _probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x), dtype=np.float64)[:, 1]


def _fit_pairwise(
    x: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    temperature_bps: float,
    seed: int,
) -> dict[str, Any]:
    delta_bps = (
        labels["action_realized_utility"].to_numpy(dtype=np.float64)
        - labels["enter_now_net_ev"].to_numpy(dtype=np.float64)
    ) * 10_000.0
    logit = np.clip(delta_bps / float(temperature_bps), -40.0, 40.0)
    soft = 1.0 / (1.0 + np.exp(-logit))
    positive = delta_bps > 0.0
    negative = ~positive
    fill = labels["fill_indicator"].to_numpy(dtype=np.int8)
    missed_bps = labels["missed_opportunity_ev"].to_numpy(dtype=np.float64) * 10_000.0
    return {
        "soft": _lgbm_regressor(x, soft, seed=seed),
        "positive": _lgbm_regressor(
            x.iloc[np.flatnonzero(positive)],
            np.log1p(delta_bps[positive]),
            seed=seed + 1,
        ),
        "negative": _lgbm_regressor(
            x.iloc[np.flatnonzero(negative)],
            np.log1p(-delta_bps[negative]),
            seed=seed + 2,
        ),
        "fill": _lgbm_classifier(x, fill, seed=seed + 3),
        "missed": _lgbm_regressor(
            x.iloc[np.flatnonzero(fill == 0)],
            np.log1p(missed_bps[fill == 0]),
            seed=seed + 4,
        ),
    }


def _predict_pairwise(models: Mapping[str, Any], x: pd.DataFrame) -> dict[str, np.ndarray]:
    probability = np.clip(
        np.asarray(models["soft"].predict(x), dtype=np.float64), 0.0, 1.0
    )
    positive = np.expm1(
        np.maximum(np.asarray(models["positive"].predict(x), dtype=np.float64), 0.0)
    )
    negative = np.expm1(
        np.maximum(np.asarray(models["negative"].predict(x), dtype=np.float64), 0.0)
    )
    fill = np.clip(_probability(models["fill"], x), 0.0, 1.0)
    missed = np.expm1(
        np.maximum(np.asarray(models["missed"].predict(x), dtype=np.float64), 0.0)
    )
    expected = probability * positive - (1.0 - probability) * negative
    return {
        "better_probability": probability,
        "positive_delta_bps": positive,
        "negative_delta_bps": negative,
        "expected_delta_bps": expected,
        "fill_probability": fill,
        "expected_missed_bps": missed,
    }


def _action_matrix(labels: pd.DataFrame, rows: int) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for action in (ENTER_NOW, *ACTIONS):
        part = labels.loc[labels["action_id"].eq(action)].sort_values(
            "base_position", kind="stable"
        )
        if len(part) != rows or not np.array_equal(
            part["base_position"].to_numpy(dtype=np.int64), np.arange(rows)
        ):
            raise ValueError(f"counterfactual labels do not exactly cover {action}")
        result[action] = part.reset_index(drop=True)
    return result


def _inner_lcb_residual(
    frame: pd.DataFrame,
    x: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    temperature_bps: float,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    minimum = max(200, min(1_000, len(frame) // 3))
    try:
        splits = chronological_purged_splits(
            frame,
            n_splits=2,
            min_train_size=minimum,
            decision_time_col="__decision_ts__",
            label_end_time_col="execution_label_end_utc",
            horizon_hours=12.0,
            embargo_hours=12.0,
        )
    except ValueError:
        return -np.inf, {"status": "no_inner_oof", "rows": 0}
    residuals: list[np.ndarray] = []
    for split in splits:
        models = _fit_pairwise(
            x.iloc[split.train_indices].reset_index(drop=True),
            labels.iloc[split.train_indices].reset_index(drop=True),
            temperature_bps=temperature_bps,
            seed=seed + split.fold * 10,
        )
        prediction = _predict_pairwise(models, x.iloc[split.validation_indices])
        actual = (
            labels.iloc[split.validation_indices]["action_realized_utility"].to_numpy(
                dtype=np.float64
            )
            - labels.iloc[split.validation_indices]["enter_now_net_ev"].to_numpy(
                dtype=np.float64
            )
        ) * 10_000.0
        residuals.append(actual - prediction["expected_delta_bps"])
    residual = np.concatenate(residuals)
    return float(np.quantile(residual, 0.10)), {
        "status": "inner_oof_residual_q10",
        "rows": int(len(residual)),
        "residual_q10_bps": float(np.quantile(residual, 0.10)),
        "residual_mean_bps": float(residual.mean()),
    }


def _train_outer_oof(
    frame: pd.DataFrame,
    matrix: Mapping[str, pd.DataFrame],
    *,
    temperature_bps: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    splits = chronological_purged_splits(
        frame,
        n_splits=3,
        min_train_size=500,
        min_train_group_col="side_name",
        required_train_groups=("long", "short"),
        decision_time_col="__decision_ts__",
        label_end_time_col="execution_label_end_utc",
        horizon_hours=12.0,
        embargo_hours=12.0,
    )
    output = frame.loc[:, [*IDENTITY, "__decision_ts__", "execution_label_end_utc"]].copy()
    output["pairwise_oof_fold"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["pairwise_oof_train_cutoff_utc"] = pd.Series(
        pd.NaT, index=output.index, dtype="datetime64[ns, UTC]"
    )
    audit: dict[str, Any] = {}
    x_all = frame.loc[:, FEATURES].astype(np.float32)
    if not np.isfinite(x_all.to_numpy()).all():
        raise ValueError("pairwise feature matrix contains non-finite values")
    for action in ACTIONS:
        for field in (
            "better_probability",
            "positive_delta_bps",
            "negative_delta_bps",
            "expected_delta_bps",
            "lcb_delta_bps",
            "fill_probability",
            "expected_missed_bps",
        ):
            output[f"{action}__{field}"] = np.nan
    for split in splits:
        output.loc[split.validation_indices, "pairwise_oof_fold"] = split.fold
        cutoff = pd.to_datetime(
            frame.iloc[split.train_indices]["__decision_ts__"], utc=True
        ).max()
        output.loc[
            split.validation_indices, "pairwise_oof_train_cutoff_utc"
        ] = cutoff
        fold_audit: dict[str, Any] = {
            "train_rows": int(len(split.train_indices)),
            "validation_rows": int(len(split.validation_indices)),
            "validation_start": split.validation_start,
            "train_cutoff": cutoff,
            "actions": {},
        }
        for side in ("long", "short"):
            train = split.train_indices[
                frame.iloc[split.train_indices]["side_name"].astype(str).to_numpy()
                == side
            ]
            valid = split.validation_indices[
                frame.iloc[split.validation_indices]["side_name"]
                .astype(str)
                .to_numpy()
                == side
            ]
            if len(train) < 200 or not len(valid):
                raise ValueError(f"insufficient {side} rows in pairwise outer fold")
            train_frame = frame.iloc[train].reset_index(drop=True)
            train_x = x_all.iloc[train].reset_index(drop=True)
            valid_x = x_all.iloc[valid]
            for action_index, action in enumerate(ACTIONS):
                train_labels = matrix[action].iloc[train].reset_index(drop=True)
                lcb_offset, lcb_report = _inner_lcb_residual(
                    train_frame,
                    train_x,
                    train_labels,
                    temperature_bps=temperature_bps,
                    seed=seed + split.fold * 100 + action_index * 20,
                )
                models = _fit_pairwise(
                    train_x,
                    train_labels,
                    temperature_bps=temperature_bps,
                    seed=seed + split.fold * 100 + action_index * 20 + 10,
                )
                prediction = _predict_pairwise(models, valid_x)
                for field, values in prediction.items():
                    output.loc[valid, f"{action}__{field}"] = values
                output.loc[valid, f"{action}__lcb_delta_bps"] = (
                    prediction["expected_delta_bps"] + lcb_offset
                )
                fold_audit["actions"][f"{side}__{action}"] = lcb_report
        audit[str(split.fold)] = fold_audit
    return output, audit


def _route(
    admitted: pd.DataFrame,
    matrix: Mapping[str, pd.DataFrame],
    *,
    lcb_threshold_bps: float,
    allow_limit: bool,
    min_limit_fill: float,
    max_limit_missed_bps: float,
) -> pd.DataFrame:
    result = admitted.copy()
    best_action = np.full(len(result), ENTER_NOW, dtype=object)
    best_lcb = np.full(len(result), float(lcb_threshold_bps), dtype=np.float64)
    for action in ACTIONS:
        if action.startswith("adverse_limit") and not allow_limit:
            continue
        lcb = result[f"{action}__lcb_delta_bps"].to_numpy(dtype=np.float64)
        eligible = np.isfinite(lcb) & (lcb > best_lcb)
        if action.startswith("adverse_limit"):
            eligible &= (
                result[f"{action}__fill_probability"].to_numpy(dtype=np.float64)
                >= float(min_limit_fill)
            )
            eligible &= (
                result[f"{action}__expected_missed_bps"].to_numpy(dtype=np.float64)
                <= float(max_limit_missed_bps)
            )
        best_action[eligible] = action
        best_lcb[eligible] = lcb[eligible]
    result["recommended_action"] = best_action
    result["recommended_lcb_delta_bps"] = best_lcb
    utility = np.empty(len(result), dtype=np.float64)
    fill = np.empty(len(result), dtype=np.float64)
    missed = np.empty(len(result), dtype=np.float64)
    for action in (ENTER_NOW, *ACTIONS):
        mask = best_action == action
        positions = result.loc[mask, "__base_position__"].to_numpy(dtype=np.int64)
        utility[mask] = matrix[action].iloc[positions][
            "action_realized_utility"
        ].to_numpy(dtype=np.float64)
        fill[mask] = matrix[action].iloc[positions]["fill_indicator"].to_numpy(
            dtype=np.float64
        )
        missed[mask] = matrix[action].iloc[positions][
            "missed_opportunity_ev"
        ].to_numpy(dtype=np.float64)
    result["realized_action_utility"] = utility
    result["realized_fill"] = fill
    result["realized_missed_opportunity"] = missed
    return result


def _metric_rows(frame: pd.DataFrame, *, policy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", frame)]
    scopes.extend(
        ("month", str(key), group)
        for key, group in frame.groupby(frame["__ts__"].dt.strftime("%Y-%m"))
    )
    scopes.extend(
        ("side", str(key), group) for key, group in frame.groupby("side_name")
    )
    scopes.extend(
        ("fold", str(key), group) for key, group in frame.groupby("pairwise_oof_fold")
    )
    for scope, value, group in scopes:
        action_ev = float(group["realized_action_utility"].mean() * 10_000.0)
        enter_ev = float(group["enter_now_net_ev"].mean() * 10_000.0)
        rows.append(
            {
                "policy": policy,
                "scope": scope,
                "scope_value": value,
                "rows": int(len(group)),
                "action_ev_bps": action_ev,
                "enter_now_ev_bps": enter_ev,
                "delta_vs_enter_now_bps": action_ev - enter_ev,
                "fill_rate": float(group["realized_fill"].mean()),
                "missed_opportunity_bps": float(
                    group["realized_missed_opportunity"].mean() * 10_000.0
                ),
                "enter_now_share": float(
                    group["recommended_action"].eq(ENTER_NOW).mean()
                ),
                "wait_share": float(
                    group["recommended_action"].eq(ACTIONS[0]).mean()
                ),
                "limit_share": float(
                    group["recommended_action"].eq(ACTIONS[1]).mean()
                ),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    frame = pd.read_parquet(args.handoff)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["__decision_ts__"] = pd.to_datetime(
        frame["__decision_ts__"], utc=True, errors="raise"
    )
    frame["execution_label_end_utc"] = pd.to_datetime(
        frame["execution_label_end_utc"], utc=True, errors="raise"
    )
    labels = pd.read_parquet(args.counterfactual_labels)
    matrix = _action_matrix(labels, len(frame))
    predictions, audit = _train_outer_oof(
        frame,
        matrix,
        temperature_bps=args.temperature_bps,
        seed=args.seed,
    )
    predictions.to_parquet(
        args.output_dir / "pairwise_oof_predictions.parquet",
        index=False,
        compression="zstd",
    )

    mapped = pd.read_parquet(
        args.mapped_scores,
        columns=[*IDENTITY, "causal_recent_side_isotonic_ev"],
    )
    mapped["__ts__"] = pd.to_datetime(mapped["__ts__"], utc=True, errors="raise")
    scored = mapped.merge(
        predictions.dropna(subset=["pairwise_oof_fold"]),
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    scored = scored.merge(
        frame.reset_index(names="__base_position__").loc[
            :, [*IDENTITY, "__base_position__"]
        ],
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    count = max(1, int(np.ceil(0.10 * len(scored))))
    admitted = scored.sort_values(
        ["causal_recent_side_isotonic_ev", *IDENTITY],
        ascending=[False, True, True, True, True],
        kind="stable",
    ).iloc[:count].copy()
    base_positions = admitted["__base_position__"].to_numpy(dtype=np.int64)
    admitted["enter_now_net_ev"] = matrix[ENTER_NOW].iloc[base_positions][
        "enter_now_net_ev"
    ].to_numpy(dtype=np.float64)

    policies = []
    for threshold in (0.0, 25.0, 50.0):
        policies.append(
            (
                f"wait_only_lcb{int(threshold)}",
                dict(
                    lcb_threshold_bps=threshold,
                    allow_limit=False,
                    min_limit_fill=1.0,
                    max_limit_missed_bps=0.0,
                ),
            )
        )
        policies.append(
            (
                f"wait_plus_limit_lcb{int(threshold)}",
                dict(
                    lcb_threshold_bps=threshold,
                    allow_limit=True,
                    min_limit_fill=0.80,
                    max_limit_missed_bps=15.0,
                ),
            )
        )
    metric_records = []
    decision_parts = []
    baseline = admitted.copy()
    baseline["recommended_action"] = ENTER_NOW
    baseline["recommended_lcb_delta_bps"] = 0.0
    baseline["realized_action_utility"] = baseline["enter_now_net_ev"]
    baseline["realized_fill"] = 1.0
    baseline["realized_missed_opportunity"] = 0.0
    metric_records.extend(_metric_rows(baseline, policy="enter_now"))
    baseline["policy"] = "enter_now"
    decision_parts.append(baseline)
    for policy, kwargs in policies:
        routed = _route(admitted, matrix, **kwargs)
        metric_records.extend(_metric_rows(routed, policy=policy))
        routed["policy"] = policy
        decision_parts.append(routed)
    metrics = pd.DataFrame(metric_records)
    metrics.to_csv(args.output_dir / "pairwise_policy_metrics.csv", index=False)
    pd.concat(decision_parts, ignore_index=True).to_parquet(
        args.output_dir / "pairwise_policy_decisions.parquet",
        index=False,
        compression="zstd",
    )

    head_metrics = []
    for action in ACTIONS:
        valid = predictions[f"{action}__better_probability"].notna()
        positions = np.flatnonzero(valid.to_numpy())
        action_labels = matrix[action].iloc[positions]
        actual_delta = (
            action_labels["action_realized_utility"].to_numpy(dtype=np.float64)
            - action_labels["enter_now_net_ev"].to_numpy(dtype=np.float64)
        )
        better = actual_delta > 0.0
        probability = predictions.loc[valid, f"{action}__better_probability"].to_numpy()
        head_metrics.append(
            {
                "action": action,
                "rows": int(len(positions)),
                "better_rate": float(better.mean()),
                "better_auc": (
                    float(roc_auc_score(better, probability))
                    if np.unique(better).size == 2
                    else np.nan
                ),
                "expected_delta_spearman": float(
                    pd.Series(
                        predictions.loc[
                            valid, f"{action}__expected_delta_bps"
                        ].to_numpy(dtype=np.float64)
                    ).corr(
                        pd.Series(actual_delta * 10_000.0),
                        method="spearman",
                    )
                ),
                "lcb_positive_rate": float(
                    (
                        predictions.loc[valid, f"{action}__lcb_delta_bps"] > 0.0
                    ).mean()
                ),
            }
        )
    pd.DataFrame(head_metrics).to_csv(
        args.output_dir / "pairwise_head_metrics.csv", index=False
    )
    overall = metrics.loc[metrics["scope"].eq("overall")].copy()
    baseline_ev = float(
        overall.loc[overall["policy"].eq("enter_now"), "action_ev_bps"].iloc[0]
    )
    challengers = overall.loc[~overall["policy"].eq("enter_now")].copy()
    best = challengers.sort_values(
        ["delta_vs_enter_now_bps", "policy"], ascending=[False, True]
    ).iloc[0]
    summary = {
        "schema": "pairwise_entry_timing_ablation_v1",
        "status": (
            "research_challenger_positive_aggregate"
            if float(best["delta_vs_enter_now_bps"]) > 0.0
            else "enter_now_retained"
        ),
        "contract": {
            "ranking": (
                "causal recent side-isotonic EV; one pooled global top 10%; "
                "timing never reranks admission"
            ),
            "training": (
                "side-local action-specific expanding outer OOF with 12h label "
                "purge and embargo; fixed models and soft pairwise target"
            ),
            "action_value": (
                "P(action utility > enter-now) plus conditional positive and "
                "negative delta magnitudes; limit fill and missed-opportunity "
                "heads remain separate"
            ),
            "routing": (
                "inner-OOF residual q10 lower confidence bound; enter-now "
                "mandatory fallback"
            ),
        },
        "temperature_bps": float(args.temperature_bps),
        "mapped_intersection_rows": int(len(scored)),
        "global_top10_rows": int(len(admitted)),
        "enter_now_ev_bps": baseline_ev,
        "best_challenger": best.to_dict(),
        "head_metrics": head_metrics,
        "outer_audit": audit,
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in {
                "handoff": args.handoff,
                "counterfactual_labels": args.counterfactual_labels,
                "mapped_scores": args.mapped_scores,
            }.items()
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--counterfactual-labels", type=Path, required=True)
    parser.add_argument("--mapped-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature-bps", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser


def main() -> None:
    summary = run(_parser().parse_args())
    print(json.dumps(_safe(summary), indent=2))


if __name__ == "__main__":
    main()
