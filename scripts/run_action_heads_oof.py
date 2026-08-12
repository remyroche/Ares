#!/usr/bin/env python3
"""Train reusable side-local pre-entry action heads with chronological OOF.

This runner is deliberately separate from execution-EV ranking and policy
replay.  It consumes a causal pre-entry feature table and an outcome-only
target table, enforces label maturity at every held-month boundary, and emits
OOF action scores plus diagnostics.  It supports the existing wait-10 target
pack and the exact fixed-horizon action target pack; post-entry heads are
rejected until causal prefix-state features are explicitly materialised.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
SEED = 20260807
DEFAULT_FEATURES = Path(
    "/Users/remyroche/Documents/Ares/data_perp/artifacts/2023apr_2024_current_policy_wait10_action_20260730_v1/preentry_features.parquet"
)
DEFAULT_LABELS = Path(
    "/Users/remyroche/Documents/Ares/data_perp/artifacts/2023apr_2024_current_policy_wait10_action_20260730_v1/action_labels.parquet"
)
DEFAULT_OUT = Path(
    "/Users/remyroche/Documents/Ares/data_perp/artifacts/action_heads_oof_20260807_v1"
)


class ContractError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _availability_column(labels: pd.DataFrame) -> str:
    candidates = (
        "label_available_at_12h_utc",
        "execution_label_end_utc",
        "execution_label_end_utc_target",
        "label_available_utc",
    )
    for col in candidates:
        if col in labels.columns:
            return col
    raise ContractError(
        "labels need an explicit label availability column; refusing to infer maturity"
    )


def _safe_feature_columns(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_roles_path: Path | None = None,
) -> tuple[list[str], str]:
    """Return admissible fields and the contract mode used.

    The older action packs do not carry a role registry, so they retain the
    conservative name-based filter below.  A frozen handoff *does* carry an
    explicit ``model_inputs`` list; when supplied, that list is authoritative
    and the runner refuses to silently broaden it with other handoff columns.
    """
    label_names = set(labels.columns) - set(IDENTITY)
    forbidden_tokens = (
        "target_",
        "label_",
        "wait10_",
        "wait_delta",
        "wait_better",
        "enter_now_",
        "execution_",
        "path_",
        "future_",
        "mfe",
        "mae",
        "gross",
        "net",
        "cost",
        "slope",
        "giveback",
        "underwater",
    )
    if feature_roles_path is not None:
        if not feature_roles_path.exists():
            raise ContractError(f"feature-role registry not found: {feature_roles_path}")
        try:
            roles = json.loads(feature_roles_path.read_text())
        except Exception as exc:  # pragma: no cover - contract failure path
            raise ContractError(f"cannot read feature-role registry: {exc}") from exc
        explicit = roles.get("model_inputs") if isinstance(roles, dict) else None
        if not isinstance(explicit, list) or not explicit or not all(
            isinstance(col, str) and col for col in explicit
        ):
            raise ContractError(
                "feature-role registry must contain a non-empty string model_inputs list"
            )
        fields = list(dict.fromkeys(explicit))
        missing = sorted(set(fields) - set(features.columns))
        if missing:
            raise ContractError(
                f"explicit feature contract contains fields absent from features: {missing[:10]}"
            )
        non_numeric = [col for col in fields if not pd.api.types.is_numeric_dtype(features[col])]
        if non_numeric:
            raise ContractError(f"explicit model inputs are not numeric: {non_numeric[:10]}")
        overlap = sorted(set(fields) & label_names)
        if overlap:
            raise ContractError(f"target leakage in explicit feature contract: {overlap[:10]}")
        # Do not reapply the name-based heuristic to an explicit frozen role
        # registry.  Some authorized causal support/regime fields necessarily
        # contain words such as ``net`` or ``execution`` in their provenance
        # name.  The registry's ``model_inputs``/``target_only_never_model_inputs``
        # split is the authoritative contract; label-column overlap is still a
        # hard failure above.
        return fields, "explicit_model_inputs"

    fields = []
    for col in features.columns:
        if col in IDENTITY or col in label_names:
            continue
        lowered = str(col).lower()
        if any(token in lowered for token in forbidden_tokens):
            continue
        if pd.api.types.is_numeric_dtype(features[col]):
            fields.append(str(col))
    if not fields:
        raise ContractError("no admissible numeric pre-entry feature fields")
    overlap = sorted(set(fields) & label_names)
    if overlap:
        raise ContractError(f"target leakage in feature contract: {overlap[:10]}")
    return fields, "name_filtered_numeric"


def _load(
    features_path: Path,
    labels_path: Path,
    feature_roles_path: Path | None = None,
) -> tuple[pd.DataFrame, list[str], str, str]:
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)
    for name, frame in (("features", features), ("labels", labels)):
        missing = [c for c in IDENTITY if c not in frame.columns]
        if missing:
            raise ContractError(f"{name} missing identity columns: {missing}")
        if frame[list(IDENTITY)].duplicated().any():
            raise ContractError(f"duplicate candidate identities in {name}")
    features["__ts__"] = pd.to_datetime(features["__ts__"], utc=True)
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
    availability = _availability_column(labels)
    labels[availability] = pd.to_datetime(labels[availability], utc=True)
    fields, feature_contract_mode = _safe_feature_columns(
        features, labels, feature_roles_path
    )
    joined = features[list(IDENTITY) + fields].merge(
        labels, on=list(IDENTITY), how="inner", validate="one_to_one", suffixes=("", "__label")
    )
    if len(joined) != len(features) or len(joined) != len(labels):
        raise ContractError(
            f"feature/label identity mismatch: features={len(features)} labels={len(labels)} joined={len(joined)}"
        )
    joined["__label_available_ts__"] = joined[availability]
    joined["__month__"] = joined["__ts__"].dt.strftime("%Y-%m")
    joined.attrs["availability_column"] = availability
    return joined, fields, availability, feature_contract_mode


def _target_specs(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    if "wait_delta" in frame.columns:
        specs.update(
            {
                "wait_better": {
                    "kind": "classifier",
                    "target": "wait_delta",
                    "utility": "wait_delta",
                    "positive": "gt_zero",
                },
                "wait_delta": {
                    "kind": "regressor",
                    "target": "wait_delta",
                    "utility": "wait_delta",
                },
            }
        )
    if "target_fixed_12h_net_return" in frame.columns:
        specs.update(
            {
                "trade_positive_12h": {
                    "kind": "classifier",
                    "target": "target_fixed_12h_net_return",
                    "utility": "target_fixed_12h_net_return",
                    "positive": "gt_zero",
                },
                "cost_clear_25bps": {
                    "kind": "classifier",
                    "target": "target_cost_clear_opportunity_25bps",
                    "utility": "target_fixed_12h_net_return",
                    "positive": "native",
                },
            }
        )
    if not specs:
        raise ContractError("no supported action targets found")
    return specs


def _target_values(frame: pd.DataFrame, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    target = pd.to_numeric(frame[spec["target"]], errors="coerce").to_numpy(float)
    utility = pd.to_numeric(frame[spec["utility"]], errors="coerce").to_numpy(float)
    valid = np.isfinite(target) & np.isfinite(utility) & frame["__label_available_ts__"].notna().to_numpy()
    if spec["kind"] == "classifier":
        if spec.get("positive") == "native":
            y = target >= 0.5
        else:
            y = target > 0.0
        return y.astype(np.int8), valid
    return target.astype(np.float32), valid


def _fit_model(kind: str, x: pd.DataFrame, y: np.ndarray, seed: int) -> Any:
    common = {
        "n_estimators": 160,
        "learning_rate": 0.03,
        "max_depth": 4,
        "num_leaves": 16,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": 2,
    }
    if kind == "classifier":
        model = lgb.LGBMClassifier(objective="binary", **common)
    else:
        model = lgb.LGBMRegressor(objective="regression_l2", **common)
    model.fit(x, y)
    return model


def _predict(model: Any, kind: str, x: pd.DataFrame) -> np.ndarray:
    if kind == "classifier":
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float32)
    return np.asarray(model.predict(x), dtype=np.float32)


def _numeric(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    return frame[fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _fold_metric(test: pd.DataFrame, score: np.ndarray, utility: np.ndarray, name: str, side: str, month: str) -> dict[str, Any]:
    valid = np.isfinite(score) & np.isfinite(utility)
    if valid.sum() == 0:
        return {"head": name, "side": side, "held_month": month, "rows": 0}
    s = score[valid]
    u = utility[valid]
    order = np.argsort(-s, kind="stable")
    n = max(1, int(np.ceil(0.10 * len(order))))
    top = u[order[:n]]
    rank_ic = pd.Series(s).corr(pd.Series(u), method="spearman") if len(s) > 3 else np.nan
    return {
        "head": name,
        "side": side,
        "held_month": month,
        "rows": int(len(s)),
        "utility_pool": float(np.mean(u)),
        "top10_utility": float(np.mean(top)),
        "rank_ic": float(rank_ic) if pd.notna(rank_ic) else None,
        "score_mean": float(np.mean(s)),
    }


def run(
    features_path: Path,
    labels_path: Path,
    output: Path,
    min_train_rows: int = 500,
    feature_roles_path: Path | None = None,
) -> dict[str, Any]:
    frame, fields, availability, feature_contract_mode = _load(
        features_path, labels_path, feature_roles_path
    )
    specs = _target_specs(frame)
    months = sorted(frame["__month__"].unique().tolist())
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for i, held_month in enumerate(months):
        prior = months[:i]
        if not prior:
            continue
        held_start = pd.Timestamp(f"{held_month}-01", tz="UTC")
        prior_frame = frame.loc[frame["__month__"].isin(prior)].copy()
        train = prior_frame.loc[prior_frame["__label_available_ts__"] < held_start].copy()
        test = frame.loc[frame["__month__"].eq(held_month)].copy()
        fold_record = {
            "held_month": held_month,
            "train_months": prior,
            "train_rows_before_maturity": int(len(prior_frame)),
            "train_rows_after_maturity": int(len(train)),
            "held_rows": int(len(test)),
            "held_start": held_start.isoformat(),
            "availability_column": availability,
            "label_maturity_enforced": True,
            "no_held_outcomes_in_fit": True,
        }
        fold_rows.append(fold_record)
        if len(train) < min_train_rows:
            continue
        for side in sorted(frame.side_name.astype(str).unique()):
            tr = train.loc[train.side_name.astype(str).eq(side)].copy()
            te = test.loc[test.side_name.astype(str).eq(side)].copy()
            if len(tr) < min_train_rows or len(te) == 0:
                continue
            xtr = _numeric(tr, fields)
            xte = _numeric(te, fields)
            for head, spec in specs.items():
                y, valid = _target_values(tr, spec)
                if int(valid.sum()) < min_train_rows or np.unique(y[valid]).size < (2 if spec["kind"] == "classifier" else 1):
                    continue
                model = _fit_model(spec["kind"], xtr.loc[valid], y[valid], SEED + i)
                score = _predict(model, spec["kind"], xte)
                utility = pd.to_numeric(te[spec["utility"]], errors="coerce").to_numpy(float)
                pred = te[list(IDENTITY)].copy()
                pred["held_month"] = held_month
                pred["head"] = head
                pred["side"] = side
                pred["action_score"] = score
                pred["action_utility"] = utility
                prediction_parts.append(pred)
                metric_rows.append(_fold_metric(te, score, utility, head, side, held_month))
    if not prediction_parts:
        raise ContractError("no OOF folds produced predictions")
    predictions = pd.concat(prediction_parts, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "action_oof_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(metric_rows).to_csv(output / "action_oof_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_json(output / "fold_audit.json", orient="records", indent=2)
    coverage = pd.DataFrame({"feature": fields, "finite_fraction": [_numeric(frame, [f])[f].notna().mean() for f in fields]})
    coverage.to_csv(output / "feature_coverage.csv", index=False)
    manifest = {
        "schema": "action_heads_oof_v1",
        "status": "RESEARCH_ONLY_NO_PROMOTION_NO_POLICY_REPLAY",
        "features_path": str(features_path),
        "labels_path": str(labels_path),
        "features_sha256": _sha256(features_path),
        "labels_sha256": _sha256(labels_path),
        "feature_roles_path": str(feature_roles_path) if feature_roles_path else None,
        "feature_roles_sha256": _sha256(feature_roles_path) if feature_roles_path else None,
        "feature_contract_mode": feature_contract_mode,
        "rows": int(len(frame)),
        "oof_rows": int(len(predictions)),
        "feature_count": len(fields),
        "feature_columns": fields,
        "target_specs": specs,
        "availability_column": availability,
        "label_maturity_rule": "label_available_ts < held_month_first_signal_ts",
        "sides": sorted(frame.side_name.astype(str).unique().tolist()),
        "months": months,
        "heads": sorted(predictions["head"].astype(str).unique().tolist()),
        "promotion_gates": [
            "positive net by side and month",
            "positive clustered lower bound",
            "improvement versus enter-now and fixed controls",
            "no action target or future path field in inference features",
        ],
    }
    _write_json(output / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-train-rows", type=int, default=500)
    p.add_argument(
        "--feature-roles",
        type=Path,
        default=None,
        help="optional frozen role registry; its model_inputs list is authoritative",
    )
    args = p.parse_args()
    print(
        json.dumps(
            _json_safe(
                run(
                    args.features,
                    args.labels,
                    args.output,
                    args.min_train_rows,
                    args.feature_roles,
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
