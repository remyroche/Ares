#!/usr/bin/env python3
"""Evaluate sealed post-freeze 2026 regime/transition score arms.

This runner is intentionally only an evaluator.  It does not train, tune or
duplicate a scoring model.  It joins the authoritative v2 current-regime and
transition sidecars at an exact 2026 timestamp, then applies a separately
causal EV map to each *precomputed* score arm.  Lifecycle/ex-post phase fields
are never accepted as inputs.

The JSON configuration is deliberately explicit because the authoritative v2
sidecars do not exist yet.  See ``example_config`` for the required contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "post_freeze_2026_regime_transition_combined_evaluation_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
ARMS = ("baseline_context_free", "regime_only", "transition_only", "combined")
FORBIDDEN_INPUT_TOKENS = ("lifecycle", "ex_post", "ex-post", "phase")
SEALED_STATUS = "SEALED_POST_FREEZE_2026_AUTHORITATIVE"


class PostFreezeEvaluationError(RuntimeError):
    """Raised when frozen post-freeze evaluation provenance cannot be proven."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PostFreezeEvaluationError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, default=str, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def example_config() -> dict[str, Any]:
    """Return the source contract expected after the v2 sidecars are sealed."""
    return {
        "candidate_scores_path": "/absolute/path/precomputed_2026_arm_scores.parquet",
        "candidate_scores_manifest_path": "/absolute/path/precomputed_2026_arm_scores.manifest.json",
        "candidate_scores_manifest_sidecar_path": "/absolute/path/precomputed_2026_arm_scores.manifest.sha256",
        "regime_sidecar_path": "/absolute/path/authoritative_regime_v2.parquet",
        "regime_sidecar_manifest_path": "/absolute/path/authoritative_regime_v2.manifest.json",
        "regime_sidecar_manifest_sidecar_path": "/absolute/path/authoritative_regime_v2.manifest.sha256",
        "transition_sidecar_path": "/absolute/path/authoritative_transition_v2.parquet",
        "transition_sidecar_manifest_path": "/absolute/path/authoritative_transition_v2.manifest.json",
        "transition_sidecar_manifest_sidecar_path": "/absolute/path/authoritative_transition_v2.manifest.sha256",
        "output_dir": "/absolute/path/new_immutable_post_freeze_evaluation",
        "columns": {
            "net_ev": "execution_net_ev_12h",
            "gross_ev": "execution_gross_ev_12h",
            "cost": "execution_cost_return",
            "label_available_at": "execution_label_available_at",
            "alpha_target": "__first_touch_target_soft__",
            "baseline_context_free": "baseline_context_free_raw_score",
            "regime_only": "regime_only_raw_score",
            "transition_only": "transition_only_raw_score",
            "combined": "combined_raw_score"
        },
        "regime": {
            "timestamp_column": "__ts__",
            "available_at_column": "regime_available_utc",
            "input_columns": ["regime_state_p__0", "regime_entropy"]
        },
        "transition": {
            "timestamp_column": "__ts__",
            "available_at_column": "transition_available_utc",
            "input_columns": ["transition_active_probability", "transition_state_entropy"]
        },
        "frozen_contextual_coefficients": {
            "training_start_utc": "2022-01-01T00:00:00Z",
            "training_end_exclusive_utc": "2026-01-01T00:00:00Z",
            "arms": ["baseline_context_free", "regime_only", "transition_only", "combined"],
            "status": "FROZEN_2022_2025_CANDIDATE_CONTEXT"
        },
        "min_mapping_train_rows": 500,
        "top_fraction": 0.10
    }


def _contains_hash(value: Any, digest: str) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_hash(item, digest) for item in value.values())
    if isinstance(value, list):
        return any(_contains_hash(item, digest) for item in value)
    return isinstance(value, str) and value == digest


def _sealed_binding(path: Path, manifest_path: Path, detached_path: Path, *, role: str,
                    authoritative_v2: bool) -> dict[str, str]:
    manifest = _json(manifest_path)
    if not detached_path.is_file() or detached_path.read_text(encoding="utf-8").split()[0:1] != [_sha256(manifest_path)]:
        raise PostFreezeEvaluationError(f"{role} manifest detached checksum fails or is absent")
    digest = _sha256(path)
    if not _contains_hash(manifest, digest):
        raise PostFreezeEvaluationError(f"{role} manifest does not hash-bind {path.name}")
    schema = str(manifest.get("schema", ""))
    if authoritative_v2:
        try:
            version = int(schema.rsplit("_v", 1)[1])
        except (IndexError, ValueError) as exc:
            raise PostFreezeEvaluationError(
                f"{role} must be an authoritative v2+ sidecar, got schema {schema!r}"
            ) from exc
        if version < 2:
            raise PostFreezeEvaluationError(
                f"{role} must be an authoritative v2+ sidecar, got schema {schema!r}"
            )
    if authoritative_v2 and manifest.get("status") != SEALED_STATUS:
        raise PostFreezeEvaluationError(f"{role} is not sealed for post-freeze evaluation")
    if not authoritative_v2 and not str(manifest.get("status", "")).startswith("SEALED_POST_FREEZE_2026"):
        raise PostFreezeEvaluationError(f"{role} is not sealed for the post-freeze 2026 evaluation")
    return {"path": str(path), "sha256": digest, "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path), "schema": schema, "status": str(manifest["status"])}


def _assert_frozen_contextual_coefficients(manifest_path: Path, *, role: str,
                                            require_arms: bool) -> None:
    """Reject a source that learned contextual scoring coefficients in 2026."""
    manifest = _json(manifest_path)
    contract = manifest.get("frozen_contextual_coefficients")
    if not isinstance(contract, Mapping) or contract.get("status") != "FROZEN_2022_2025_CANDIDATE_CONTEXT":
        raise PostFreezeEvaluationError(f"{role} lacks a compatible frozen 2022--2025 candidate-context training contract")
    try:
        start = pd.to_datetime(contract["training_start_utc"], utc=True, errors="raise")
        end = pd.to_datetime(contract["training_end_exclusive_utc"], utc=True, errors="raise")
    except (KeyError, TypeError, ValueError) as exc:
        raise PostFreezeEvaluationError(f"{role} has invalid frozen contextual coefficient dates") from exc
    if start < pd.Timestamp("2022-01-01", tz="UTC") or end > pd.Timestamp("2026-01-01", tz="UTC") or start >= end:
        raise PostFreezeEvaluationError(f"{role} contextual coefficients are not strictly frozen within 2022--2025")
    if require_arms and set(contract.get("arms", ())) != set(ARMS):
        raise PostFreezeEvaluationError(f"{role} does not prove frozen pre-2026 provenance for all four score arms")


def _assert_frozen_sidecar_model(manifest_path: Path, *, role: str) -> None:
    """Require sidecar model fitting to end before the untouched 2026 test."""
    manifest = _json(manifest_path)
    contract = manifest.get("frozen_model_training")
    if isinstance(contract, Mapping):
        start_value = contract.get("training_start_utc")
        end_value = contract.get("training_end_exclusive_utc")
    else:
        start_value = manifest.get("training_start_utc")
        end_value = manifest.get("training_end_exclusive_utc")
        if (start_value is None or end_value is None) and isinstance(
            manifest.get("train_coverage"), list
        ):
            coverage = manifest["train_coverage"]
            if len(coverage) == 2:
                start_value = coverage[0]
                end_value = pd.to_datetime(coverage[1], utc=True) + pd.Timedelta(hours=1)
    try:
        start = pd.to_datetime(start_value, utc=True, errors="raise")
        end = pd.to_datetime(end_value, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise PostFreezeEvaluationError(
            f"{role} lacks valid frozen model-training dates"
        ) from exc
    if (
        start < pd.Timestamp("2022-01-01", tz="UTC")
        or end > pd.Timestamp("2026-01-01", tz="UTC")
        or start >= end
    ):
        raise PostFreezeEvaluationError(
            f"{role} model was not trained strictly within 2022--2025"
        )


def _canonical_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise PostFreezeEvaluationError(f"candidate scores lack identity columns: {missing}")
    out = frame.copy()
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="raise")
    if not out["__ts__"].dt.year.eq(2026).all():
        raise PostFreezeEvaluationError("post-freeze evaluator accepts 2026 candidate timestamps only")
    out["__symbol__"] = out["__symbol__"].astype(str)
    out["side_name"] = out["side_name"].astype(str).str.lower().str.strip()
    out["candidate_id"] = out["candidate_id"].astype(str)
    if not out.side_name.isin(("long", "short")).all() or out.loc[:, list(IDENTITY)].isna().any().any():
        raise PostFreezeEvaluationError("candidate identity is invalid")
    if out.duplicated(list(IDENTITY)).any():
        raise PostFreezeEvaluationError("candidate scores have duplicate exact identities")
    return out


def _reject_forbidden(fields: Sequence[str], *, role: str) -> None:
    bad = [field for field in fields if any(token in str(field).lower() for token in FORBIDDEN_INPUT_TOKENS)]
    if bad:
        raise PostFreezeEvaluationError(f"{role} attempts to use lifecycle/ex-post phase fields: {sorted(bad)}")


def _sidecar_timestamp_join(candidates: pd.DataFrame, sidecar: pd.DataFrame, *, role: str,
                            timestamp_column: str, available_at_column: str,
                            input_columns: Sequence[str]) -> pd.DataFrame:
    _reject_forbidden(input_columns, role=role)
    prefix = "regime_" if role == "regime" else "transition_"
    wrong_namespace = [column for column in input_columns if not str(column).startswith(prefix)]
    if wrong_namespace:
        raise PostFreezeEvaluationError(f"{role} inputs must retain their own namespace: {wrong_namespace}")
    required = [timestamp_column, available_at_column, *input_columns]
    missing = sorted(set(required).difference(sidecar.columns))
    if missing:
        raise PostFreezeEvaluationError(f"{role} sidecar lacks required fields: {missing}")
    payload = sidecar.loc[:, required].copy()
    payload[timestamp_column] = pd.to_datetime(payload[timestamp_column], utc=True, errors="raise")
    payload[available_at_column] = pd.to_datetime(payload[available_at_column], utc=True, errors="raise")
    if payload.duplicated(timestamp_column).any():
        raise PostFreezeEvaluationError(f"{role} sidecar has duplicate timestamps")
    if payload[available_at_column].gt(payload[timestamp_column]).any():
        raise PostFreezeEvaluationError(f"{role} sidecar contains context unavailable at its timestamp")
    numeric = payload.loc[:, list(input_columns)].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy(float)).any():
        raise PostFreezeEvaluationError(f"{role} sidecar contains infinite input values")
    payload.loc[:, list(input_columns)] = numeric
    marked = candidates.copy()
    marked["__row_order__"] = np.arange(len(marked), dtype=np.int64)
    joined = marked.merge(payload, how="left", left_on="__ts__", right_on=timestamp_column, validate="many_to_one", sort=False)
    joined = joined.sort_values("__row_order__", kind="stable").drop(columns="__row_order__")
    if len(joined) != len(candidates) or not joined.loc[:, list(IDENTITY)].reset_index(drop=True).equals(candidates.loc[:, list(IDENTITY)].reset_index(drop=True)):
        raise PostFreezeEvaluationError(f"{role} exact timestamp join changed candidate support")
    if joined[timestamp_column].isna().any():
        raise PostFreezeEvaluationError(f"{role} sidecar lacks exact coverage for candidate timestamps")
    if not joined["__ts__"].eq(joined[timestamp_column]).all():
        raise PostFreezeEvaluationError(f"{role} sidecar was not joined on exact timestamp")
    if joined[available_at_column].gt(joined["__ts__"]).any():
        raise PostFreezeEvaluationError(f"{role} availability violates candidate timestamp")
    # Pandas retains a single key when both sides use ``__ts__``.  Dropping it
    # would erase the candidate timestamp before the next exact sidecar join.
    return joined if timestamp_column == "__ts__" else joined.drop(columns=[timestamp_column])


def build_evaluation_panel(candidates: pd.DataFrame, regime: pd.DataFrame, transition: pd.DataFrame,
                           config: Mapping[str, Any]) -> pd.DataFrame:
    """Exact-join sidecars while retaining the two current-context roles separately."""
    columns = config["columns"]
    required = [columns[name] for name in ("net_ev", "gross_ev", "cost", "label_available_at", *ARMS)]
    if "alpha_target" in columns:
        required.append(columns["alpha_target"])
    missing = sorted(set(required).difference(candidates.columns))
    if missing:
        raise PostFreezeEvaluationError(f"candidate score source lacks required fields: {missing}")
    base = _canonical_candidates(candidates)
    base[columns["label_available_at"]] = pd.to_datetime(base[columns["label_available_at"]], utc=True, errors="raise")
    if base[columns["label_available_at"]].lt(base["__ts__"]).any():
        raise PostFreezeEvaluationError("exact outcome availability predates candidate timestamp")
    for field in required:
        if field == columns["label_available_at"]:
            continue
        values = pd.to_numeric(base[field], errors="coerce")
        if values.isna().any() or np.isinf(values.to_numpy(float)).any():
            raise PostFreezeEvaluationError(f"candidate field {field!r} must be finite on common support")
        base[field] = values
    after_regime = _sidecar_timestamp_join(base, regime, role="regime", **config["regime"])
    after_transition = _sidecar_timestamp_join(after_regime, transition, role="transition", **config["transition"])
    return after_transition


def _fit_map(train_score: pd.Series, train_ev: pd.Series):
    x, y = np.asarray(train_score, dtype=float), np.asarray(train_ev, dtype=float)
    if len(x) < 2 or np.unique(x).size < 2:
        value = float(np.mean(y)) if len(y) else 0.0
        return lambda values: np.full(len(values), value, dtype=float), {"kind": "constant", "value": value}
    model = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(x, y)
    return lambda values: np.asarray(model.predict(np.asarray(values, dtype=float)), dtype=float), {"kind": "isotonic", "increasing": bool(model.increasing_)}


def causal_map_arms(panel: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map each supplied score using only labels resolved before its month."""
    columns = config["columns"]
    min_rows = int(config.get("min_mapping_train_rows", 500))
    parts: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    months = sorted(panel["__ts__"].dt.to_period("M").unique())
    for month in months:
        start = month.start_time.tz_localize("UTC")
        end = (month + 1).start_time.tz_localize("UTC")
        evaluation = panel.loc[panel["__ts__"].ge(start) & panel["__ts__"].lt(end)].copy()
        prior = panel.loc[panel[columns["label_available_at"]].lt(start)].copy()
        if len(prior) < min_rows:
            continue
        for arm in ARMS:
            score_column = columns[arm]
            mapper, details = _fit_map(prior[score_column], prior[columns["net_ev"]])
            out = evaluation.loc[:, list(IDENTITY)].copy()
            out["arm"] = arm
            out["raw_score"] = evaluation[score_column].to_numpy(float)
            out["mapped_score"] = mapper(out["raw_score"])
            parts.append(out)
            provenance.append({"arm": arm, "evaluation_month": str(month), "evaluation_start_utc": start,
                               "mapping_train_rows": int(len(prior)), "mapping_label_available_max_utc": prior[columns["label_available_at"]].max(),
                               "mapping_details_json": json.dumps(details, sort_keys=True),
                               "contract": "precomputed score only; causal EV map trained on exact labels resolved strictly before month"})
    if not parts:
        raise PostFreezeEvaluationError("no month has enough resolved prior labels for causal mapping")
    mapped = pd.concat(parts, ignore_index=True).sort_values(["arm", "__ts__", "candidate_id"], kind="stable")
    reference = mapped.loc[mapped.arm.eq(ARMS[0]), list(IDENTITY)].reset_index(drop=True)
    for arm in ARMS[1:]:
        current = mapped.loc[mapped.arm.eq(arm), list(IDENTITY)].reset_index(drop=True)
        if not current.equals(reference):
            raise PostFreezeEvaluationError("arms do not retain identical mapped candidate rows")
    return mapped, pd.DataFrame(provenance)


def _rank_ic(score: pd.Series, target: pd.Series) -> float:
    valid = score.notna() & target.notna()
    return float(score.loc[valid].rank().corr(target.loc[valid].rank())) if valid.sum() >= 3 else float("nan")


def _monthly_global_top10(frame: pd.DataFrame) -> pd.Series:
    selected = pd.Series(False, index=frame.index)
    for _, group in frame.groupby(frame["__ts__"].dt.strftime("%Y-%m"), sort=True):
        wanted = max(1, int(math.ceil(len(group) * .10)))
        order = group.sort_values(["mapped_score", "candidate_id"], ascending=[False, True], kind="stable")
        selected.loc[order.index[:wanted]] = True
    return selected


def _period_metrics(frame: pd.DataFrame, *, arm: str, selected: pd.Series, period: str,
                    columns: Mapping[str, str]) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    key = frame["__ts__"].dt.strftime("%Y-%m") if period == "month" else frame["__ts__"].dt.to_period("W-SUN").astype(str)
    # Weekly rows are a view of the one monthly global selection; no rerank.
    for value, group in frame.assign(_period=key, _selected=selected).groupby("_period", sort=True):
        chosen = group.loc[group._selected]
        net = chosen[columns["net_ev"]]
        out.append({"arm": arm, "period_type": period, "period": value, "candidate_rows": int(len(group)),
                    "global_monthly_selected_rows": int(len(chosen)), "mean_net_ev_bps": float(net.mean() * 1e4),
                    "mean_gross_ev_bps": float(chosen[columns["gross_ev"]].mean() * 1e4),
                    "mean_cost_bps": float(chosen[columns["cost"]].mean() * 1e4),
                    "hit_rate": float(net.gt(0).mean()), "positive_fraction": float(net.gt(0).mean()),
                    "net_ev_q10_bps": float(net.quantile(.10) * 1e4), "net_ev_q50_bps": float(net.quantile(.50) * 1e4)})
    return pd.DataFrame(out)


def evaluate(panel: pd.DataFrame, mapped: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = config["columns"]
    summaries: list[dict[str, Any]] = []
    periods: list[pd.DataFrame] = []
    selections: list[pd.DataFrame] = []
    for arm in ARMS:
        scores = mapped.loc[mapped.arm.eq(arm)].copy()
        frame = panel.merge(scores, on=list(IDENTITY), how="inner", validate="one_to_one")
        if len(frame) != len(scores):
            raise PostFreezeEvaluationError(f"{arm} mapping support no longer matches exact economics")
        selected = _monthly_global_top10(frame)
        month = _period_metrics(frame, arm=arm, selected=selected, period="month", columns=columns)
        week = _period_metrics(frame, arm=arm, selected=selected, period="week", columns=columns)
        periods.extend((month, week))
        latest = month.sort_values("period", kind="stable").tail(1).iloc[0]
        summaries.append({"arm": arm, "candidate_rows": int(len(frame)), "selected_rows": int(selected.sum()),
                          "raw_net_ev_rank_ic": _rank_ic(frame.raw_score, frame[columns["net_ev"]]),
                          "mapped_net_ev_rank_ic": _rank_ic(frame.mapped_score, frame[columns["net_ev"]]),
                          "raw_alpha_rank_ic": _rank_ic(frame.raw_score, frame[columns["alpha_target"]]) if "alpha_target" in columns else float("nan"),
                          "mapped_alpha_rank_ic": _rank_ic(frame.mapped_score, frame[columns["alpha_target"]]) if "alpha_target" in columns else float("nan"),
                          "global_monthly_top10_net_ev_bps": float(frame.loc[selected, columns["net_ev"]].mean() * 1e4),
                          "global_monthly_top10_gross_ev_bps": float(frame.loc[selected, columns["gross_ev"]].mean() * 1e4),
                          "global_monthly_top10_cost_bps": float(frame.loc[selected, columns["cost"]].mean() * 1e4),
                          "global_monthly_top10_hit_rate": float(frame.loc[selected, columns["net_ev"]].gt(0).mean()),
                          "global_monthly_top10_positive_fraction": float(frame.loc[selected, columns["net_ev"]].gt(0).mean()),
                          "latest_month": latest.period, "latest_month_candidate_rows": int(latest.candidate_rows),
                          "latest_month_selected_rows": int(latest.global_monthly_selected_rows),
                          "latest_month_coverage": float(latest.global_monthly_selected_rows / latest.candidate_rows),
                          "latest_month_net_ev_bps": float(latest.mean_net_ev_bps), "latest_month_net_q10_bps": float(latest.net_ev_q10_bps),
                          "latest_month_net_q50_bps": float(latest.net_ev_q50_bps)})
        selections.append(frame.loc[selected, [*IDENTITY, "arm", "raw_score", "mapped_score", columns["net_ev"], columns["gross_ev"], columns["cost"]]].copy())
    return pd.DataFrame(summaries), pd.concat(periods, ignore_index=True), pd.concat(selections, ignore_index=True)


def build_stability_gate_ledger(summary: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    """Report, rather than optimise on, the mandatory promotion comparison gates.

    A contextual arm is never selected by this helper.  It simply makes the
    aggregate-vs-baseline, latest-month and cross-month economics requirements
    explicit in the sealed evidence.  The 9--11% coverage envelope permits the
    exact global ``ceil(10%)`` selection at realistic monthly sample sizes.
    """
    monthly = periods.loc[periods.period_type.eq("month")].copy()
    baseline = summary.set_index("arm").loc["baseline_context_free"]
    rows: list[dict[str, Any]] = []
    for item in summary.to_dict("records"):
        arm_months = monthly.loc[monthly.arm.eq(item["arm"])].sort_values("period", kind="stable")
        positive_fraction = float(arm_months.mean_net_ev_bps.gt(0).mean()) if len(arm_months) else 0.0
        coverage = float(item["latest_month_coverage"])
        rows.append({
            "arm": item["arm"],
            "aggregate_net_ev_bps": float(item["global_monthly_top10_net_ev_bps"]),
            "baseline_aggregate_net_ev_bps": float(baseline["global_monthly_top10_net_ev_bps"]),
            "delta_vs_baseline_aggregate_bps": float(item["global_monthly_top10_net_ev_bps"] - baseline["global_monthly_top10_net_ev_bps"]),
            "latest_month": item["latest_month"], "latest_month_net_ev_bps": float(item["latest_month_net_ev_bps"]),
            "baseline_latest_month_net_ev_bps": float(baseline["latest_month_net_ev_bps"]),
            "delta_vs_baseline_latest_bps": float(item["latest_month_net_ev_bps"] - baseline["latest_month_net_ev_bps"]),
            "latest_month_selected_coverage": coverage, "monthly_periods": int(len(arm_months)),
            "positive_month_fraction": positive_fraction,
            "gate_common_support": int(item["candidate_rows"]) == int(baseline["candidate_rows"]),
            "gate_aggregate_economics_positive": float(item["global_monthly_top10_net_ev_bps"]) > 0.0,
            "gate_beats_baseline_aggregate": item["arm"] == "baseline_context_free" or float(item["global_monthly_top10_net_ev_bps"]) > float(baseline["global_monthly_top10_net_ev_bps"]),
            "gate_latest_month_coverage": 0.09 <= coverage <= 0.11,
            "gate_latest_month_economics_positive": float(item["latest_month_net_ev_bps"]) > 0.0,
            "gate_beats_baseline_latest_month": item["arm"] == "baseline_context_free" or float(item["latest_month_net_ev_bps"]) > float(baseline["latest_month_net_ev_bps"]),
            "gate_stability_majority_positive_months": positive_fraction >= 0.50,
        })
    ledger = pd.DataFrame(rows)
    checks = [column for column in ledger.columns if column.startswith("gate_")]
    ledger["promotion_gate_pass"] = ledger.loc[:, checks].all(axis=1)
    return ledger


def run(config_path: Path, *, materialize: bool = True) -> dict[str, Any]:
    config = _json(Path(config_path))
    required = {"candidate_scores_path", "candidate_scores_manifest_path", "candidate_scores_manifest_sidecar_path", "regime_sidecar_path", "regime_sidecar_manifest_path", "regime_sidecar_manifest_sidecar_path", "transition_sidecar_path", "transition_sidecar_manifest_path", "transition_sidecar_manifest_sidecar_path", "output_dir", "columns", "regime", "transition"}
    missing = sorted(required.difference(config))
    if missing:
        raise PostFreezeEvaluationError(f"configuration lacks required fields: {missing}")
    paths = {name: Path(config[name]) for name in required if name.endswith("_path")}
    if float(config.get("top_fraction", .10)) != .10:
        raise PostFreezeEvaluationError("this frozen protocol requires one pooled global monthly top10 (top_fraction=0.10)")
    bindings = {
        "candidate_scores": _sealed_binding(paths["candidate_scores_path"], paths["candidate_scores_manifest_path"], paths["candidate_scores_manifest_sidecar_path"], role="candidate score source", authoritative_v2=False),
        "regime": _sealed_binding(paths["regime_sidecar_path"], paths["regime_sidecar_manifest_path"], paths["regime_sidecar_manifest_sidecar_path"], role="regime sidecar", authoritative_v2=True),
        "transition": _sealed_binding(paths["transition_sidecar_path"], paths["transition_sidecar_manifest_path"], paths["transition_sidecar_manifest_sidecar_path"], role="transition sidecar", authoritative_v2=True),
    }
    _assert_frozen_contextual_coefficients(paths["candidate_scores_manifest_path"], role="candidate score source", require_arms=True)
    _assert_frozen_sidecar_model(paths["regime_sidecar_manifest_path"], role="regime sidecar")
    _assert_frozen_sidecar_model(paths["transition_sidecar_manifest_path"], role="transition sidecar")
    if not materialize:
        return {"schema": SCHEMA, "status": "READY_BUT_NOT_MATERIALIZED", "bindings": bindings,
                "contract": "awaiting explicit materialization; evaluator fits no score model or HPO"}
    destination = Path(config["output_dir"])
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {destination}")
    candidates = pd.read_parquet(paths["candidate_scores_path"])
    regime = pd.read_parquet(paths["regime_sidecar_path"])
    transition = pd.read_parquet(paths["transition_sidecar_path"])
    panel = build_evaluation_panel(candidates, regime, transition, config)
    mapped, mapping_provenance = causal_map_arms(panel, config)
    summary, periods, selections = evaluate(panel, mapped, config)
    gates = build_stability_gate_ledger(summary, periods)
    stage = Path(tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.staging-"))
    try:
        panel.to_parquet(stage / "exact_joined_panel.parquet", index=False, compression="zstd")
        mapped.to_parquet(stage / "causal_mapped_arm_scores.parquet", index=False, compression="zstd")
        mapping_provenance.to_parquet(stage / "mapping_provenance.parquet", index=False, compression="zstd")
        summary.to_csv(stage / "metrics_summary.csv", index=False)
        periods.to_parquet(stage / "period_metrics.parquet", index=False, compression="zstd")
        selections.to_parquet(stage / "monthly_global_top10_selection.parquet", index=False, compression="zstd")
        gates.to_csv(stage / "baseline_latest_stability_gates.csv", index=False)
        files = [path for path in stage.iterdir() if path.is_file()]
        manifest = {"schema": SCHEMA, "status": "SEALED_POST_FREEZE_2026_COMBINED_EVALUATION",
                    "arms": list(ARMS), "score_models": "not fit here; four arm scores must be precomputed, hash-bound, and frozen on 2022--2025 candidate-level context only",
                    "sidecar_join": "exact candidate __ts__ == sidecar timestamp; no as-of join, fill or lifecycle/ex-post phase input",
                    "context_roles": "regime-only, transition-only and combined remain distinct score arms; current-regime and transition sidecars are never conflated",
                    "mapping": "one independent isotonic causal EV map per arm/month, trained only on exact labels available strictly before that month",
                    "selection": {"one_pooled_global_monthly_top10": True, "top_fraction": float(config.get("top_fraction", .10)), "per_timestamp": False, "per_side": False, "weekly_metrics": "attribution of monthly selection; never re-ranked"},
                    "promotion_gates": "reported only; require common support, positive aggregate and latest economics, baseline deltas, 9--11% latest global coverage, and a majority of positive months",
                    "common_support_rows": int(len(panel)), "mapped_common_support_rows": int(len(mapped) // len(ARMS)),
                    "bindings": bindings, "outputs": {path.name: _sha256(path) for path in files}}
        _write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{_sha256(stage / 'manifest.json')}  manifest.json\n", encoding="utf-8")
        os.replace(stage, destination)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--write-example-config", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="verify sealed inputs but do not write an evaluation artifact")
    args = parser.parse_args(argv)
    if args.write_example_config:
        _write_json(args.write_example_config, example_config())
        return 0
    if args.config is None:
        parser.error("--config is required unless --write-example-config is used")
    print(json.dumps(run(args.config, materialize=not args.prepare_only), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
