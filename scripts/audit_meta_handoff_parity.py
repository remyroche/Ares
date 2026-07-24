#!/usr/bin/env python3
"""Audit historical-to-observed parity at the base-to-meta handoff.

The audit deliberately conditions meta parity on base-score parity.  A changed
base score can legitimately change rank bands, margin bands, frozen reliability
priors, and therefore the meta score; that is an upstream divergence, not a
meta-input defect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd

KEY_ALIASES = {
    "timestamp": ("__ts__", "signal_bar_ts", "timestamp"),
    "symbol": ("__symbol__", "symbol"),
    "side": ("side_name", "side"),
}
BASE_SCORE_ALIASES = ("score_base", "base_pred", "raw_prediction_score")
META_SCORE_ALIASES = ("score_meta_base_soft_label", "meta_pred")
META_FEATURE_JSON_ALIASES = ("meta_model_feature_values_json",)
BASE_HANDOFF_CONTEXT = (
    "score_base",
    "score",
    "base_score_rank_pct_train_prior",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
)


def _first_present(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    available = set(map(str, columns))
    return next((name for name in aliases if name in available), None)


def _as_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _json_values(value: Any) -> dict[str, float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    return {
        str(key): numeric
        for key, raw in value.items()
        if math.isfinite(numeric := _safe_float(raw))
    }


def feature_contract_hash(feature_names: Iterable[str]) -> str:
    payload = json.dumps(list(map(str, feature_names)), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_feature_contract(
    feature_names: Iterable[str], *, reference_columns: Iterable[str]
) -> dict[str, Any]:
    names = list(map(str, feature_names))
    reference = set(map(str, reference_columns))

    def matching(predicate: Any) -> list[str]:
        return [name for name in names if predicate(name.lower())]

    reliability = matching(lambda name: name.startswith("rel_rankband_") or name.startswith("rel_marginband_"))
    ood = matching(lambda name: "ood" in name)
    leaf = matching(lambda name: "leaf" in name)
    drift = matching(lambda name: "drift" in name or name.startswith("support_"))
    latent = matching(
        lambda name: any(
            token in name
            for token in (
                "aegmm",
                "gmm_",
                "dae_",
                "reconstruction",
                "mahal",
                "cluster_entropy",
                "latent_",
            )
        )
    )
    anchors = [name for name in BASE_HANDOFF_CONTEXT if name in names]
    return {
        "base_score_rank_margin_available_in_handoff": [
            name for name in BASE_HANDOFF_CONTEXT if name in reference
        ],
        "base_score_rank_margin_selected_by_meta_model": anchors,
        "reliability_prior_features": reliability,
        "ood_features": ood,
        "leaf_features": leaf,
        "drift_or_support_features": drift,
        "ae_gmm_latent_features": latent,
        "note": (
            "A handoff column can build train-derived reliability priors without "
            "being selected as a direct meta-model input."
        ),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_frame(
    frame: pd.DataFrame, *, role: str, feature_names: Iterable[str]
) -> pd.DataFrame:
    names: dict[str, str] = {}
    for key, aliases in KEY_ALIASES.items():
        column = _first_present(frame.columns, aliases)
        if column is None:
            raise ValueError(f"{role} frame has no {key} column; tried {aliases}")
        names[key] = column
    base_col = _first_present(frame.columns, BASE_SCORE_ALIASES)
    meta_col = _first_present(frame.columns, META_SCORE_ALIASES)
    if base_col is None or meta_col is None:
        raise ValueError(f"{role} frame is missing base or meta score")
    feature_json_col = _first_present(frame.columns, META_FEATURE_JSON_ALIASES)

    out = pd.DataFrame(index=frame.index)
    out["timestamp"] = _as_utc(frame[names["timestamp"]])
    out["symbol"] = frame[names["symbol"]].astype(str)
    out["side"] = frame[names["side"]].astype(str).str.lower()
    out["base_score"] = pd.to_numeric(frame[base_col], errors="coerce")
    out["meta_score"] = pd.to_numeric(frame[meta_col], errors="coerce")
    feature_values = (
        frame[feature_json_col].map(_json_values)
        if feature_json_col is not None
        else pd.Series([{} for _ in range(len(frame))], index=frame.index, dtype=object)
    )
    direct_features = [name for name in feature_names if name in frame.columns]
    if direct_features:
        direct = frame[direct_features].apply(pd.to_numeric, errors="coerce")
        feature_values = pd.Series(
            [
                {
                    **payload,
                    **{
                        name: float(value)
                        for name, value in direct.loc[index].items()
                        if math.isfinite(_safe_float(value))
                    },
                }
                for index, payload in feature_values.items()
            ],
            index=frame.index,
            dtype=object,
        )
    out["feature_values"] = feature_values
    out = out.dropna(subset=["timestamp", "base_score", "meta_score"])
    if out.duplicated(["timestamp", "symbol", "side"]).any():
        duplicates = int(out.duplicated(["timestamp", "symbol", "side"], keep=False).sum())
        raise ValueError(f"{role} frame has {duplicates} duplicate canonical row keys")
    return out.reset_index(drop=True)


def _feature_value(row: pd.Series, feature: str) -> float:
    values = row["feature_values"]
    if feature in values:
        return _safe_float(values[feature])
    return _safe_float(row.get(feature))


def compare_meta_handoff(
    reference: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    feature_names: list[str],
    base_tolerance: float = 1e-6,
    feature_tolerance: float = 1e-6,
    score_tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare matching rows and identify the first divergence after base parity."""

    ref = _canonical_frame(reference, role="reference", feature_names=feature_names)
    obs = _canonical_frame(observed, role="observed", feature_names=feature_names)

    merged = ref.merge(obs, on=["timestamp", "symbol", "side"], suffixes=("_ref", "_obs"))
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        base_delta = abs(float(row["base_score_ref"]) - float(row["base_score_obs"]))
        meta_delta = abs(float(row["meta_score_ref"]) - float(row["meta_score_obs"]))
        base_parity = bool(base_delta <= base_tolerance)

        ref_values = row["feature_values_ref"]
        obs_values = row["feature_values_obs"]
        reference_missing = [name for name in feature_names if name not in ref_values]
        observed_missing = [name for name in feature_names if name not in obs_values]
        observed_extra = sorted(set(obs_values).difference(feature_names))
        feature_deltas: list[tuple[str, float]] = []
        contract_mismatches: list[tuple[str, str]] = []
        for feature in feature_names:
            ref_value = _safe_float(ref_values.get(feature, np.nan))
            obs_value = _safe_float(obs_values.get(feature, np.nan))
            if math.isfinite(ref_value) and math.isfinite(obs_value):
                feature_deltas.append((feature, abs(ref_value - obs_value)))
            elif not math.isfinite(ref_value):
                contract_mismatches.append((feature, "missing_or_nonfinite_reference"))
            else:
                contract_mismatches.append((feature, "missing_or_nonfinite_observed"))
        numeric_mismatches = [
            (name, delta) for name, delta in feature_deltas if delta > feature_tolerance
        ]
        mismatch_names = {
            *(name for name, _ in contract_mismatches),
            *(name for name, _ in numeric_mismatches),
        }
        first_feature = next(
            (name for name in feature_names if name in mismatch_names), ""
        )
        if not base_parity:
            first_divergence = "upstream_base_score"
        elif first_feature:
            first_divergence = f"meta_input:{first_feature}"
        elif meta_delta > score_tolerance:
            first_divergence = "meta_score"
        else:
            first_divergence = "none"
        rows.append(
            {
                "timestamp": row["timestamp"],
                "symbol": row["symbol"],
                "side": row["side"],
                "base_score_reference": float(row["base_score_ref"]),
                "base_score_observed": float(row["base_score_obs"]),
                "base_abs_delta": base_delta,
                "base_parity": base_parity,
                "meta_score_reference": float(row["meta_score_ref"]),
                "meta_score_observed": float(row["meta_score_obs"]),
                "meta_abs_delta": meta_delta,
                "meta_parity": bool(meta_delta <= score_tolerance),
                "reference_contract_missing_count": len(reference_missing),
                "observed_contract_missing_count": len(observed_missing),
                "observed_contract_extra_count": len(observed_extra),
                "shared_feature_count": len(feature_deltas),
                "feature_mismatch_count": len(mismatch_names),
                "feature_contract_mismatch_count": len(contract_mismatches),
                "feature_max_abs_delta": max((delta for _, delta in feature_deltas), default=float("nan")),
                "first_feature_divergence": first_feature,
                "first_divergence_after_base_boundary": first_divergence,
            }
        )

    detail = pd.DataFrame(rows)
    base_equal = detail[detail["base_parity"]] if not detail.empty else detail
    post_base_failures = base_equal[
        base_equal["first_divergence_after_base_boundary"].ne("none")
    ] if not base_equal.empty else base_equal
    summary = {
        "reference_rows": int(len(ref)),
        "observed_rows": int(len(obs)),
        "overlap_rows": int(len(detail)),
        "base_parity_rows": int(len(base_equal)),
        "base_mismatch_rows": int((~detail["base_parity"]).sum()) if not detail.empty else 0,
        "base_max_abs_delta": float(detail["base_abs_delta"].max()) if not detail.empty else None,
        "base_mean_abs_delta": float(detail["base_abs_delta"].mean()) if not detail.empty else None,
        "post_base_meta_input_or_score_mismatch_rows": int(len(post_base_failures)),
        "post_base_meta_score_max_abs_delta": float(base_equal["meta_abs_delta"].max()) if not base_equal.empty else None,
        "post_base_meta_score_mean_abs_delta": float(base_equal["meta_abs_delta"].mean()) if not base_equal.empty else None,
        "post_base_feature_max_abs_delta": (
            float(base_equal["feature_max_abs_delta"].dropna().max())
            if not base_equal.empty and base_equal["feature_max_abs_delta"].notna().any()
            else None
        ),
        "first_post_base_divergence": (
            str(post_base_failures.iloc[0]["first_divergence_after_base_boundary"])
            if not post_base_failures.empty
            else "none"
        ),
        "feature_contract_count": int(len(feature_names)),
        "feature_contract_hash": feature_contract_hash(feature_names),
        "observed_contract_missing_max": (
            int(detail["observed_contract_missing_count"].max()) if not detail.empty else None
        ),
        "reference_contract_missing_max": (
            int(detail["reference_contract_missing_count"].max())
            if not detail.empty
            else None
        ),
        "observed_contract_extra_max": (
            int(detail["observed_contract_extra_count"].max()) if not detail.empty else None
        ),
        "base_tolerance": float(base_tolerance),
        "feature_tolerance": float(feature_tolerance),
        "score_tolerance": float(score_tolerance),
    }
    return detail, summary


def rescore_observed_meta_matrix(
    observed: pd.DataFrame,
    *,
    feature_names: list[str],
    model: Any,
    score_tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-score captured meta matrices with the frozen model artifact."""

    canonical = _canonical_frame(
        observed, role="observed", feature_names=feature_names
    )
    complete = canonical[canonical["feature_values"].map(
        lambda values: all(name in values for name in feature_names)
    )].copy()
    if complete.empty:
        return complete, {
            "observed_rows": int(len(canonical)),
            "complete_matrix_rows": 0,
            "rescored_rows": 0,
            "max_abs_delta": None,
            "mean_abs_delta": None,
            "mismatch_rows": 0,
        }
    matrix = pd.DataFrame(
        [[row[name] for name in feature_names] for row in complete["feature_values"]],
        columns=feature_names,
        dtype=np.float64,
    )
    predicted = np.asarray(model.predict(matrix), dtype=np.float64).reshape(-1)
    complete["meta_score_rescored"] = predicted
    complete["meta_rescore_abs_delta"] = np.abs(
        predicted - complete["meta_score"].to_numpy(dtype=np.float64)
    )
    return complete, {
        "observed_rows": int(len(canonical)),
        "complete_matrix_rows": int(len(complete)),
        "rescored_rows": int(len(complete)),
        "max_abs_delta": float(complete["meta_rescore_abs_delta"].max()),
        "mean_abs_delta": float(complete["meta_rescore_abs_delta"].mean()),
        "mismatch_rows": int((complete["meta_rescore_abs_delta"] > score_tolerance).sum()),
        "score_tolerance": float(score_tolerance),
    }


def audit_prior_provenance(prior_path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = json.loads(prior_path.read_text(encoding="utf-8"))
    source = payload.get("source", {})
    raw_source_path = Path(str(source.get("scored_ledger", "")))
    source_path = raw_source_path if raw_source_path.is_absolute() else repo_root / raw_source_path
    cutoff = pd.Timestamp(source.get("train_end_exclusive"))
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    selected_col = str(payload.get("selected_col", "")).strip()
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    quoted_path = str(source_path).replace("'", "''")
    columns = set(
        con.execute(f"DESCRIBE SELECT * FROM read_parquet('{quoted_path}')").df()["column_name"].astype(str)
    )
    ts_col = "__ts__" if "__ts__" in columns else "timestamp"
    symbol_col = "__symbol__" if "__symbol__" in columns else "symbol"
    side_col = "side_name" if "side_name" in columns else "side"
    selected_clause = (
        f" AND COALESCE(CAST({selected_col} AS BOOLEAN), FALSE)"
        if selected_col and selected_col in columns
        else ""
    )
    cutoff_sql = cutoff.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z")
    ts_expr = f"CAST({ts_col} AS TIMESTAMPTZ)"
    query = f"""
        SELECT
            count(*) FILTER (WHERE {ts_expr} < TIMESTAMPTZ '{cutoff_sql}'{selected_clause}) AS fit_rows,
            count(DISTINCT struct_pack(ts := {ts_expr}, symbol := {symbol_col}, side := {side_col}))
                FILTER (WHERE {ts_expr} < TIMESTAMPTZ '{cutoff_sql}'{selected_clause}) AS fit_unique_rows,
            min({ts_expr}) FILTER (WHERE {ts_expr} < TIMESTAMPTZ '{cutoff_sql}'{selected_clause}) AS fit_min_ts,
            max({ts_expr}) FILTER (WHERE {ts_expr} < TIMESTAMPTZ '{cutoff_sql}'{selected_clause}) AS fit_max_ts,
            count(*) FILTER (WHERE {ts_expr} >= TIMESTAMPTZ '{cutoff_sql}') AS excluded_future_rows
        FROM read_parquet('{quoted_path}')
    """
    metrics = con.execute(query).df().iloc[0].to_dict()
    con.close()
    fit_rows = int(metrics["fit_rows"])
    fit_unique_rows = int(metrics["fit_unique_rows"])
    max_ts = pd.Timestamp(metrics["fit_max_ts"])
    if max_ts.tzinfo is None:
        max_ts = max_ts.tz_localize("UTC")
    else:
        max_ts = max_ts.tz_convert("UTC")
    return {
        "prior_path": str(prior_path),
        "prior_sha256": file_sha256(prior_path),
        "schema": payload.get("schema"),
        "payload_rows": int(payload.get("rows", -1)),
        "source_path": str(source_path),
        "source_selected_col": selected_col or None,
        "train_end_exclusive": cutoff.isoformat(),
        "fit_rows": fit_rows,
        "fit_unique_rows": fit_unique_rows,
        "fit_min_ts": pd.Timestamp(metrics["fit_min_ts"]).isoformat(),
        "fit_max_ts": max_ts.isoformat(),
        "excluded_future_rows": int(metrics["excluded_future_rows"]),
        "payload_row_count_matches": bool(fit_rows == int(payload.get("rows", -1))),
        "canonical_keys_unique": bool(fit_rows == fit_unique_rows),
        "causal_cutoff_pass": bool(max_ts < cutoff),
        "exact_groups_only": bool(payload.get("exact_groups_only", False)),
        "group_count": int(len(payload.get("groups", {}))),
        "side_archetype_prior_count": int(len(payload.get("side_arch_priors", {}))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-predictions", required=True, type=Path)
    parser.add_argument("--observed-ledger", required=True, type=Path)
    parser.add_argument("--columns-json", required=True, type=Path)
    parser.add_argument("--prior-json", required=True, type=Path)
    parser.add_argument("--model-joblib", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--base-tolerance", type=float, default=1e-6)
    parser.add_argument("--feature-tolerance", type=float, default=1e-6)
    parser.add_argument("--score-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = json.loads(args.columns_json.read_text(encoding="utf-8"))
    feature_names = list(map(str, contract["feature_names"]))
    computed_hash = feature_contract_hash(feature_names)
    expected_hash = str(contract.get("feature_contract_hash", ""))
    if expected_hash and computed_hash != expected_hash:
        raise ValueError(f"Feature contract hash mismatch: {computed_hash} != {expected_hash}")

    reference = pd.read_parquet(args.reference_predictions)
    observed = pd.read_parquet(args.observed_ledger)
    if args.timestamp:
        cutoff = pd.Timestamp(args.timestamp)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        ref_ts = _as_utc(reference[_first_present(reference.columns, KEY_ALIASES["timestamp"])])
        obs_ts = _as_utc(observed[_first_present(observed.columns, KEY_ALIASES["timestamp"])])
        reference = reference.loc[ref_ts.eq(cutoff)].copy()
        observed = observed.loc[obs_ts.eq(cutoff)].copy()

    detail, parity = compare_meta_handoff(
        reference,
        observed,
        feature_names=feature_names,
        base_tolerance=args.base_tolerance,
        feature_tolerance=args.feature_tolerance,
        score_tolerance=args.score_tolerance,
    )
    prior = audit_prior_provenance(args.prior_json, repo_root=Path.cwd())
    model_rescore: dict[str, Any] | None = None
    reference_model_rescore: dict[str, Any] | None = None
    if args.model_joblib is not None:
        import joblib

        model = joblib.load(args.model_joblib)
        model_features = list(map(str, getattr(model, "feature_name_", feature_names)))
        if model_features != feature_names:
            raise ValueError("Frozen model feature order does not match columns.json")
        _, model_rescore = rescore_observed_meta_matrix(
            observed,
            feature_names=feature_names,
            model=model,
            score_tolerance=args.score_tolerance,
        )
        model_rescore["model_joblib"] = str(args.model_joblib)
        model_rescore["model_joblib_sha256"] = file_sha256(args.model_joblib)
        _, reference_model_rescore = rescore_observed_meta_matrix(
            reference,
            feature_names=feature_names,
            model=model,
            score_tolerance=args.score_tolerance,
        )
        reference_model_rescore["model_joblib"] = str(args.model_joblib)
        reference_model_rescore["model_joblib_sha256"] = file_sha256(
            args.model_joblib
        )
    summary = {
        "schema": "meta_handoff_parity_audit_v1",
        "reference_predictions": str(args.reference_predictions),
        "observed_ledger": str(args.observed_ledger),
        "columns_json": str(args.columns_json),
        "feature_contract_expected_hash": expected_hash,
        "feature_contract_audit": audit_feature_contract(
            feature_names, reference_columns=reference.columns
        ),
        "parity": parity,
        "frozen_model_rescore": model_rescore,
        "frozen_reference_model_rescore": reference_model_rescore,
        "prior_provenance": prior,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_parquet(args.output_dir / "meta_handoff_parity_rows.parquet", index=False)
    (args.output_dir / "meta_handoff_parity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
