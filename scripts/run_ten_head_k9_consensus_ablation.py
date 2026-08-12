#!/usr/bin/env python3
"""Matched optimized-policy ten-head and consensus-stage K9 ablation.

The conditional-usefulness funnel is reused as an immutable per-head contract:
target, query, raw feature subset, weighting, and ranker parameters are not
reselected here.  Every head is refit on the current optimized-policy residual
around the current D2 base anchor.

K9 is deliberately frozen once on the target-free October--December 2024
market surface.  The same centres, ordering, scale, and temperature transform
every training and held row in every fold.  This prevents cluster positions
from changing meaning between consensus models.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import mutual_info_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_causal_geometry_k9_c3_ablation as c3  # noqa: E402
import scripts.run_strict_r3_self_distillation as sd  # noqa: E402
import scripts.run_ten_head_conditional_usefulness_funnel as ten  # noqa: E402
from scripts.run_ten_head_c3_full_stack_replay import (  # noqa: E402
    FROZEN_HEADS,
    _fit_and_score_head,
    _head_seed_for_month,
    _load_frozen_configs,
)


POLICY_OUTCOMES = ROOT / (
    "data_perp/artifacts/strict_r3_policy_outcomes_hourly_backfill_long_"
    "2025_jul2026_20260810_v1/candidate_policy_outcomes.parquet"
)
BASE_D2 = ROOT / (
    "data_perp/artifacts/strict_r3_self_distillation_base_d2_top20_boost15_"
    "long_2024_jul2026_20260810_v1/base_oof_predictions.parquet"
)
BASE_ARM = "D2_top20_boost1.5"
K9_FIT_START = pd.Timestamp("2024-10-01", tz="UTC")
K9_FIT_END = pd.Timestamp("2025-01-01", tz="UTC")
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
MODES = ("conditional_none", "conditional_summary", "conditional_raw9", "conditional_cmi3")
K9_MEMBERSHIPS = tuple(
    f"k09__cluster_{cluster:02d}__membership" for cluster in range(c3.K)
)
K9_SUMMARIES = ("k9_entropy", "k9_top2_margin", "k9_ood_distance")
SEED = 20260810


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _month_add(value: pd.Timestamp, offset: int) -> pd.Timestamp:
    return (value.tz_convert(None).to_period("M") + offset).to_timestamp().tz_localize("UTC")


def _load_frozen_raw_k9(fields: Sequence[str]) -> tuple[c3.RawK9Bundle, dict[str, Any]]:
    columns = ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", *fields]
    available = set(pq.ParquetFile(sd.SOURCE_PANEL).schema.names)
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"target-free K9 source lacks fields: {missing[:12]}")
    warmup = pd.read_parquet(
        sd.SOURCE_PANEL,
        columns=columns,
        filters=[
            ("__decision_ts__", ">=", K9_FIT_START),
            ("__decision_ts__", "<", K9_FIT_END),
        ],
    )
    warmup["__ts__"] = pd.to_datetime(warmup["__ts__"], utc=True)
    warmup["__decision_ts__"] = pd.to_datetime(warmup["__decision_ts__"], utc=True)
    bundle, audit = c3._fit_raw_k9(
        warmup,
        fields,
        bundle_id="consensus_fixed_raw_k9_octdec2024_v1",
        fit_start=K9_FIT_START,
        fit_end=K9_FIT_END,
        source_kind="target_free_point_in_time_market_surface",
        previous=None,
    )
    audit.update(
        {
            "definition_is_frozen_across_folds": True,
            "fit_population_rows_before_cap": int(len(warmup)),
            "fit_uses_policy_or_h12_outcomes": False,
        }
    )
    return bundle, audit


def _attach_k9(frame: pd.DataFrame, bundle: c3.RawK9Bundle) -> pd.DataFrame:
    k9 = bundle.transform(frame).reset_index(drop=True)
    output = frame.reset_index(drop=True).copy()
    for column in (*K9_MEMBERSHIPS, *K9_SUMMARIES):
        output[column] = k9[column].to_numpy(np.float32)
    return output


def _quantile_bins(values: pd.Series, bins: int = 10) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    ranked = numeric.rank(method="average", pct=True).fillna(0.5).to_numpy(float)
    return np.minimum((ranked * bins).astype(np.int32), bins - 1)


def conditional_membership_mi(
    frame: pd.DataFrame,
    *,
    target_column: str = "__target__",
    context_column: str = "prequential_base_rank42",
    top_k: int = 3,
) -> pd.DataFrame:
    """Binned train-only I(K9 membership; grade | base-rank decile)."""
    if top_k < 1 or top_k > len(K9_MEMBERSHIPS):
        raise ValueError("top_k must select between one and nine memberships")
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(float)
    context = _quantile_bins(frame[context_column], bins=10)
    rows: list[dict[str, Any]] = []
    for field in K9_MEMBERSHIPS:
        feature = _quantile_bins(frame[field], bins=10)
        score = 0.0
        support = 0
        for bucket in range(10):
            mask = (context == bucket) & np.isfinite(target)
            count = int(mask.sum())
            if count < 50 or np.unique(target[mask]).size < 2:
                continue
            score += count * float(mutual_info_score(target[mask].astype(np.int32), feature[mask]))
            support += count
        rows.append(
            {
                "field": field,
                "conditional_mi": float(score / max(support, 1)),
                "support_rows": int(support),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["conditional_mi", "field"], ascending=[False, True], kind="stable",
    ).reset_index(drop=True)
    result["selected"] = False
    result.loc[: top_k - 1, "selected"] = True
    return result


def _mode_fields(mode: str, cmi: pd.DataFrame) -> tuple[str, ...]:
    if mode == "conditional_none":
        return ()
    if mode == "conditional_summary":
        return K9_SUMMARIES
    if mode == "conditional_raw9":
        return K9_MEMBERSHIPS
    if mode == "conditional_cmi3":
        selected = tuple(cmi.loc[cmi["selected"], "field"].astype(str))
        if len(selected) != 3:
            raise AssertionError("train-only K9 CMI selector did not return three fields")
        return (*K9_SUMMARIES, *selected)
    raise ValueError(f"unknown K9 consensus mode: {mode}")


def _prepare_ledger(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fields: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = sd._read_residual_ledger(start, end, fields)
    # The repaired override population begins in 2025.  Earlier warm-up rows
    # retain their already-prequential ledger policy/base outputs and are used
    # only to establish causal committee history for January 2025.
    override_coverage_start = max(start, pd.Timestamp("2025-01-01", tz="UTC"))
    frame, policy_audit = sd._apply_policy_outcome_overrides(
        frame,
        path=POLICY_OUTCOMES,
        evaluation_start=override_coverage_start,
        evaluation_end=end,
    )
    frame, base_audit = sd._apply_base_prediction_overrides(
        frame,
        path=BASE_D2,
        arm=BASE_ARM,
        evaluation_start=override_coverage_start,
        evaluation_end=end,
    )
    frame["month"] = frame["__decision_ts__"].dt.strftime("%Y-%m")
    return frame, {
        **policy_audit, **base_audit,
        "pre_override_warmup_start": start,
        "override_coverage_start": override_coverage_start,
        "pre_override_warmup_uses_existing_prequential_ledger": bool(
            start < override_coverage_start
        ),
    }


def _fit_month(
    frame: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    held_end: pd.Timestamp,
    fields: Sequence[str],
    bundle: c3.RawK9Bundle,
    configs: dict[str, ten.HeadConfig],
    modes: Sequence[str],
    max_train_rows: int,
    month_position: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    held = frame.loc[
        frame["__decision_ts__"].ge(cutoff) & frame["__decision_ts__"].lt(held_end)
    ].copy()
    earlier = frame.loc[
        frame["__decision_ts__"].lt(cutoff)
        & frame["policy_label_available_ts"].lt(cutoff)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["prequential_base_rank42"], errors="coerce"))
    ].copy()
    if cutoff < K9_FIT_END and any(mode != "conditional_none" for mode in modes):
        raise ValueError("frozen consensus K9 cannot score a fold before 2025-01-01")
    if held.empty or len(earlier) < 1_000:
        raise ValueError(f"{cutoff:%Y-%m}: insufficient residual train/held support")
    policy_map = sd.fit_policy_net_map(
        earlier["prequential_base_rank42"], earlier["policy_net_bps"],
    )
    earlier["base_anchor_bps"] = policy_map.predict(
        earlier["prequential_base_rank42"],
    )
    held["base_anchor_bps"] = policy_map.predict(held["prequential_base_rank42"])
    earlier["base_rank"] = earlier["prequential_base_rank42"]
    held["base_rank"] = held["prequential_base_rank42"]
    # One paired complete-query sample is shared by every arm. Individual
    # head sampling is then query-aware but cannot change the source support.
    model_frame = sd._cap_complete_queries(earlier, max_train_rows)
    model_frame = _attach_k9(model_frame, bundle)
    held = _attach_k9(held, bundle)
    model_frame["__target__"] = ten.residual_grade(
        model_frame["policy_net_bps"].to_numpy(float)
        - model_frame["base_anchor_bps"].to_numpy(float),
        ten.TARGETS["resid_default_150_50"],
    )
    cmi = conditional_membership_mi(model_frame)
    cmi["held_month"] = cutoff.strftime("%Y-%m")
    cmi["fit_max_label_available_ts"] = model_frame["policy_label_available_ts"].max()
    base_columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "prequential_p_adverse", "prequential_p_weak", "prequential_p_clear",
        "prequential_base_score", "prequential_base_rank42",
    ]
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for mode in modes:
        additions = _mode_fields(mode, cmi)
        ranks: list[np.ndarray] = []
        for head_index, spec in enumerate(ten.HEAD_SPECS):
            source_config = configs[spec.name]
            config = replace(
                source_config,
                fields=[*source_config.fields, *additions],
            )
            rank, audit = _fit_and_score_head(
                model_frame,
                held,
                config,
                seed=_head_seed_for_month(spec.name, cutoff.strftime("%Y-%m"), month_position),
                max_train_rows=max_train_rows,
            )
            ranks.append(rank)
            audits.append(
                {
                    "arm": mode,
                    "held_month": cutoff.strftime("%Y-%m"),
                    "head": spec.name,
                    "frozen_target": config.target_name,
                    "frozen_query": config.query_name,
                    "raw_field_count": len(source_config.fields),
                    "k9_fields": list(additions),
                    "k9_field_count": len(additions),
                    "fit_max_label_available_ts": model_frame["policy_label_available_ts"].max(),
                    "held_outcomes_consumed": False,
                    **audit,
                }
            )
        output = held.loc[:, base_columns].copy()
        output["prequential_base_anchor_bps"] = held["base_anchor_bps"].to_numpy(np.float32)
        output["prequential_consensus_rank"] = np.nanmedian(
            np.column_stack(ranks), axis=1,
        ).astype(np.float32)
        for spec, rank in zip(ten.HEAD_SPECS, ranks, strict=True):
            output[f"conditional_head__{spec.name}__rank"] = np.asarray(
                rank, dtype=np.float32,
            )
        output["prequential_residual_rank"] = output["prequential_consensus_rank"]
        output["prequential_upstream"] = (
            0.75 * output["prequential_base_rank42"]
            + 0.25 * output["prequential_consensus_rank"]
        ).astype(np.float32)
        output["stack_is_prequential"] = True
        output["arm"] = mode
        output["held_month"] = cutoff.strftime("%Y-%m")
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(audits), cmi


def _attach_incumbent(
    predictions: pd.DataFrame,
    *,
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    incumbent = pd.read_parquet(path)
    if "arm" in incumbent:
        incumbent = incumbent.loc[incumbent["arm"].astype(str).eq("D0")].copy()
    incumbent["__decision_ts__"] = pd.to_datetime(incumbent["__decision_ts__"], utc=True)
    incumbent = incumbent.loc[
        incumbent["__decision_ts__"].ge(start) & incumbent["__decision_ts__"].lt(end)
    ].copy()
    incumbent["arm"] = "incumbent_ordinary"
    required = set(predictions.columns)
    missing = sorted(required - set(incumbent.columns))
    optional_head_fields = [
        field for field in missing if field.startswith("conditional_head__")
    ]
    for field in optional_head_fields:
        incumbent[field] = np.nan
    missing = sorted(required - set(incumbent.columns))
    if missing:
        raise ValueError(f"incumbent score override lacks fields: {missing}")
    incumbent = incumbent.loc[:, predictions.columns]
    if incumbent["candidate_id"].duplicated().any():
        raise ValueError("incumbent score override is duplicated")
    expected = predictions.loc[predictions["arm"].eq(predictions["arm"].iloc[0]), "candidate_id"]
    if set(incumbent["candidate_id"]) != set(expected):
        raise ValueError("incumbent and conditional arms do not share candidate identities")
    return pd.concat([incumbent, predictions], ignore_index=True)


def _metrics(
    predictions: pd.DataFrame,
    policy_path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy = pd.read_parquet(
        policy_path,
        columns=[
            "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        ],
        filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
    )
    frame = predictions.merge(policy, on="candidate_id", how="left", validate="many_to_one")
    global_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for arm, local in frame.groupby("arm", sort=True):
        local = local.loc[np.isfinite(local["prequential_upstream"])].copy()
        top2_month_values: list[float] = []
        for tail in TAILS:
            selected = local.nlargest(
                max(1, int(math.ceil(tail * len(local)))),
                "prequential_upstream",
                keep="first",
            )
            valid = selected.loc[
                selected["policy_path_valid"].fillna(False).astype(bool)
                & np.isfinite(pd.to_numeric(selected["policy_net_bps"], errors="coerce"))
            ]
            global_rows.append(
                {
                    "arm": arm,
                    "tail": tail,
                    "population_rows": int(len(local)),
                    "selected_score_rows": int(len(selected)),
                    "valid_outcomes": int(len(valid)),
                    "outcome_coverage": float(len(valid) / max(len(selected), 1)),
                    "gross_bps_per_trade": float(valid["policy_gross_bps"].mean()),
                    "net_bps_per_trade": float(valid["policy_net_bps"].mean()),
                    "positive_rate": float(valid["policy_net_bps"].gt(0).mean()),
                }
            )
            for month, block in selected.groupby("held_month", sort=True):
                valid_month = block.loc[
                    block["policy_path_valid"].fillna(False).astype(bool)
                    & np.isfinite(pd.to_numeric(block["policy_net_bps"], errors="coerce"))
                ]
                value = float(valid_month["policy_net_bps"].mean())
                monthly_rows.append(
                    {
                        "arm": arm,
                        "tail": tail,
                        "month": month,
                        "selected_score_rows": int(len(block)),
                        "valid_outcomes": int(len(valid_month)),
                        "outcome_coverage": float(len(valid_month) / max(len(block), 1)),
                        "net_bps_per_trade": value,
                        "positive_rate": float(valid_month["policy_net_bps"].gt(0).mean()),
                    }
                )
                if tail == 0.02 and np.isfinite(value):
                    top2_month_values.append(value)
        values = np.asarray(top2_month_values, dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        worst = float(values.min())
        pooled = next(
            row["net_bps_per_trade"]
            for row in global_rows
            if row["arm"] == arm and row["tail"] == 0.02
        )
        stability_rows.append(
            {
                "arm": arm,
                "top2_pooled_net_bps": float(pooled),
                "top2_month_median_net_bps": median,
                "top2_month_mad_bps": mad,
                "top2_worst_month_net_bps": worst,
                "top2_positive_months": int((values > 0).sum()),
                "top2_months": int(len(values)),
                "top2_portability_score": float(
                    median - 0.5 * mad - max(0.0, -worst)
                ),
            }
        )
    return pd.DataFrame(global_rows), pd.DataFrame(monthly_rows), pd.DataFrame(stability_rows)


def run(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    modes: Sequence[str],
    max_train_rows: int,
    incumbent: Path | None,
    out: Path,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise ValueError(f"unknown modes: {unknown}")
    fields = sd._base_fields()
    configs = _load_frozen_configs(FROZEN_HEADS)
    bundle, geometry_audit = _load_frozen_raw_k9(fields)
    frame, source_audit = _prepare_ledger(start=start, end=end, fields=fields)
    months = list(pd.date_range(start, end, freq="MS", inclusive="left"))
    predictions: list[pd.DataFrame] = []
    fit_audits: list[pd.DataFrame] = []
    cmi_audits: list[pd.DataFrame] = []
    for position, cutoff in enumerate(months):
        held_end = min(cutoff + pd.offsets.MonthBegin(1), end)
        print(json.dumps({"event": "fold_start", "month": cutoff.strftime("%Y-%m")}), flush=True)
        output, audit, cmi = _fit_month(
            frame,
            cutoff=cutoff,
            held_end=held_end,
            fields=fields,
            bundle=bundle,
            configs=configs,
            modes=modes,
            max_train_rows=max_train_rows,
            month_position=position,
        )
        predictions.append(output)
        fit_audits.append(audit)
        cmi_audits.append(cmi)
        print(json.dumps({"event": "fold_complete", "month": cutoff.strftime("%Y-%m")}), flush=True)
    result = pd.concat(predictions, ignore_index=True)
    if incumbent is not None:
        result = _attach_incumbent(result, path=incumbent, start=start, end=end)
    global_metrics, monthly_metrics, stability = _metrics(
        result,
        POLICY_OUTCOMES,
        start=start,
        end=end,
    )
    candidates = stability.loc[stability["arm"].isin(modes)].sort_values(
        ["top2_portability_score", "top2_pooled_net_bps"],
        ascending=[False, False],
        kind="stable",
    )
    winner = candidates.iloc[0].to_dict()
    out.mkdir(parents=True, exist_ok=False)
    result.to_parquet(out / "residual_oof_score_overrides.parquet", index=False, compression="zstd")
    pd.concat(fit_audits, ignore_index=True).to_parquet(out / "head_fit_audit.parquet", index=False)
    pd.concat(cmi_audits, ignore_index=True).to_parquet(out / "k9_conditional_mi_audit.parquet", index=False)
    global_metrics.to_parquet(out / "upstream_global_tail_metrics.parquet", index=False)
    monthly_metrics.to_parquet(out / "upstream_monthly_global_tail_contribution.parquet", index=False)
    stability.to_parquet(out / "upstream_top2_stability.parquet", index=False)
    pd.DataFrame([geometry_audit]).to_parquet(out / "frozen_k9_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_ten_head_k9_consensus_ablation_v1",
        "side": "long",
        "evaluation_start": start.isoformat(),
        "evaluation_end_exclusive": end.isoformat(),
        "modes": list(modes),
        "incumbent": None if incumbent is None else str(incumbent),
        "policy_outcomes": str(POLICY_OUTCOMES),
        "base_predictions": str(BASE_D2),
        "base_arm": BASE_ARM,
        "frozen_head_configs": str(FROZEN_HEADS),
        "frozen_head_semantics": "target/query/raw-fields/weights/params reused; refit on current optimized-policy residual",
        "max_train_rows": int(max_train_rows),
        "k9_fit_start": K9_FIT_START.isoformat(),
        "k9_fit_end_exclusive": K9_FIT_END.isoformat(),
        "k9_bundle_sha256": geometry_audit["bundle_sha256"],
        "k9_refit_monthly": False,
        "k9_target_free": True,
        "selected_candidate_on_this_evaluation": winner,
        "candidate_selection_period": {
            "start": start.isoformat(),
            "end_exclusive": end.isoformat(),
        },
        "selection_rule": "Top-2 monthly portability, then pooled Top-2 net EV",
        "severe_target_changed": False,
        "severe_target": "outside this upstream ablation; downstream remains exact H12 TP6/SL4 net <= -200 bps shadow only",
        **source_audit,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"event": "complete", "out": str(out), "winner": winner}, default=str), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", required=True)
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--max-train-rows", type=int, default=80_000)
    parser.add_argument("--incumbent", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(
        start=_utc(args.evaluation_start),
        end=_utc(args.evaluation_end),
        modes=tuple(args.modes),
        max_train_rows=int(args.max_train_rows),
        incumbent=args.incumbent,
        out=args.out,
    )


if __name__ == "__main__":
    main()
