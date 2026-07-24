#!/usr/bin/env python3
"""Attribute meta parameter regressions on a frozen feature and OOS contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.run_s52_train_meta_regime_handoff_smoke import (
    KEY_COLUMNS,
    LEDGER_CONTEXT_COLUMNS,
    OUTCOME_COLUMNS,
    _candidate_column,
    _projected_handoff_columns_for_selected,
    run_smoke,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _materialize_diagnostic_sample(
    *,
    handoff_path: Path,
    ledger_path: Path,
    selected_features: list[str],
    frontier: str,
    out_dir: Path,
    train_rows: int,
    eval_month: str,
) -> tuple[Path, Path]:
    sample_dir = out_dir / "sampled_source"
    sample_handoff = sample_dir / "train_meta_regime_handoff.parquet"
    sample_ledger = sample_dir / "s52_trailing_regime_scored_ledger.parquet"
    if sample_handoff.exists() and sample_ledger.exists():
        return sample_handoff, sample_ledger
    sample_dir.mkdir(parents=True, exist_ok=True)
    frontier_col = _candidate_column(frontier)
    handoff_schema = pq.read_schema(handoff_path)
    ledger_schema = pq.read_schema(ledger_path)
    hcols = set(_projected_handoff_columns_for_selected(handoff_path, selected_features) or [])
    hcols.update(KEY_COLUMNS)
    hcols.update({"score", frontier_col})
    hcols = [c for c in handoff_schema.names if c in hcols]
    lcols = list(KEY_COLUMNS) + ["month", "score", frontier_col]
    lcols += [c for c in OUTCOME_COLUMNS if c not in lcols]
    lcols += [c for c in LEDGER_CONTEXT_COLUMNS if c not in lcols]
    lcols = [c for c in ledger_schema.names if c in set(lcols)]

    # The source files are row-aligned. Keep all evaluation rows and a causal,
    # evenly spread sample of prior rows without materializing either full file.
    eval_start = pd.Timestamp(f"{eval_month}-01", tz="UTC")
    total_prior = 0
    for batch in pq.ParquetFile(handoff_path).iter_batches(
        batch_size=131_072, columns=["__ts__", frontier_col]
    ):
        frame = batch.to_pandas()
        ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        selected = pd.to_numeric(frame[frontier_col], errors="coerce").fillna(0).gt(0.5)
        total_prior += int((selected & ts.lt(eval_start)).sum())
    stride = max(1, int(np.ceil(total_prior / max(int(train_rows), 1))))
    hwriter = None
    lwriter = None
    prior_seen = 0
    kept_rows = 0
    hiter = pq.ParquetFile(handoff_path).iter_batches(batch_size=65_536, columns=hcols)
    liter = pq.ParquetFile(ledger_path).iter_batches(batch_size=65_536, columns=lcols)
    try:
        for hbatch, lbatch in zip(hiter, liter, strict=True):
            if hbatch.num_rows != lbatch.num_rows:
                raise RuntimeError("Handoff and ledger batches are not row aligned")
            hframe = hbatch.to_pandas()
            lframe = lbatch.to_pandas()
            ts = pd.to_datetime(hframe["__ts__"], utc=True, errors="coerce")
            selected = pd.to_numeric(hframe[frontier_col], errors="coerce").fillna(0).gt(0.5)
            prior = selected & ts.lt(eval_start)
            eval_rows = selected & ts.ge(eval_start)
            local_prior = np.flatnonzero(prior.to_numpy())
            keep_prior = np.zeros(len(hframe), dtype=bool)
            if len(local_prior):
                ordinals = prior_seen + np.arange(len(local_prior), dtype=np.int64)
                keep_prior[local_prior] = (ordinals % stride) == 0
                prior_seen += len(local_prior)
            keep = keep_prior | eval_rows.to_numpy(dtype=bool)
            if not bool(keep.any()):
                continue
            ht = pa.Table.from_pandas(hframe.loc[keep], preserve_index=False)
            lt = pa.Table.from_pandas(lframe.loc[keep], preserve_index=False)
            if hwriter is None:
                hwriter = pq.ParquetWriter(sample_handoff, ht.schema, compression="zstd")
                lwriter = pq.ParquetWriter(sample_ledger, lt.schema, compression="zstd")
            hwriter.write_table(ht)
            lwriter.write_table(lt)
            kept_rows += int(keep.sum())
    finally:
        if hwriter is not None:
            hwriter.close()
        if lwriter is not None:
            lwriter.close()
    if kept_rows <= 0:
        raise RuntimeError("Diagnostic source sampling produced no rows")
    return sample_handoff, sample_ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-manifest",
        type=Path,
        default=Path(
            "data_perp/artifacts/20260713_meta_fullthroughjul10_old55_oldparams_ablation/manifest.json"
        ),
    )
    parser.add_argument(
        "--new-params",
        type=Path,
        default=Path(
            "data_perp/artifacts/20260713_meta_fullthroughjul10_hpo150_work/s52_train_meta_hpo_best.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_perp/artifacts/20260713_meta_old55_parameter_oaat_july"),
    )
    parser.add_argument("--eval-month", default="2026-07")
    parser.add_argument("--train-rows", type=int, default=300_000)
    parser.add_argument(
        "--max-new-arms",
        type=int,
        default=0,
        help="Stop after this many newly fitted arms; zero runs all remaining arms.",
    )
    args = parser.parse_args()

    old_manifest = _load_json(args.old_manifest)
    new_payload = _load_json(args.new_params)
    old_params = dict(old_manifest["classifier_params"])
    new_params = dict(new_payload["classifier_params"])
    selected = list(old_manifest["selected_feature_union"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_handoff, sample_ledger = _materialize_diagnostic_sample(
        handoff_path=Path(old_manifest["handoff_path"]),
        ledger_path=Path(old_manifest["ledger_path"]),
        selected_features=selected,
        frontier=str(old_manifest["frontier"]),
        out_dir=args.out_dir,
        train_rows=int(args.train_rows),
        eval_month=str(args.eval_month),
    )

    arms: list[tuple[str, dict]] = [("old_baseline", dict(old_params))]
    for name in old_params:
        if name in new_params and old_params[name] != new_params[name]:
            params = dict(old_params)
            params[name] = new_params[name]
            arms.append((f"new_{name}", params))
    exposure = dict(old_params)
    exposure["n_estimators"] = new_params["n_estimators"]
    exposure["learning_rate"] = new_params["learning_rate"]
    arms.append(("new_boosting_exposure", exposure))
    arms.append(("all_new", dict(new_params)))

    rows: list[dict] = []
    new_arms = 0
    for idx, (name, params) in enumerate(arms):
        arm_dir = args.out_dir / name
        manifest_path = arm_dir / "manifest.json"
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
        else:
            manifest = run_smoke(
                handoff_dir=sample_handoff.parent,
                handoff_path=sample_handoff,
                ledger_path=sample_ledger,
                out_dir=arm_dir,
                frontier=str(old_manifest["frontier"]),
                seed=20260713,
                train_scope=str(old_manifest["train_scope"]),
                enable_base_prior_features=bool(
                    old_manifest.get("enable_base_prior_features", False)
                ),
                enable_reliability_features=bool(
                    old_manifest.get("enable_reliability_features", False)
                ),
                enable_support_drift_features=bool(
                    old_manifest.get("enable_support_drift_features", False)
                ),
                enable_hit_surprise_features=bool(
                    old_manifest.get("enable_hit_surprise_features", False)
                ),
                enable_path_order_heads=False,
                enable_path_order_blends=False,
                feature_selection_top_n=0,
                feature_selection_target=str(
                    old_manifest.get("feature_selection_target", "ev_frontier")
                ),
                feature_selection_method="lgbm_pipeline",
                max_oos_model_age_days=int(
                    old_manifest.get("max_oos_model_age_days", 30)
                ),
                validation_scope="all",
                model_train_max_rows=0,
                model_params={"classifier": params, "regressor": params},
                model_profile_name=f"old55_oaat_{name}",
                meta_head_mode="single_base_soft_label",
                minimal_artifacts=True,
                fixed_selected_features=selected,
                handoff_columns=_projected_handoff_columns_for_selected(
                    sample_handoff, selected
                ),
                eval_months=[str(args.eval_month)],
                side_specific_single_head=False,
            )
            new_arms += 1
        summary = pd.read_csv(
            arm_dir / "s52_train_meta_regime_handoff_smoke_summary.csv"
        )
        metric = summary.loc[summary["selector"].eq("meta_base_soft_label")]
        if metric.empty:
            raise RuntimeError(f"Missing meta_base_soft_label metrics for {name}")
        rec = metric.iloc[0].to_dict()
        rows.append(
            {
                "arm": name,
                "mean_top10_ev_after_1pct": rec.get("mean_keep010_ev_after_1pct"),
                "worst_top10_ev_after_1pct": rec.get("worst_keep010_ev_after_1pct"),
                "top10_clean_precision": rec.get("mean_keep010_clean_exec_precision"),
                "top10_bad_mae": rec.get("mean_keep010_full_path_bad_mae"),
                "top10_timeout": rec.get("mean_keep010_timeout"),
                **{key: value for key, value in params.items()},
            }
        )
        pd.DataFrame(rows).to_csv(args.out_dir / "parameter_oaat_metrics.csv", index=False)
        if int(args.max_new_arms) > 0 and new_arms >= int(args.max_new_arms):
            break

    result = pd.DataFrame(rows)
    baseline_rows = result.loc[result["arm"].eq("old_baseline"), "mean_top10_ev_after_1pct"]
    if not baseline_rows.empty:
        baseline = float(baseline_rows.iloc[0])
        result["delta_top10_ev_vs_old"] = result["mean_top10_ev_after_1pct"] - baseline
    result.to_csv(args.out_dir / "parameter_oaat_metrics.csv", index=False)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "meta_parameter_oaat_ablation_v1",
                "feature_contract": str(args.old_manifest),
                "new_params_source": str(args.new_params),
                "evaluation_month": str(args.eval_month),
                "diagnostic_train_rows": int(args.train_rows),
                "diagnostic_sampling": "beginning_middle_end_time_spread",
                "side_model_contract": "legacy_global_for_exact_parameter_attribution",
                "cost_contract": "ev_after_1pct",
                "output": str(args.out_dir / "parameter_oaat_metrics.csv"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
