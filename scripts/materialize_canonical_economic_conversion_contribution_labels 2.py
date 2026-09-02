#!/usr/bin/env python3
"""Materialize reconciled upside/downside conversion-transition contributions.

The existing conditional favorable-payoff target is noisy when few candidates
finish net-positive.  This immutable outcome-only artifact adds unconditional
positive and loss contributions plus a soft net-positive rate.  The raw
positive-minus-loss pair must reconcile exactly to the existing direct-net
label; robust variants are sensitivity labels and make no such accounting
claim.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.materialize_canonical_economic_conversion_transition_labels import (
        DECILES,
        HORIZONS,
        POST_WINDOW_PUBLICATION_LAG,
        REQUIRED_COLUMNS,
        ROOT,
        SOURCE,
        _global_hour_completeness,
        _normalise_source,
        _safe,
        _source_hashes,
        add_frozen_causal_score_deciles,
        robust_mean,
        sha256,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from materialize_canonical_economic_conversion_transition_labels import (
        DECILES,
        HORIZONS,
        POST_WINDOW_PUBLICATION_LAG,
        REQUIRED_COLUMNS,
        ROOT,
        SOURCE,
        _global_hour_completeness,
        _normalise_source,
        _safe,
        _source_hashes,
        add_frozen_causal_score_deciles,
        robust_mean,
        sha256,
    )


BASE_LABEL_SOURCE = (
    ROOT
    / "data_perp/artifacts/canonical_economic_conversion_transition_labels_20260729_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/canonical_economic_conversion_contribution_labels_20260729_v1"
)
SCHEMA = "canonical_economic_conversion_contribution_labels_v1"
KEY = ("cohort_anchor_utc", "side_name", "frozen_base_score_decile", "horizon_hours")


def _base_label_hashes(root: Path) -> dict[str, str]:
    paths = (
        root / "cohort_transition_labels.parquet",
        root / "manifest.json",
        root / "manifest.sha256",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"base transition-label artifact is incomplete: {missing}")
    expected = (root / "manifest.sha256").read_text(encoding="utf-8").split()[0]
    if expected != sha256(root / "manifest.json"):
        raise ValueError("base transition-label manifest checksum mismatch")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "canonical_economic_conversion_transition_labels_v1":
        raise ValueError("unexpected base transition-label schema")
    return {str(path): sha256(path) for path in paths}


def _window_contributions(rows: pd.DataFrame) -> dict[str, Any]:
    support = int(len(rows))
    result: dict[str, Any] = {
        "candidate_support": support,
        "net_positive_support": 0,
        "net_nonpositive_support": 0,
        "direct_mean_net": float("nan"),
        "positive_net_contribution": float("nan"),
        "loss_net_contribution": float("nan"),
        "net_positive_rate": float("nan"),
        "positive_net_contribution_robust_mean": float("nan"),
        "loss_net_contribution_robust_mean": float("nan"),
        "robust_component_net": float("nan"),
        "raw_components_reconcile_direct_mean_flag": False,
        "target_available_utc": pd.NaT,
    }
    if not support:
        return result
    net = rows["execution_net_ev_12h"].to_numpy(dtype=float)
    positive = np.maximum(net, 0.0)
    loss = np.maximum(-net, 0.0)
    positive_mean = float(positive.mean())
    loss_mean = float(loss.mean())
    direct = float(net.mean())
    positive_robust = robust_mean(positive)
    loss_robust = robust_mean(loss)
    result.update(
        {
            "net_positive_support": int((net > 0.0).sum()),
            "net_nonpositive_support": int((net <= 0.0).sum()),
            "direct_mean_net": direct,
            "positive_net_contribution": positive_mean,
            "loss_net_contribution": loss_mean,
            "net_positive_rate": float((net > 0.0).mean()),
            "positive_net_contribution_robust_mean": positive_robust,
            "loss_net_contribution_robust_mean": loss_robust,
            "robust_component_net": positive_robust - loss_robust,
            "raw_components_reconcile_direct_mean_flag": bool(
                np.isclose(positive_mean - loss_mean, direct, rtol=0.0, atol=1e-12)
            ),
            "target_available_utc": rows["execution_label_end_utc"].max()
            + POST_WINDOW_PUBLICATION_LAG,
        }
    )
    return result


def materialize_contribution_labels(source: pd.DataFrame) -> pd.DataFrame:
    rows = add_frozen_causal_score_deciles(_normalise_source(source))
    global_hours = pd.DatetimeIndex(sorted(rows["__ts__"].unique()))
    if not len(global_hours):
        raise ValueError("canonical source is empty")
    records: list[dict[str, Any]] = []
    for horizon_hours, horizon_role in HORIZONS:
        horizon = pd.Timedelta(hours=horizon_hours)
        for side in ("long", "short"):
            for decile in DECILES:
                cohort = rows.loc[
                    rows["side_name"].eq(side)
                    & rows["frozen_base_score_decile"].eq(decile)
                ].sort_values("__ts__", kind="stable")
                stamps = cohort["__ts__"].to_numpy(dtype="datetime64[ns]")
                for anchor in global_hours:
                    before_left = np.searchsorted(
                        stamps, (anchor - horizon).to_datetime64(), side="left"
                    )
                    before_right = np.searchsorted(
                        stamps, anchor.to_datetime64(), side="left"
                    )
                    after_left = before_right
                    after_right = np.searchsorted(
                        stamps, (anchor + horizon).to_datetime64(), side="left"
                    )
                    before = _window_contributions(
                        cohort.iloc[before_left:before_right]
                    )
                    after = _window_contributions(
                        cohort.iloc[after_left:after_right]
                    )
                    before_hours, before_complete, after_hours, after_complete = (
                        _global_hour_completeness(
                            global_hours, anchor, horizon_hours
                        )
                    )
                    record: dict[str, Any] = {
                        "cohort_anchor_utc": anchor,
                        "side_name": side,
                        "frozen_base_score_decile": decile,
                        "horizon_hours": horizon_hours,
                        "horizon_role": horizon_role,
                        "before_global_hour_support": before_hours,
                        "after_global_hour_support": after_hours,
                        "before_global_hour_complete_flag": before_complete,
                        "after_global_hour_complete_flag": after_complete,
                        "outcome_only_not_model_feature": True,
                    }
                    record.update(
                        {f"before_{name}": value for name, value in before.items()}
                    )
                    record.update(
                        {f"after_{name}": value for name, value in after.items()}
                    )
                    for metric in (
                        "direct_mean_net",
                        "positive_net_contribution",
                        "loss_net_contribution",
                        "net_positive_rate",
                        "positive_net_contribution_robust_mean",
                        "loss_net_contribution_robust_mean",
                        "robust_component_net",
                    ):
                        record[f"delta_{metric}"] = after[metric] - before[metric]
                    records.append(record)
    return (
        pd.DataFrame.from_records(records)
        .sort_values(list(KEY), kind="stable")
        .reset_index(drop=True)
    )


def verify_against_base(
    contribution: pd.DataFrame, base: pd.DataFrame
) -> dict[str, Any]:
    columns = [
        *KEY,
        "before_candidate_support",
        "after_candidate_support",
        "before_target_available_utc",
        "after_target_available_utc",
        "before_global_hour_complete_flag",
        "after_global_hour_complete_flag",
        "delta_direct_mean_net",
    ]
    joined = contribution.merge(
        base.loc[:, columns],
        on=list(KEY),
        how="outer",
        validate="one_to_one",
        suffixes=("", "__base"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("contribution/base label cohort identity mismatch")
    checks = {
        "before_candidate_support": np.array_equal(
            joined["before_candidate_support"],
            joined["before_candidate_support__base"],
        ),
        "after_candidate_support": np.array_equal(
            joined["after_candidate_support"],
            joined["after_candidate_support__base"],
        ),
        "before_complete": np.array_equal(
            joined["before_global_hour_complete_flag"],
            joined["before_global_hour_complete_flag__base"],
        ),
        "after_complete": np.array_equal(
            joined["after_global_hour_complete_flag"],
            joined["after_global_hour_complete_flag__base"],
        ),
        "before_availability": pd.to_datetime(
            joined["before_target_available_utc"], utc=True
        ).equals(
            pd.to_datetime(joined["before_target_available_utc__base"], utc=True)
        ),
        "after_availability": pd.to_datetime(
            joined["after_target_available_utc"], utc=True
        ).equals(
            pd.to_datetime(joined["after_target_available_utc__base"], utc=True)
        ),
    }
    valid = (
        pd.to_numeric(joined["delta_direct_mean_net"], errors="coerce").notna()
        & pd.to_numeric(
            joined["delta_direct_mean_net__base"], errors="coerce"
        ).notna()
    )
    checks["delta_direct_mean"] = bool(
        np.allclose(
            joined.loc[valid, "delta_direct_mean_net"],
            joined.loc[valid, "delta_direct_mean_net__base"],
            rtol=0.0,
            atol=1e-12,
        )
    )
    raw_delta = (
        joined["delta_positive_net_contribution"]
        - joined["delta_loss_net_contribution"]
    )
    checks["delta_component_reconciliation"] = bool(
        np.allclose(
            raw_delta.loc[valid],
            joined.loc[valid, "delta_direct_mean_net"],
            rtol=0.0,
            atol=1e-12,
        )
    )
    if not all(checks.values()):
        raise ValueError(f"contribution-label reconciliation failed: {checks}")
    return {
        "checks": checks,
        "joined_rows": int(len(joined)),
        "resolved_direct_delta_rows": int(valid.sum()),
    }


def plan(source: Path, base_label_source: Path, output: Path) -> dict[str, Any]:
    return {
        "action": "PLAN_ONLY_NO_MATERIALIZATION",
        "schema": SCHEMA,
        "source": str(source),
        "base_label_source": str(base_label_source),
        "output": str(output),
        "source_sha256": _source_hashes(source),
        "base_label_sha256": _base_label_hashes(base_label_source),
        "raw_accounting": "direct mean net = positive net contribution - loss net contribution",
        "soft_target": "net-positive rate is a cohort mean, not a hard candidate label",
        "feature_surface": [],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source)
    base_label_source = Path(args.base_label_source)
    output = Path(args.output_dir)
    if args.plan_only:
        return plan(source, base_label_source, output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    source_hashes = _source_hashes(source)
    base_hashes = _base_label_hashes(base_label_source)
    source_frame = pd.read_parquet(
        source / "panel.parquet", columns=list(REQUIRED_COLUMNS)
    )
    contribution = materialize_contribution_labels(source_frame)
    base = pd.read_parquet(base_label_source / "cohort_transition_labels.parquet")
    verification = verify_against_base(contribution, base)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    contribution.to_parquet(
        temporary / "cohort_contribution_labels.parquet",
        index=False,
        compression="zstd",
    )
    summary = (
        contribution.groupby(["horizon_hours", "side_name"], observed=True)
        .agg(
            rows=("cohort_anchor_utc", "size"),
            complete_after=("after_global_hour_complete_flag", "sum"),
            after_candidate_support_median=("after_candidate_support", "median"),
            after_net_positive_support_median=("after_net_positive_support", "median"),
            reconciled_before=(
                "before_raw_components_reconcile_direct_mean_flag",
                "sum",
            ),
            reconciled_after=(
                "after_raw_components_reconcile_direct_mean_flag",
                "sum",
            ),
        )
        .reset_index()
    )
    summary.to_parquet(
        temporary / "support_and_reconciliation_summary.parquet",
        index=False,
        compression="zstd",
    )
    manifest = {
        "schema": SCHEMA,
        "status": "IMMUTABLE_OUTCOME_ONLY_LABEL_ARTIFACT",
        "source_sha256": source_hashes,
        "base_label_sha256": base_hashes,
        "rows": int(len(contribution)),
        "verification": verification,
        "label_columns_not_model_features": [
            column
            for column in contribution.columns
            if column not in KEY
        ],
        "contracts": {
            "raw_accounting": "positive contribution is mean(max(net,0)); loss contribution is mean(max(-net,0)); their difference exactly equals direct mean net",
            "soft_label": "net-positive rate is mean(net>0) over all candidates",
            "robust_sensitivity": "p05/p95 within-window winsorized means of zero-inclusive positive/loss contributions; they are not claimed to reconcile raw direct net",
            "availability": "actual execution-label end plus one hour; exact parity with the base transition-label artifact",
            "feature_usage": "all columns are outcomes/audit metadata and prohibited model features",
        },
        "outputs_sha256": {
            path.name: sha256(path) for path in sorted(temporary.glob("*.parquet"))
        },
        "checksum_convention": "manifest.json is verified by detached manifest.sha256",
    }
    (temporary / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {
        "output": str(output),
        "rows": int(len(contribution)),
        "resolved_direct_delta_rows": verification["resolved_direct_delta_rows"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", type=Path, default=SOURCE)
    result.add_argument("--base-label-source", type=Path, default=BASE_LABEL_SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--plan-only", action="store_true")
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
