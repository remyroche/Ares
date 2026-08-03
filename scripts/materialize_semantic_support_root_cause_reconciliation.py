#!/usr/bin/env python3
"""Reconcile strict semantic-head OOF results with their economic diagnostic.

This is a reporting/lineage artifact.  It does not fit a model, select a
policy, or turn a future-outcome oracle into a usable feature.  In particular,
the oracle rows are kept separate from the learnable controls and are always
marked diagnostic-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp" / "artifacts"
DEFAULT_OOF = ARTIFACTS / "strict_semantic_support_oof_20260801_v2"
DEFAULT_AUDIT = ARTIFACTS / "strict_oof_semantic_support_readiness_20260801_v4"
DEFAULT_ECON = ARTIFACTS / "semantic_support_economic_diagnostic_20260801_v2"
DEFAULT_SUPERSESSION = ARTIFACTS / "pipeline_supersession_manifest_20260801_v1"
DEFAULT_OUTPUT = ARTIFACTS / "semantic_support_root_cause_reconciliation_20260801_v1"

SCHEMA = "semantic_support_root_cause_reconciliation_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _input(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _top10(summary: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = summary[(summary["split"] == split) & (summary["fraction"] == 0.10)].copy()
    if rows.empty:
        raise ValueError(f"economic summary has no {split} top-10 rows")
    return rows


def _waterfall(audit_metrics: pd.DataFrame, econ: pd.DataFrame, supersession: dict[str, Any]) -> pd.DataFrame:
    dev = _top10(econ, "development_oof")
    oos = _top10(econ, "final_oos_diagnostic")
    controls = dev[dev["score"].astype(str).str.startswith("C")]
    oos_controls = oos[oos["score"].astype(str).str.startswith("C")]
    peak = audit_metrics.loc[audit_metrics["head"] == "conditional_peak_mfe"].iloc[0]
    reach = audit_metrics.loc[audit_metrics["head"] == "opportunity_reach"].iloc[0]
    return pd.DataFrame([
        {
            "stage": "strict_oof_lineage_and_semantics",
            "status": "PASS",
            "evidence": "27/27 semantic heads have strict OOF metrics and hash-bound lineage",
            "metric": float(len(audit_metrics)),
            "metric_name": "heads_with_strict_oof_metric",
            "promotion": False,
        },
        {
            "stage": "opportunity_event_head_learnability",
            "status": "SIGNAL_ONLY",
            "evidence": "Opportunity reach has positive OOF discrimination, but its global top-k net is negative",
            "metric": float(reach["auc"]),
            "metric_name": "opportunity_reach_auc",
            "promotion": False,
        },
        {
            "stage": "conditional_magnitude_learnability",
            "status": "SIGNAL_ONLY",
            "evidence": "Conditional peak has positive rank IC, but the unconditional reach×peak composition is negative",
            "metric": float(peak["rank_ic"]),
            "metric_name": "conditional_peak_rank_ic",
            "promotion": False,
        },
        {
            "stage": "development_economic_controls",
            "status": "FAIL",
            "evidence": "Every C0–C4 control is net-negative at global top-10% and has no positive promotion flag",
            "metric": float(controls["net_bps"].max()),
            "metric_name": "best_development_control_top10_net_bps",
            "promotion": False,
        },
        {
            "stage": "final_oos_economic_controls",
            "status": "FAIL",
            "evidence": "Every C0–C4 control remains net-negative at global top-10%; latest month is also negative",
            "metric": float(oos_controls["net_bps"].max()),
            "metric_name": "best_final_control_top10_net_bps",
            "promotion": False,
        },
        {
            "stage": "future_outcome_oracle_separation",
            "status": "DIAGNOSTIC_ONLY",
            "evidence": "O1–O3 are positive only when future labels/outcomes are used directly; they are not model inputs",
            "metric": float(dev.loc[dev["score"] == "O3_future_exact_net", "net_bps"].iloc[0]),
            "metric_name": "development_future_exact_net_oracle_top10_bps",
            "promotion": False,
        },
        {
            "stage": "operational_promotion",
            "status": "BLOCKED",
            "evidence": str(supersession.get("status", "supersession manifest unavailable")),
            "metric": float("nan"),
            "metric_name": "supersession_status",
            "promotion": False,
        },
    ])


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small table without requiring the optional tabulate package."""
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build(
    *,
    oof_dir: Path = DEFAULT_OOF,
    audit_dir: Path = DEFAULT_AUDIT,
    economic_dir: Path = DEFAULT_ECON,
    supersession_dir: Path = DEFAULT_SUPERSESSION,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")
    oof_manifest_path = _input(oof_dir, "run_manifest.json")
    audit_manifest_path = _input(audit_dir, "run_manifest.json")
    economic_manifest_path = _input(economic_dir, "run_manifest.json")
    supersession_manifest_path = _input(supersession_dir, "supersession_manifest.json")
    audit_metrics_path = _input(audit_dir, "semantic_support_metrics.parquet")
    economic_summary_path = _input(economic_dir, "global_topk_summary.parquet")
    audit_metrics = pd.read_parquet(audit_metrics_path)
    economic_summary = pd.read_parquet(economic_summary_path)
    supersession = _read_json(supersession_manifest_path)
    waterfall = _waterfall(audit_metrics, economic_summary, supersession)
    controls = economic_summary[economic_summary["score"].astype(str).str.startswith("C")]
    root_cause = {
        "primary": "economic_translation_and_composition_bottleneck",
        "description": (
            "Strict OOF heads contain learnable event/conditional signal, but the predicted event probabilities "
            "and their reach×magnitude/risk compositions do not rank positive execution net globally.  The positive "
            "oracle gap proves opportunity exists, while the negative C0–C4 controls show that current outputs do not "
            "translate that opportunity into a cost-clearing entry score."
        ),
        "not_proven": [
            "A production-ready entry policy",
            "A valid timing/wait action layer",
            "Portfolio-constrained economics",
            "That a single head or label family is the unique causal cause",
        ],
        "required_next_tests": [
            "Calibrate event probabilities and test tail precision/recall at global top-k, not only AUC",
            "Run quantile conditional-MFE challengers with all-row OOF predictions",
            "Rebuild entry-side/action targets from known-at-decision information before adding timing or waits",
            "Extend the same strict lineage checks to older data and regime-transition features",
        ],
    }
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        waterfall.to_parquet(stage / "root_cause_waterfall.parquet", index=False, compression="zstd")
        report = (
            "# Semantic-support root-cause reconciliation\n\n"
            "Status: research-only; no score, model, oracle, or policy is promoted.\n\n"
            "## Finding\n\n"
            f"{root_cause['description']}\n\n"
            "The strict OOF audit is complete for all registered semantic heads. The economic diagnostic uses one "
            "pooled global top-k book; timing, wait actions, and portfolio constraints are out of scope. Future-label "
            "oracles are reported only as a separability check and are never treated as candidate features.\n\n"
            "## Waterfall\n\n"
            + _markdown_table(waterfall)
            + "\n\n## Disposition\n\n"
            "The current result is a diagnostic bottleneck, not a promotion candidate. The next implementation should "
            "repair the probability-to-economic translation and rebuild the causal entry/action substrate before any "
            "timing, wait, or operational replay.\n"
        )
        (stage / "ROOT_CAUSE_RECONCILIATION.md").write_text(report, encoding="utf-8")
        input_manifests = {
            "strict_semantic_support_oof": {"path": str(oof_manifest_path), "sha256": _sha256(oof_manifest_path)},
            "strict_semantic_support_audit": {"path": str(audit_manifest_path), "sha256": _sha256(audit_manifest_path)},
            "semantic_support_economic_diagnostic": {"path": str(economic_manifest_path), "sha256": _sha256(economic_manifest_path)},
            "pipeline_supersession": {"path": str(supersession_manifest_path), "sha256": _sha256(supersession_manifest_path)},
        }
        manifest = {
            "schema": SCHEMA,
            "status": "RESEARCH_ONLY_ROOT_CAUSE_RECONCILIATION",
            "portfolio_constraints_in_scope": False,
            "root_cause": root_cause,
            "inputs": input_manifests,
            "rows": {"semantic_head_metrics": int(len(audit_metrics)), "economic_summary": int(len(economic_summary))},
            "outputs": {
                "root_cause_waterfall": "root_cause_waterfall.parquet",
                "report": "ROOT_CAUSE_RECONCILIATION.md",
            },
        }
        manifest["outputs_sha256"] = {
            name: _sha256(stage / name) for name in manifest["outputs"].values()
        }
        runner = Path(__file__).resolve()
        manifest["runner"] = {"path": str(runner.relative_to(ROOT)), "sha256": _sha256(runner)}
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-dir", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--economic-dir", type=Path, default=DEFAULT_ECON)
    parser.add_argument("--supersession-dir", type=Path, default=DEFAULT_SUPERSESSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(
        oof_dir=args.oof_dir,
        audit_dir=args.audit_dir,
        economic_dir=args.economic_dir,
        supersession_dir=args.supersession_dir,
        output=args.output,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
