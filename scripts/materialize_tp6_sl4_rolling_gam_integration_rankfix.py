#!/usr/bin/env python3
"""Re-materialize integration metrics with native base-score control ranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_rolling_gam_residual_integration import _metrics, _metrics_by_arm


def run(input_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    pred = pd.read_parquet(input_dir / "predictions.parquet")
    pred["base_score_rank_native"] = pred.groupby(["arm", "month"], sort=False)["base_score"].transform(lambda z: z.rank(pct=True, method="average")).astype("float32")
    no_mod = pred["arm"].isin(["control", "gam_input"])
    pred.loc[no_mod, "anchor_rank"] = pred.loc[no_mod, "base_score_rank_native"].to_numpy()
    pred["stack_rank"] = (0.50 * pred["anchor_rank"] + 0.25 * pred["consensus_rank"] + 0.25 * pred["residual_rank"]).astype("float32")
    pred["stack_score"] = pred.groupby(["arm", "month"], sort=False)["stack_rank"].transform(lambda z: z.rank(pct=True, method="average")).astype("float32")
    pred["anchor_score"] = pred.groupby(["arm", "month"], sort=False)["anchor_rank"].transform(lambda z: z.rank(pct=True, method="average")).astype("float32")
    glob, monthly, stability = _metrics_by_arm(pred, ["stack_score", "anchor_score"])
    control_eval = pred.drop_duplicates(["candidate_id", "month"]).copy()
    if "existing_control_stack" in control_eval:
        cg, cm, cs = _metrics(control_eval.rename(columns={"existing_control_stack": "existing_control_stack_score"}), ["existing_control_stack_score"])
        cg["arm"] = "existing_control__existing_control_stack_score"; cm["arm"] = "existing_control__existing_control_stack_score"; cs["arm"] = "existing_control__existing_control_stack_score"
        glob = pd.concat([cg, glob], ignore_index=True); monthly = pd.concat([cm, monthly], ignore_index=True); stability = pd.concat([cs, stability], ignore_index=True)
    output_dir.mkdir(parents=True)
    pred.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    glob.to_parquet(output_dir / "metrics_global.parquet", index=False)
    monthly.to_parquet(output_dir / "metrics_monthly.parquet", index=False)
    stability.to_parquet(output_dir / "metrics_stability.parquet", index=False)
    for name in ("fit_audit.parquet",):
        source = input_dir / name
        if source.exists():
            pd.read_parquet(source).to_parquet(output_dir / name, index=False)
    manifest = json.loads((input_dir / "run_manifest.json").read_text())
    manifest["schema"] = "tp6_sl4_rolling_gam_residual_integration_v1_rankfixed"
    manifest["status"] = "COMPLETE"
    manifest["control_ranking"] = "native base_score percentile; GAM-modulated arms rank gam_expected_bps"
    manifest["source_artifact"] = str(input_dir)
    manifest["artifacts"] = sorted(p.name for p in output_dir.iterdir())
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    lines = ["# TP6/SL4 one-month gated GAM residual/meta integration (rank-fixed)", "", "The matched control uses the native base score for its base ranking; GAM-modulated arms use the gated GAM bps anchor.", "", "## Global metrics", "", glob.round(3).to_string(index=False), "", "## Stability", "", stability.round(3).to_string(index=False)]
    (output_dir / "TP6_SL4_ROLLING_GAM_RESIDUAL_INTEGRATION_REPORT.md").write_text("\n".join(lines) + "\n")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.input_dir, args.output_dir))
