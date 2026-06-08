#!/usr/bin/env python3
"""Run label -> weight -> distillation Optuna phases sequentially."""
from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _phase_cmd(
    *,
    phase: str,
    out_dir: Path,
    study_name: str,
    run_id_prefix: str,
    base_recipe: Path | None = None,
    description: str = "",
    trials: int = 300,
    study_patience: int = 40,
) -> list[str]:
    del description
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "extreme_price_movements.label_weight_optuna",
        "--phase",
        phase,
        "--study-name",
        study_name,
        "--trials",
        str(trials),
        "--study-patience",
        str(study_patience),
        "--pruner",
        "successive_halving",
        "--out-dir",
        str(out_dir),
        "--best-recipe-path",
        str(out_dir / "best_recipe.json"),
        "--lgbm-n-estimators-cap",
        "200",
        "--lgbm-hpo-trials",
        "0",
        "--lgbm-early-stopping-rounds",
        "30",
        "--eval-command",
        "__fast_long_dist__",
        "--metrics-json",
        str(out_dir / "trial_metrics_{trial}.json"),
    ]
    if base_recipe is not None:
        cmd.extend(["--base-recipe", str(base_recipe), "--previous-best", str(base_recipe)])
    return cmd


def main() -> int:
    base_out = Path("reports_perp/label_weight_optuna/long_dist_seq_fast_20260607_v10_topk_noise_robust")
    phases = [
        {
            "phase": "label_geometry",
            "description": "phase 0/4: optimise hard/path label geometry",
            "out_dir": base_out / "00_label_geometry",
            "study_name": "label_weight_long_dist_seq_fast_geometry_20260607_v10_topk_noise_robust",
            "run_id_prefix": "20260607_label_weight_optuna_long_dist_seq_fast_v10_geometry",
            "base_recipe": None,
        },
        {
            "phase": "labels",
            "description": "phase 1/4: optimise soft-label construction with best geometry fixed",
            "out_dir": base_out / "01_labels",
            "study_name": "label_weight_long_dist_seq_fast_labels_20260607_v10_topk_noise_robust",
            "run_id_prefix": "20260607_label_weight_optuna_long_dist_seq_fast_v10_labels",
            "base_recipe": base_out / "00_label_geometry" / "best_recipe.json",
        },
        {
            "phase": "weights",
            "description": "phase 2/4: optimise sample weights with best geometry and labels fixed",
            "out_dir": base_out / "02_weights",
            "study_name": "label_weight_long_dist_seq_fast_weights_20260607_v10_topk_noise_robust",
            "run_id_prefix": "20260607_label_weight_optuna_long_dist_seq_fast_v10_weights",
            "base_recipe": base_out / "01_labels" / "best_recipe.json",
        },
        {
            "phase": "distillation",
            "description": "phase 3/4: optimise self-distillation with best geometry, labels, and weights fixed",
            "out_dir": base_out / "03_distillation",
            "study_name": "label_weight_long_dist_seq_fast_distillation_20260607_v10_topk_noise_robust",
            "run_id_prefix": "20260607_label_weight_optuna_long_dist_seq_fast_v10_distill",
            "base_recipe": base_out / "02_weights" / "best_recipe.json",
        },
    ]
    log_path = LOG_DIR / "label_weight_optuna_long_dist_sequential_20260607_v10_topk_noise_robust.log"
    with log_path.open("ab", buffering=0) as log_fp:
        log_fp.write(b"\n=== START sequential label_weight_optuna long_dist ===\n")
        for spec in phases:
            base_recipe = spec["base_recipe"]
            if base_recipe is not None and not Path(base_recipe).exists():
                raise FileNotFoundError(f"Previous phase best recipe missing: {base_recipe}")
            phase_best = Path(spec["out_dir"]) / "best_recipe.json"
            rejected_marker = Path(spec["out_dir"]) / "promotion_rejected.json"
            if phase_best.exists():
                log_fp.write(
                    (
                        f"\n=== PHASE {spec['phase']} SKIP "
                        f"existing best_recipe={phase_best} ===\n"
                    ).encode()
                )
                continue
            if rejected_marker.exists() and base_recipe is not None:
                shutil.copyfile(base_recipe, phase_best)
                log_fp.write(
                    (
                        f"\n=== PHASE {spec['phase']} CARRY_FORWARD "
                        f"rejected_marker={rejected_marker} "
                        f"base_recipe={base_recipe} best_recipe={phase_best} ===\n"
                    ).encode()
                )
                continue
            cmd = _phase_cmd(**spec)
            log_fp.write(
                (
                    f"\n=== PHASE {spec['phase']} START "
                    f"{spec['description']} "
                    f"out_dir={spec['out_dir']} base_recipe={base_recipe or ''} ===\n"
                ).encode()
            )
            ret = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=log_fp,
                stderr=subprocess.STDOUT,
            ).returncode
            log_fp.write(f"\n=== PHASE {spec['phase']} END ret={ret} ===\n".encode())
            if ret != 0:
                return int(ret)
            if not phase_best.exists() and base_recipe is not None:
                shutil.copyfile(base_recipe, phase_best)
                log_fp.write(
                    (
                        f"\n=== PHASE {spec['phase']} CARRY_FORWARD "
                        f"missing best after successful phase; "
                        f"base_recipe={base_recipe} best_recipe={phase_best} ===\n"
                    ).encode()
                )
        log_fp.write(b"\n=== END sequential label_weight_optuna long_dist ret=0 ===\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
