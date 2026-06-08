#!/usr/bin/env python3
"""Run v17 direct-generator label-weight Optuna phases for long_dist.

v17 tunes the native label/sample-weight generator numbers directly. The fast
evaluator recomputes native soft labels and base sample weights from those
recipe parameters instead of scoring only post-generation multipliers.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _phase_cmd(
    *,
    phase: str,
    out_dir: Path,
    study_name: str,
    base_recipe: Path | None = None,
    neutral_baseline_metrics: Path | None = None,
    trials: int = 300,
    study_patience: int = 40,
) -> list[str]:
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
    if neutral_baseline_metrics is not None:
        cmd.extend(["--neutral-baseline-metrics", str(neutral_baseline_metrics)])
    return cmd


def main() -> int:
    base_out = Path("reports_perp/label_weight_optuna/long_dist_seq_fast_20260608_v17_aligned_economic_objective")
    global_best_recipe = Path("reports_perp/label_weight_optuna/best_recipe.json")
    global_best_trial = Path("reports_perp/label_weight_optuna/best_trial.json")
    phases = [
        {
            "phase": "label_geometry",
            "out_dir": base_out / "00_label_geometry",
            "study_name": "label_weight_long_dist_seq_fast_geometry_20260608_v17_aligned_economic_objective",
            "base_recipe": None,
        },
        {
            "phase": "labels",
            "out_dir": base_out / "01_labels",
            "study_name": "label_weight_long_dist_seq_fast_labels_20260608_v17_aligned_economic_objective",
            "base_recipe": base_out / "00_label_geometry" / "best_recipe.json",
        },
        {
            "phase": "weights",
            "out_dir": base_out / "02_weights",
            "study_name": "label_weight_long_dist_seq_fast_weights_20260608_v17_aligned_economic_objective",
            "base_recipe": base_out / "01_labels" / "best_recipe.json",
        },
        {
            "phase": "distillation",
            "out_dir": base_out / "03_distillation",
            "study_name": "label_weight_long_dist_seq_fast_distillation_20260608_v17_aligned_economic_objective",
            "base_recipe": base_out / "02_weights" / "best_recipe.json",
        },
    ]
    log_path = LOG_DIR / "label_weight_optuna_long_dist_sequential_20260608_v17_aligned_economic_objective.log"
    with log_path.open("ab", buffering=0) as log_fp:
        log_fp.write(b"\n=== START sequential label_weight_optuna long_dist v17_aligned_economic_objective ===\n")
        any_promoted = False
        neutral_baseline_metrics: Path | None = None
        for spec in phases:
            base_recipe = spec["base_recipe"]
            if base_recipe is not None and not Path(base_recipe).exists():
                raise FileNotFoundError(f"Previous phase best recipe missing: {base_recipe}")
            phase_best = Path(spec["out_dir"]) / "best_recipe.json"
            rejected_marker = Path(spec["out_dir"]) / "promotion_rejected.json"
            if phase_best.exists():
                log_fp.write(
                    f"\n=== PHASE {spec['phase']} SKIP existing best_recipe={phase_best} ===\n".encode()
                )
                continue
            if rejected_marker.exists() and base_recipe is not None:
                shutil.copyfile(base_recipe, phase_best)
                log_fp.write(
                    (
                        f"\n=== PHASE {spec['phase']} CARRY_FORWARD rejected_marker={rejected_marker} "
                        f"base_recipe={base_recipe} best_recipe={phase_best} ===\n"
                    ).encode()
                )
                continue
            if spec["phase"] != "label_geometry":
                neutral_baseline_metrics = base_out / "00_label_geometry" / "trial_metrics_0.json"
                if not neutral_baseline_metrics.exists():
                    raise FileNotFoundError(f"Neutral baseline metrics missing: {neutral_baseline_metrics}")
            cmd = _phase_cmd(**spec, neutral_baseline_metrics=neutral_baseline_metrics)
            log_fp.write(
                f"\n=== PHASE {spec['phase']} START out_dir={spec['out_dir']} ===\n".encode()
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
            if not rejected_marker.exists():
                any_promoted = True
            if not phase_best.exists() and base_recipe is not None:
                shutil.copyfile(base_recipe, phase_best)
                log_fp.write(
                    (
                        f"\n=== PHASE {spec['phase']} CARRY_FORWARD missing best after successful phase; "
                        f"base_recipe={base_recipe} best_recipe={phase_best} ===\n"
                    ).encode()
                )
        final_best_recipe = phases[-1]["out_dir"] / "best_recipe.json"
        final_best_trial = phases[-1]["out_dir"] / "best_trial.json"
        if final_best_recipe.exists():
            shutil.copyfile(final_best_recipe, base_out / "best_recipe.json")
            if any_promoted:
                global_best_recipe.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(final_best_recipe, global_best_recipe)
            log_fp.write(
                (
                    f"\n=== PUBLISHED best_recipe={final_best_recipe} "
                    f"run_best={base_out / 'best_recipe.json'} "
                    f"global_best={'updated:' + str(global_best_recipe) if any_promoted else 'preserved'} ===\n"
                ).encode()
            )
        if final_best_trial.exists():
            shutil.copyfile(final_best_trial, base_out / "best_trial.json")
            if any_promoted:
                global_best_trial.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(final_best_trial, global_best_trial)
            log_fp.write(
                (
                    f"\n=== PUBLISHED best_trial={final_best_trial} "
                    f"run_best={base_out / 'best_trial.json'} "
                    f"global_best={'updated:' + str(global_best_trial) if any_promoted else 'preserved'} ===\n"
                ).encode()
            )
        log_fp.write(b"\n=== END sequential label_weight_optuna long_dist v17_aligned_economic_objective ret=0 ===\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
