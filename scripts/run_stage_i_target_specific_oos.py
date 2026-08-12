#!/usr/bin/env python3
"""Run or preflight the frozen direct S/O -> FQ3 Stage-I OOS evaluator.

Input layout (one immutable directory per side) is deliberately narrow:

```
input_root/
  long/{features.parquet,contract.parquet,manifest.json}
  short/{features.parquet,contract.parquet,manifest.json}
```

Each input manifest must contain byte hashes for ``features.parquet`` and
``contract.parquet`` under ``artifact_sha256``, plus the declared
``base_target_column`` and ``meta_target_column``.  It is a real execution
CLI, but ``--preflight`` verifies every source/winner/feature/timing contract
without fitting a model or creating an artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.lgbm_pipeline import _fit_lgbm_model
from extreme_price_movements.stage_i_adapter_winner_bundle import StageIAdapterWinnerBundle
from extreme_price_movements.stage_i_causal_admission import Causal21dAdmissionSpec
from extreme_price_movements.stage_i_target_adapter import canonical_sha256, file_sha256
from extreme_price_movements.stage_i_target_specific_oos import (
    StageITargetSpecificInput,
    _validate_input,
    preflight_strict_meta_availability,
    run_stage_i_target_specific_oos,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _selector(root: Path, side: str) -> tuple[dict[str, Any], str]:
    path = root / side / "manifest.json"
    if not path.is_file():
        side_root = root / side
        pointer_path = side_root / "resume_complete.json"
        if pointer_path.is_file():
            pointer = _read_json(pointer_path)
            if pointer.get("schema") != "stage_i_direct_fq3_resume_complete_v1" or pointer.get("side") != side:
                raise ValueError(f"{side}: invalid direct-FQ3 resume pointer")
            relative = pointer.get("attempt_relative_path")
            if not isinstance(relative, str):
                raise ValueError(f"{side}: direct-FQ3 resume pointer lacks attempt path")
            resolved = (side_root / relative / "manifest.json").resolve()
            attempts = (side_root / "_resume_attempts").resolve()
            if attempts not in resolved.parents or resolved.parent.parent != attempts:
                raise ValueError(f"{side}: direct-FQ3 resume pointer escapes attempt root")
            if not resolved.is_file() or file_sha256(resolved) != pointer.get("attempt_manifest_sha256"):
                raise ValueError(f"{side}: direct-FQ3 resume manifest hash drift")
            path = resolved
    return _read_json(path), file_sha256(path)


def _source(root: Path, side: str, *, base_selectors: Path, meta_selectors: Path) -> StageITargetSpecificInput:
    directory = root / side
    feature_path, contract_path, manifest_path = (
        directory / "features.parquet", directory / "contract.parquet", directory / "manifest.json",
    )
    manifest = _read_json(manifest_path)
    expected = {"features.parquet": file_sha256(feature_path), "contract.parquet": file_sha256(contract_path)}
    base, base_sha = _selector(base_selectors, side)
    meta, meta_sha = _selector(meta_selectors, side)
    frozen_path_value = manifest.get("frozen_base_oof_path")
    frozen_path = None if frozen_path_value is None else Path(str(frozen_path_value))
    if frozen_path is not None and not frozen_path.is_absolute():
        frozen_path = (directory / frozen_path).resolve()
    frozen_frame = None if frozen_path is None else pd.read_parquet(frozen_path)
    frozen_sha = "" if frozen_path is None else file_sha256(frozen_path)
    return StageITargetSpecificInput(
        side=side, frame=pd.read_parquet(feature_path), contract_frame=pd.read_parquet(contract_path),
        source_manifest=manifest, source_manifest_sha256=canonical_sha256(manifest), source_file_sha256=expected,
        base_selector_manifest=base, meta_selector_manifest=meta,
        base_selector_manifest_sha256=base_sha, meta_selector_manifest_sha256=meta_sha,
        base_target_column=str(manifest["base_target_column"]), meta_target_column=str(manifest["meta_target_column"]),
        frozen_base_oof=frozen_frame,
        frozen_base_oof_manifest=base if frozen_frame is not None else None,
        frozen_base_oof_file_sha256=frozen_sha,
        frozen_base_oof_manifest_sha256=base_sha if frozen_frame is not None else "",
        n_validation_folds=int(manifest.get("n_validation_folds", 4)), min_train_rows=int(manifest.get("min_train_rows", 500)),
    )


def _fit(frame: pd.DataFrame, target: np.ndarray, weight: np.ndarray, *, classifier: bool, params: dict[str, Any], objective_mode: str) -> Any:
    return _fit_lgbm_model(
        frame, np.asarray(target), np.asarray(weight, dtype=np.float32), classifier=bool(classifier),
        params=dict(params), objective_mode=str(objective_mode),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner-bundle", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--base-selector-dir", type=Path, required=True)
    parser.add_argument("--meta-selector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--admission-floor-bps", type=float, default=50.0)
    args = parser.parse_args(argv)
    if args.preflight == (args.output_dir is not None):
        parser.error("provide exactly one of --preflight or --output-dir")
    bundle = StageIAdapterWinnerBundle.from_dict(_read_json(args.winner_bundle))
    inputs = tuple(_source(args.input_root, side, base_selectors=args.base_selector_dir, meta_selectors=args.meta_selector_dir) for side in ("long", "short"))
    if args.preflight:
        summaries = []
        for source in inputs:
            base, meta, _frame, contract, values = _validate_input(bundle, source)
            summaries.append({
                "side": source.side, "rows": len(contract), "base_target": base.family,
                "meta_target": meta.family, "geometry": base.geometry,
                "base_execution": (
                    "reuse_frozen_strict_oof_no_refit" if source.frozen_base_oof is not None
                    else "fit_strict_oof"
                ),
                # Base and meta selection are intentionally independent
                # contracts.  Do not collapse them into one implied policy in
                # a preflight receipt.
                "base_correlation_policy": values["base_correlation_policy"],
                "meta_correlation_policy": values["meta_correlation_policy"],
            })
        _projection, availability = preflight_strict_meta_availability(bundle, inputs)
        print(json.dumps({
            "status": "preflight_complete_no_fit",
            "schema": "stage_i_target_specific_direct_fq3_oos_v1",
            "sides": summaries,
            "strict_meta_availability": availability,
        }, indent=2))
        return 0
    spec = Causal21dAdmissionSpec(net_floor_bps=float(args.admission_floor_bps))
    result = run_stage_i_target_specific_oos(
        bundle=bundle, inputs=inputs, output_dir=args.output_dir, fit_model=_fit, admission_spec=spec,
    )
    print(json.dumps({"status": result["status"], "schema": result["schema"], "output_dir": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
