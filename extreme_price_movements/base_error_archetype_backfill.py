from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.lgbm_archetype_features import (
    BASE_ERROR_ARCHETYPE_FEATURE_NAMES,
    ResidualErrorArchetypeState,
    fit_residual_error_archetype_state,
    transform_residual_error_archetype_features,
)
from extreme_price_movements.utils import tprint


BASE_ERROR_SIGNATURE_TOKENS = (
    "unc",
    "uncertainty",
    "pred_std",
    "pred_cv",
    "prob_std",
    "raw_score_std",
    "vote_",
    "leaf_",
    "rare_leaf",
    "low_support",
    "surprisal",
    "centroid",
    "contrib_",
    "archetype_",
    "raw_state_",
    "state_log_likelihood",
    "mahalanobis",
    "knn",
    "reconstruction",
    "transition",
    "feature_drift",
    "drift",
    "psi",
    "ks",
    "cov_shift",
    "frobenius",
    "regime_centroid",
    "rank_bin",
    "score_margin",
    "rank_margin",
    "score_path",
    "rank_path",
    "entropy",
    "variance_proxy",
)


def _parse_oof_name(path: Path) -> tuple[str, int] | None:
    match = re.match(r"^oof_(?P<strategy>.+)_H(?P<horizon>\d+)\.parquet$", path.name)
    if not match:
        return None
    return str(match.group("strategy")), int(match.group("horizon"))


def _base_error_bad_label(y_metric: Any, pred: Any) -> np.ndarray:
    y = np.asarray(y_metric, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    n = int(min(len(y), len(p)))
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out
    y = y[:n]
    p = p[:n]
    finite = np.isfinite(y) & np.isfinite(p)
    if int(np.sum(finite)) < 3:
        return out
    y01 = (y >= 0.5).astype(np.float64)
    p_clip = np.clip(p, 0.0, 1.0)
    residual = np.abs(y01 - p_clip)
    wrong = (p_clip >= 0.5) != (y01 >= 0.5)
    try:
        threshold = float(np.nanquantile(residual[finite], 0.70))
    except Exception:
        threshold = float(np.nanmedian(residual[finite]))
    out[finite] = (wrong[finite] | (residual[finite] >= threshold)).astype(np.int8)
    if len(np.unique(out[finite])) < 2:
        threshold = float(np.nanmedian(residual[finite]))
        out[finite] = (residual[finite] >= threshold).astype(np.int8)
    return out


def _normalised_signature_frame(
    df: pd.DataFrame,
    *,
    max_features: int = 128,
) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for raw_col in df.columns:
        raw_name = str(raw_col)
        name = raw_name[4:] if raw_name.startswith("oof_") else raw_name
        low = name.lower()
        if "base_error_" in low:
            continue
        if not any(token in low for token in BASE_ERROR_SIGNATURE_TOKENS):
            continue
        if name in data:
            continue
        try:
            if not pd.api.types.is_numeric_dtype(df[raw_col]):
                continue
        except Exception:
            continue
        data[name] = pd.to_numeric(df[raw_col], errors="coerce")
        if len(data) >= int(max_features):
            break
    if not data:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(data, index=df.index)


def fit_base_error_archetype_from_oof(
    df: pd.DataFrame,
    *,
    random_state: int = 42,
    max_features: int = 128,
) -> tuple[ResidualErrorArchetypeState, pd.DataFrame, dict[str, Any]]:
    if "oof_prob" not in df.columns:
        raise ValueError("base OOF frame is missing required column oof_prob")
    if "y_bin" not in df.columns:
        raise ValueError("base OOF frame is missing required column y_bin")
    signature = _normalised_signature_frame(df, max_features=max_features)
    if signature.empty:
        state = ResidualErrorArchetypeState(
            feature_names=[],
            enabled=False,
            reason="no_signature_features",
        )
        features = transform_residual_error_archetype_features(
            signature,
            state,
            index=df.index,
        )
        return state, features, {
            "enabled": False,
            "reason": state.reason,
            "signature_feature_count": 0,
            "row_count": int(len(df)),
        }
    y_bad = _base_error_bad_label(df["y_bin"].values, df["oof_prob"].values)
    state = fit_residual_error_archetype_state(
        signature,
        y_bad,
        feature_names=list(signature.columns),
        random_state=int(random_state),
    )
    features = transform_residual_error_archetype_features(
        signature,
        state,
        index=df.index,
    ).reindex(columns=BASE_ERROR_ARCHETYPE_FEATURE_NAMES, fill_value=0.0)
    bad_rate = float(np.mean(y_bad)) if len(y_bad) else float("nan")
    return state, features, {
        "enabled": bool(getattr(state, "enabled", False)),
        "reason": str(getattr(state, "reason", "")),
        "row_count": int(len(df)),
        "signature_feature_count": int(len(getattr(state, "feature_names", []) or [])),
        "input_signature_feature_count": int(signature.shape[1]),
        "bad_label_rate": bad_rate,
        "cluster_count": int(len(getattr(state, "clusters", []) or [])),
        "bad_cluster_ids": [int(x) for x in getattr(state, "bad_cluster_ids", []) or []],
        "good_cluster_ids": [int(x) for x in getattr(state, "good_cluster_ids", []) or []],
        "feature_names": list(getattr(state, "feature_names", []) or []),
        "clusters": list(getattr(state, "clusters", []) or []),
    }


def backfill_artifact_base_error_archetypes(
    artifact_dir: str | os.PathLike[str],
    *,
    random_state: int = 42,
    max_features: int = 128,
    force: bool = False,
) -> dict[str, Any]:
    artifact_path = Path(artifact_dir)
    oof_dir = artifact_path / "oof"
    if not oof_dir.exists():
        raise FileNotFoundError(f"OOF directory not found: {oof_dir}")
    state_dir = oof_dir / "base_error_archetypes"
    state_dir.mkdir(parents=True, exist_ok=True)

    states: dict[tuple[str, int], ResidualErrorArchetypeState] = {}
    reports: list[dict[str, Any]] = []
    for parquet_path in sorted(oof_dir.glob("oof_*_H*.parquet")):
        parsed = _parse_oof_name(parquet_path)
        if parsed is None:
            continue
        strategy_id, horizon = parsed
        state_key = (str(strategy_id), int(horizon))
        state_path = state_dir / f"{parquet_path.stem}.pkl"
        df = pd.read_parquet(parquet_path)
        required_cols = [f"oof_{name}" for name in BASE_ERROR_ARCHETYPE_FEATURE_NAMES]
        already_complete = all(col in df.columns for col in required_cols)
        if already_complete and not force:
            existing_state = None
            if state_path.exists():
                try:
                    with open(state_path, "rb") as fh:
                        existing_state = pickle.load(fh)
                    states[state_key] = existing_state
                except Exception as exc:
                    tprint(
                        "Base-error archetype backfill: failed to load existing "
                        f"state for {parquet_path.name}: {exc}"
                    )
            reports.append(
                {
                    "enabled": bool(getattr(existing_state, "enabled", False)),
                    "reason": str(
                        getattr(existing_state, "reason", "")
                        or "already_complete"
                    ),
                    "row_count": int(len(df)),
                    "signature_feature_count": int(
                        len(getattr(existing_state, "feature_names", []) or [])
                    ),
                    "input_signature_feature_count": int(
                        len(getattr(existing_state, "feature_names", []) or [])
                    ),
                    "bad_label_rate": float("nan"),
                    "cluster_count": int(
                        len(getattr(existing_state, "clusters", []) or [])
                    ),
                    "bad_cluster_ids": [
                        int(x)
                        for x in getattr(existing_state, "bad_cluster_ids", []) or []
                    ],
                    "good_cluster_ids": [
                        int(x)
                        for x in getattr(existing_state, "good_cluster_ids", []) or []
                    ],
                    "feature_names": list(
                        getattr(existing_state, "feature_names", []) or []
                    ),
                    "clusters": list(getattr(existing_state, "clusters", []) or []),
                    "strategy_id": str(strategy_id),
                    "horizon": int(horizon),
                    "oof_path": str(parquet_path),
                    "state_path": str(state_path),
                    "written_columns": required_cols,
                    "skipped_existing": True,
                }
            )
            tprint(
                f"Base-error archetype backfill: skipping {parquet_path.name} "
                "(already complete)."
            )
            continue
        state, features, report = fit_base_error_archetype_from_oof(
            df,
            random_state=int(random_state),
            max_features=int(max_features),
        )
        for name in BASE_ERROR_ARCHETYPE_FEATURE_NAMES:
            df[f"oof_{name}"] = pd.to_numeric(
                features[name],
                errors="coerce",
            ).astype(np.float32)
        tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, parquet_path)

        states[state_key] = state
        with open(state_path, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        report.update(
            {
                "strategy_id": str(strategy_id),
                "horizon": int(horizon),
                "oof_path": str(parquet_path),
                "state_path": str(state_path),
                "written_columns": required_cols,
            }
        )
        reports.append(report)
        tprint(
            f"Base-error archetype backfill: {parquet_path.name} "
            f"enabled={report['enabled']} reason={report['reason']} "
            f"signature_features={report['signature_feature_count']}."
        )

    states_path = state_dir / "states.pkl"
    with open(states_path, "wb") as fh:
        pickle.dump(states, fh, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "schema_version": "base_error_archetype_backfill_v1",
        "artifact_dir": str(artifact_path),
        "state_dir": str(state_dir),
        "states_path": str(states_path),
        "state_count": int(len(states)),
        "reports": reports,
    }
    manifest_path = state_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    pd.DataFrame(reports).to_csv(state_dir / "report.csv", index=False)
    return manifest


def load_base_error_archetype_states(
    artifact_dir: str | os.PathLike[str],
) -> dict[tuple[str, int], ResidualErrorArchetypeState]:
    states_path = Path(artifact_dir) / "oof" / "base_error_archetypes" / "states.pkl"
    if not states_path.exists():
        return {}
    with open(states_path, "rb") as fh:
        states = pickle.load(fh)
    if not isinstance(states, dict):
        return {}
    out: dict[tuple[str, int], ResidualErrorArchetypeState] = {}
    for key, state in states.items():
        if isinstance(key, tuple) and len(key) == 2:
            out[(str(key[0]), int(key[1]))] = state
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill base_error_* archetype OOF columns from existing base OOF artifacts.",
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    manifest = backfill_artifact_base_error_archetypes(
        args.artifact_dir,
        random_state=int(args.random_state),
        max_features=int(args.max_features),
        force=bool(args.force),
    )
    tprint(
        "Base-error archetype backfill complete: "
        f"states={manifest.get('state_count', 0)} "
        f"manifest={Path(manifest.get('state_dir', '')) / 'manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
