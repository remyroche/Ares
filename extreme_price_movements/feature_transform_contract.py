from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.utils import tprint


DEFAULT_NEVER_TRANSFORM_COLS = ("__ts__", "__symbol__", "timestamp", "symbol")
DEFAULT_PASSTHROUGH_COLS = ("barrier_pct", "atr_pct_raw")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        if not np.isfinite(val):
            return None
        return round(val, 12)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _is_passthrough_col(name: str, passthrough_cols: set[str], never_cols: set[str]) -> bool:
    if name in passthrough_cols or name in never_cols:
        return True
    if name.endswith("_raw"):
        return True
    return False


def _scope_mask(index: pd.Index, fit_scope: dict[str, Any] | None) -> np.ndarray:
    if not fit_scope:
        return np.ones(len(index), dtype=bool)
    idx = pd.to_datetime(index, utc=True, errors="coerce")
    mask = np.ones(len(idx), dtype=bool)

    def _utc_ts(value: Any) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    start = fit_scope.get("allowed_start_ts")
    end = fit_scope.get("allowed_end_ts")
    if start:
        mask &= idx >= _utc_ts(start)
    if end:
        mask &= idx <= _utc_ts(end)
    periods = fit_scope.get("allowed_periods") or []
    if periods:
        period_mask = np.zeros(len(idx), dtype=bool)
        for period in periods:
            if isinstance(period, dict):
                p_start = period.get("start") or period.get("start_ts")
                p_end = period.get("end") or period.get("end_ts")
            elif isinstance(period, (list, tuple)) and len(period) >= 2:
                p_start, p_end = period[0], period[1]
            else:
                continue
            if p_start is None or p_end is None:
                continue
            period_mask |= (idx >= _utc_ts(p_start)) & (idx <= _utc_ts(p_end))
        mask &= period_mask
    mask &= ~pd.isna(idx)
    return np.asarray(mask, dtype=bool)


def _scoped_values(df: pd.DataFrame, fit_scope: dict[str, Any] | None) -> np.ndarray:
    scoped = df
    symbols = fit_scope.get("symbols") if fit_scope else None
    if symbols:
        cols = [c for c in scoped.columns if str(c) in {str(s) for s in symbols}]
        scoped = scoped.loc[:, cols] if cols else scoped.iloc[:, :0]
    if scoped.empty:
        return np.asarray([], dtype=np.float64)
    mask = _scope_mask(scoped.index, fit_scope)
    scoped = scoped.loc[mask]
    vals = pd.to_numeric(scoped.stack(dropna=False), errors="coerce").to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return vals


def compute_contract_hash(payload: dict[str, Any]) -> str:
    hash_payload = dict(payload)
    hash_payload.pop("contract_hash", None)
    hash_payload.pop("created_at_utc", None)
    stable = json.dumps(_jsonable(hash_payload), sort_keys=True, separators=(",", ":"))
    return "sha256:" + sha256(stable.encode("utf-8")).hexdigest()


@dataclass
class FeatureTransformContract:
    schema_version: int
    run_id: str
    market_mode: str
    fit_scope: dict[str, Any]
    raw_feature_cols: list[str]
    transformed_feature_cols: list[str]
    passthrough_cols: list[str]
    never_transform_cols: list[str]
    per_column_stats: dict[str, dict[str, float | None]]
    transform_config: dict[str, Any]
    transformable_cols: list[str] = field(default_factory=list)
    required_warmup_hours: int = 0
    required_lookback_hours_by_feature: dict[str, int] = field(default_factory=dict)
    allow_missing_features: dict[str, bool] = field(default_factory=dict)
    fillable_features: dict[str, bool] = field(default_factory=dict)
    contract_hash: str = ""
    created_at_utc: str = ""

    @classmethod
    def fit_from_panels(
        cls,
        feats: dict[str, pd.DataFrame],
        cfg: dict[str, Any],
        run_id: str,
        fit_scope: dict[str, Any] | None = None,
    ) -> "FeatureTransformContract":
        if fit_scope is None and not bool(cfg.get("feature_transform_allow_full_fit", False)):
            raise RuntimeError(
                "FeatureTransformContract requires a training-scope fit_scope. "
                "Set feature_transform_allow_full_fit=true only for diagnostics/tests."
            )
        if fit_scope is None:
            tprint("WARNING: fitting feature transform contract on full available feature history.")
            fit_scope = {"stage_name": "full_available", "symbols": [], "allowed_periods": []}

        kind = str(cfg.get("feature_transform_kind", "robust")).lower()
        if kind not in {"none", "standard", "robust"}:
            raise ValueError(f"Unsupported feature_transform_kind={kind!r}")
        clip_q = list(cfg.get("feature_transform_clip_quantiles", [0.005, 0.995]) or [])
        if len(clip_q) != 2:
            clip_q = [0.005, 0.995]
        impute = str(cfg.get("feature_transform_impute", "median")).lower()
        passthrough = set(DEFAULT_PASSTHROUGH_COLS) | set(cfg.get("feature_transform_passthrough_cols", []) or [])
        never = set(DEFAULT_NEVER_TRANSFORM_COLS) | set(cfg.get("feature_transform_never_transform_cols", []) or [])

        raw_cols = sorted(str(k) for k, v in (feats or {}).items() if isinstance(v, pd.DataFrame))
        stats: dict[str, dict[str, float | None]] = {}
        transformed_cols: list[str] = []
        passthrough_cols: list[str] = []
        transformable_cols: list[str] = []
        required_lookback_by_feature: dict[str, int] = {}
        default_warmup = int(
            cfg.get(
                "feature_transform_required_warmup_hours",
                max(
                    int(cfg.get("causal_transform_roll_window_hours", 24 * 30) or 0),
                    int(cfg.get("transform_roll_window", 24 * 30) or 0),
                    24 * 30,
                ),
            )
            or 0
        )
        for col in raw_cols:
            if _is_passthrough_col(col, passthrough, never) or kind == "none":
                passthrough_cols.append(col)
                stats[col] = {"passthrough": 1.0}
                transformed_cols.append(col)
                required_lookback_by_feature[col] = 1
                continue
            transformable_cols.append(col)
            required_lookback_by_feature[col] = int(
                cfg.get("feature_required_lookback_hours_by_feature", {}).get(
                    col, default_warmup
                )
                if isinstance(cfg.get("feature_required_lookback_hours_by_feature"), dict)
                else default_warmup
            )
            vals = _scoped_values(feats[col], fit_scope)
            if vals.size == 0:
                med = mean = 0.0
                std = iqr = 1.0
                lo = hi = None
            else:
                med = float(np.nanmedian(vals))
                mean = float(np.nanmean(vals))
                std = float(np.nanstd(vals))
                q25, q75 = np.nanquantile(vals, [0.25, 0.75])
                iqr = float(q75 - q25)
                lo, hi = np.nanquantile(vals, clip_q)
                lo = float(lo)
                hi = float(hi)
            if not np.isfinite(std) or abs(std) < 1e-12:
                std = 1.0
            if not np.isfinite(iqr) or abs(iqr) < 1e-12:
                iqr = 1.0
            stats[col] = {
                "median": med if np.isfinite(med) else 0.0,
                "mean": mean if np.isfinite(mean) else 0.0,
                "std": std,
                "iqr": iqr,
                "clip_lo": lo,
                "clip_hi": hi,
            }
            transformed_cols.append(col)

        contract = cls(
            schema_version=1,
            run_id=str(run_id),
            market_mode=str(cfg.get("market_mode", "unknown")),
            fit_scope=dict(fit_scope or {}),
            raw_feature_cols=raw_cols,
            transformed_feature_cols=transformed_cols,
            passthrough_cols=sorted(set(passthrough_cols)),
            never_transform_cols=sorted(never),
            transformable_cols=sorted(set(transformable_cols)),
            required_warmup_hours=default_warmup,
            required_lookback_hours_by_feature=required_lookback_by_feature,
            allow_missing_features={col: False for col in raw_cols},
            fillable_features={col: False for col in raw_cols},
            per_column_stats=stats,
            transform_config={
                "kind": kind,
                "clip_quantiles": [float(clip_q[0]), float(clip_q[1])],
                "impute": impute,
            },
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        contract.contract_hash = compute_contract_hash(asdict(contract))
        tprint(
            "FeatureTransformContract fit: "
            f"cols={len(raw_cols)} transformed={len(raw_cols) - len(contract.passthrough_cols)} "
            f"passthrough={len(contract.passthrough_cols)} hash={contract.contract_hash}"
        )
        return contract

    def validate_no_fit_required(self) -> None:
        missing = [
            c
            for c in self.raw_feature_cols
            if c not in self.per_column_stats and c not in set(self.never_transform_cols)
        ]
        if missing:
            raise RuntimeError(
                "Feature transform contract is missing stats for columns: "
                + ", ".join(missing[:20])
            )

    def _transform_series_or_frame(self, name: str, obj: pd.DataFrame) -> pd.DataFrame:
        out = obj.copy()
        stats = self.per_column_stats.get(name, {})
        if name in set(self.passthrough_cols) or _is_passthrough_col(
            name, set(self.passthrough_cols), set(self.never_transform_cols)
        ):
            return out.astype(np.float32, copy=False)
        kind = str(self.transform_config.get("kind", "robust")).lower()
        if kind == "none":
            return out.astype(np.float32, copy=False)
        vals = out.astype(np.float64)
        fill = stats.get("median", 0.0)
        if not np.isfinite(float(fill or 0.0)):
            fill = 0.0
        vals = vals.replace([np.inf, -np.inf], np.nan).fillna(float(fill or 0.0))
        lo = stats.get("clip_lo")
        hi = stats.get("clip_hi")
        if lo is not None and hi is not None and np.isfinite(float(lo)) and np.isfinite(float(hi)):
            vals = vals.clip(lower=float(lo), upper=float(hi))
        if kind == "standard":
            center = float(stats.get("mean", 0.0) or 0.0)
            scale = float(stats.get("std", 1.0) or 1.0)
        else:
            center = float(stats.get("median", 0.0) or 0.0)
            scale = float(stats.get("iqr", 1.0) or 1.0)
        if not np.isfinite(scale) or abs(scale) < 1e-12:
            scale = 1.0
        vals = (vals - center) / scale
        return vals.astype(np.float32, copy=False)

    def transform_panels(self, feats: dict[str, pd.DataFrame], strict: bool = True) -> dict[str, pd.DataFrame]:
        self.validate_no_fit_required()
        missing = [c for c in self.raw_feature_cols if c not in feats]
        if missing and strict:
            raise KeyError(
                "Feature transform input missing required raw columns: "
                + ", ".join(missing[:20])
            )
        out: dict[str, pd.DataFrame] = {}
        for col in self.raw_feature_cols:
            df = feats.get(col)
            if isinstance(df, pd.DataFrame):
                out[col] = self._transform_series_or_frame(col, df)
        if not strict:
            for col, df in feats.items():
                if col not in out and isinstance(df, pd.DataFrame):
                    out[str(col)] = df.astype(np.float32, copy=False)
        return out

    def transform_matrix(
        self,
        X_raw: pd.DataFrame,
        strict: bool = True,
        require_finite: bool = True,
    ) -> pd.DataFrame:
        self.validate_no_fit_required()
        missing = [c for c in self.raw_feature_cols if c not in X_raw.columns]
        if missing and strict:
            raise KeyError(
                "Feature transform matrix missing required raw columns: "
                + ", ".join(missing[:20])
            )
        matrix = X_raw.reindex(columns=self.raw_feature_cols)
        if strict and require_finite:
            try:
                raw_values = matrix.astype(np.float32, copy=False)
            except Exception as exc:
                raise ValueError(
                    "Feature transform matrix contains non-numeric contracted raw columns"
                ) from exc
            nonfinite = [
                str(col)
                for col in raw_values.columns
                if not np.isfinite(raw_values[col].to_numpy(dtype=np.float32, copy=False)).all()
            ]
            if nonfinite:
                raise ValueError(
                    "Feature transform matrix contains non-finite contracted raw columns: "
                    + ", ".join(nonfinite[:20])
                )
        kind = str(self.transform_config.get("kind", "robust")).lower()
        if kind == "none":
            return matrix.reindex(columns=self.transformed_feature_cols).astype(
                np.float32, copy=False
            )

        arr = matrix.to_numpy(dtype=np.float64, copy=True)
        n_cols = len(self.raw_feature_cols)
        centers = np.zeros(n_cols, dtype=np.float64)
        scales = np.ones(n_cols, dtype=np.float64)
        fills = np.zeros(n_cols, dtype=np.float64)
        lows = np.full(n_cols, -np.inf, dtype=np.float64)
        highs = np.full(n_cols, np.inf, dtype=np.float64)
        passthrough_mask = np.zeros(n_cols, dtype=bool)
        passthrough = set(self.passthrough_cols)
        never = set(self.never_transform_cols)

        for j, col in enumerate(self.raw_feature_cols):
            stats = self.per_column_stats.get(col, {})
            if col in passthrough or _is_passthrough_col(col, passthrough, never):
                passthrough_mask[j] = True
                continue
            fill = float(stats.get("median", 0.0) or 0.0)
            fills[j] = fill if np.isfinite(fill) else 0.0
            lo = stats.get("clip_lo")
            hi = stats.get("clip_hi")
            if lo is not None and np.isfinite(float(lo)):
                lows[j] = float(lo)
            if hi is not None and np.isfinite(float(hi)):
                highs[j] = float(hi)
            if kind == "standard":
                center = float(stats.get("mean", 0.0) or 0.0)
                scale = float(stats.get("std", 1.0) or 1.0)
            else:
                center = float(stats.get("median", 0.0) or 0.0)
                scale = float(stats.get("iqr", 1.0) or 1.0)
            centers[j] = center if np.isfinite(center) else 0.0
            scales[j] = scale if np.isfinite(scale) and abs(scale) >= 1e-12 else 1.0

        transform_mask = ~passthrough_mask
        if transform_mask.any():
            sub = arr[:, transform_mask]
            sub[~np.isfinite(sub)] = np.nan
            fill_vals = fills[transform_mask]
            nan_rows, nan_cols = np.where(np.isnan(sub))
            if nan_rows.size:
                sub[nan_rows, nan_cols] = fill_vals[nan_cols]
            sub = np.minimum(np.maximum(sub, lows[transform_mask]), highs[transform_mask])
            sub = (sub - centers[transform_mask]) / scales[transform_mask]
            arr[:, transform_mask] = sub

        out = pd.DataFrame(arr, index=matrix.index, columns=self.raw_feature_cols)
        return out.reindex(columns=self.transformed_feature_cols).astype(
            np.float32, copy=False
        )


def _contract_paths(data_root: str | Path, run_id: str) -> tuple[Path, Path, Path]:
    root = Path(data_root) / "artifacts" / str(run_id) / "features"
    return root, root / "feature_transform_contract.joblib", root / "feature_transform_manifest.json"


def save_feature_transform_contract(contract: FeatureTransformContract, data_root: str | Path, run_id: str) -> dict[str, Any]:
    root, contract_path, manifest_path = _contract_paths(data_root, run_id)
    root.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix="feature_transform_contract.", suffix=".joblib", dir=str(root))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        joblib.dump(contract, tmp_path)
        tmp_path.replace(contract_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    manifest = {
        "schema_version": 1,
        "run_id": str(run_id),
        "market_mode": contract.market_mode,
        "feature_view_default": "transformed",
        "raw_feature_dir": str(Path(data_root) / "features_raw" / str(run_id)),
        "transformed_feature_dir": str(Path(data_root) / "features" / str(run_id)),
        "contract_path": str(contract_path),
        "contract_hash": contract.contract_hash,
        "fit_scope": contract.fit_scope,
        "transform_config": contract.transform_config,
        "raw_feature_cols": contract.raw_feature_cols,
        "transformed_feature_cols": contract.transformed_feature_cols,
        "passthrough_cols": contract.passthrough_cols,
        "transformable_cols": contract.transformable_cols,
        "required_warmup_hours": contract.required_warmup_hours,
        "required_lookback_hours_by_feature": contract.required_lookback_hours_by_feature,
    }
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True))
    tmp_manifest.replace(manifest_path)
    tprint(f"Saved feature transform contract: {contract.contract_hash} -> {contract_path}")
    return manifest


def load_feature_transform_manifest(data_root: str | Path, run_id: str) -> dict[str, Any] | None:
    _, _, manifest_path = _contract_paths(data_root, run_id)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def load_feature_transform_contract(data_root: str | Path, run_id: str) -> tuple[FeatureTransformContract, dict[str, Any]]:
    _, contract_path, _ = _contract_paths(data_root, run_id)
    manifest = load_feature_transform_manifest(data_root, run_id)
    if not contract_path.exists():
        raise FileNotFoundError(f"Feature transform contract not found: {contract_path}")
    contract = joblib.load(contract_path)
    if not isinstance(contract, FeatureTransformContract):
        raise TypeError(f"Invalid feature transform contract object: {type(contract)}")
    return contract, (manifest or {})


def validate_feature_transform_contract(
    *,
    contract: FeatureTransformContract,
    manifest: dict[str, Any] | None,
    required_feature_keys: list[str] | set[str] | tuple[str, ...] | None,
    cfg: dict[str, Any] | None = None,
) -> None:
    if not contract.contract_hash:
        raise RuntimeError("Feature transform contract has no contract_hash")
    if manifest and manifest.get("contract_hash") != contract.contract_hash:
        raise RuntimeError(
            f"Feature transform manifest hash mismatch: {manifest.get('contract_hash')} != {contract.contract_hash}"
        )
    if cfg and cfg.get("market_mode") and contract.market_mode not in {"unknown", str(cfg.get("market_mode"))}:
        raise RuntimeError(
            f"Feature transform market_mode mismatch: {contract.market_mode} != {cfg.get('market_mode')}"
        )
    required = {
        str(k)
        for k in (required_feature_keys or [])
        if isinstance(k, str) and k and not k.startswith("__") and not k.startswith("pred_")
    }
    def _is_post_transform_model_feature(name: str) -> bool:
        # These are deterministic model-input columns built after the transformed
        # feature view is loaded. They are not raw columns owned by this contract.
        if name in {"G_VOL", "G_TREND"}:
            return True
        return bool(re.match(r"^.+_G_(?:VOL|TREND)_[01]$", name))

    missing = sorted(
        k
        for k in required.difference(contract.transformed_feature_cols).difference(
            contract.passthrough_cols
        )
        if not _is_post_transform_model_feature(k)
    )
    if missing:
        raise RuntimeError(
            "Feature transform contract does not cover required model features: "
            + ", ".join(missing[:30])
            + (" ..." if len(missing) > 30 else "")
        )
    contract.validate_no_fit_required()
