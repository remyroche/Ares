from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.timestamp_contract import to_utc_timestamp, utc_isoformat
from extreme_price_movements.utils import tprint

DEFAULT_NEVER_TRANSFORM_COLS = ("__ts__", "__symbol__", "timestamp", "symbol")
DEFAULT_PASSTHROUGH_COLS = ("barrier_pct", "atr_pct_raw")
FLOAT16_CLIPPED_THEN_FLOAT32_V1 = "float16_clipped_then_float32_v1"


def ordered_names_hash(names: Sequence[str]) -> str:
    payload = json.dumps([str(name) for name in names], separators=(",", ":"))
    return "sha256:" + sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass
class FeatureSourceContract:
    """Immutable identity and semantic contract for a model feature source."""

    schema_version: str
    run_id: str
    source_root: str
    market_mode: str
    bar_frequency: str
    timestamp_semantics: str
    required_warmup_hours: int
    source_start_ts: str
    source_end_ts: str
    feature_names: list[str]
    model_feature_names: list[str]
    universe_symbols: list[str]
    symbol_file_map: dict[str, str]
    file_records: dict[str, dict[str, Any]]
    semantics: dict[str, Any]
    feature_names_hash: str = ""
    model_feature_names_hash: str = ""
    universe_hash: str = ""
    symbol_file_map_hash: str = ""
    contract_hash: str = ""
    created_at_utc: str = ""

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        source_root: str | Path,
        market_mode: str,
        feature_names: Sequence[str],
        model_feature_names: Sequence[str],
        universe_symbols: Sequence[str],
        symbol_file_map: Mapping[str, str],
        file_records: Mapping[str, Mapping[str, Any]],
        source_start_ts: Any,
        source_end_ts: Any,
        required_warmup_hours: int,
        bar_frequency: str = "1h",
        timestamp_semantics: str = "bar_close_utc",
        semantics: Mapping[str, Any] | None = None,
    ) -> "FeatureSourceContract":
        features = [str(value) for value in feature_names]
        model_features = [str(value) for value in model_feature_names]
        symbols = sorted({str(value) for value in universe_symbols})
        file_map = {
            str(key): str(value) for key, value in sorted(symbol_file_map.items())
        }
        records = {}
        for key, value in sorted(file_records.items(), key=lambda item: str(item[0])):
            record = dict(value)
            for timestamp_key in ("first_ts", "last_ts"):
                if record.get(timestamp_key) is not None:
                    record[timestamp_key] = to_utc_timestamp(
                        record[timestamp_key]
                    ).isoformat()
            records[str(key)] = record
        timestamp_semantics = str(timestamp_semantics)
        if "utc" not in timestamp_semantics.lower():
            raise ValueError("feature source timestamp semantics must be UTC")
        source_semantics = dict(semantics or {})
        source_semantics.setdefault("timezone", "UTC")
        if str(source_semantics["timezone"]).upper() != "UTC":
            raise ValueError("feature source timezone must be UTC")
        contract = cls(
            schema_version="feature_source_contract_v1",
            run_id=str(run_id),
            source_root=str(Path(source_root).resolve()),
            market_mode=str(market_mode),
            bar_frequency=str(bar_frequency),
            timestamp_semantics=timestamp_semantics,
            required_warmup_hours=max(0, int(required_warmup_hours)),
            source_start_ts=to_utc_timestamp(source_start_ts).isoformat(),
            source_end_ts=to_utc_timestamp(source_end_ts).isoformat(),
            feature_names=features,
            model_feature_names=model_features,
            universe_symbols=symbols,
            symbol_file_map=file_map,
            file_records=records,
            semantics=source_semantics,
            feature_names_hash=ordered_names_hash(features),
            model_feature_names_hash=ordered_names_hash(model_features),
            universe_hash=ordered_names_hash(symbols),
            symbol_file_map_hash=compute_contract_hash({"symbol_file_map": file_map}),
            created_at_utc=utc_isoformat(),
        )
        contract.contract_hash = compute_contract_hash(asdict(contract))
        return contract

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureSourceContract":
        fields = cls.__dataclass_fields__
        return cls(**{key: payload[key] for key in fields if key in payload})

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_seal(self) -> None:
        expected = compute_contract_hash(asdict(self))
        if not self.contract_hash or self.contract_hash != expected:
            raise ValueError(
                f"Feature source contract hash mismatch: {self.contract_hash!r} != {expected!r}"
            )
        checks = {
            "feature_names_hash": ordered_names_hash(self.feature_names),
            "model_feature_names_hash": ordered_names_hash(self.model_feature_names),
            "universe_hash": ordered_names_hash(sorted(self.universe_symbols)),
            "symbol_file_map_hash": compute_contract_hash(
                {"symbol_file_map": dict(sorted(self.symbol_file_map.items()))}
            ),
        }
        for field_name, expected_hash in checks.items():
            if getattr(self, field_name) != expected_hash:
                raise ValueError(f"Feature source {field_name} mismatch")
        if "utc" not in str(self.timestamp_semantics).lower():
            raise ValueError("Feature source timestamp semantics are not UTC")
        if str(self.semantics.get("timezone") or "").upper() != "UTC":
            raise ValueError("Feature source timezone is not UTC")
        unmapped_files = sorted(
            set(self.symbol_file_map.values()).difference(self.file_records)
        )
        if unmapped_files:
            raise ValueError(
                "Feature source symbol map references unrecorded files: "
                + ", ".join(unmapped_files[:10])
            )


@dataclass(frozen=True)
class ModelInputNumericContract:
    """Exact numeric representation presented to a fitted estimator."""

    schema_version: str = "model_input_numeric_contract_v1"
    name: str = FLOAT16_CLIPPED_THEN_FLOAT32_V1
    source_dtype: str = "float32"
    storage_dtype: str = "float16"
    prediction_dtype: str = "float32"
    clip_abs: float = float(np.finfo(np.float16).max)
    require_finite: bool = True
    feature_names_hash: str = ""
    reference_matrix_hash: str = ""

    def asdict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract_hash"] = compute_contract_hash(payload)
        return payload


def build_model_input_numeric_contract(
    feature_names: Sequence[str], *, reference_matrix_hash: str = ""
) -> ModelInputNumericContract:
    return ModelInputNumericContract(
        feature_names_hash=ordered_names_hash(feature_names),
        reference_matrix_hash=str(reference_matrix_hash or ""),
    )


def _numeric_contract_name(contract: Any) -> str:
    if isinstance(contract, ModelInputNumericContract):
        return contract.name
    if isinstance(contract, Mapping):
        return str(contract.get("name") or contract.get("contract") or "")
    return str(contract or "")


def apply_model_input_numeric_contract(
    frame: pd.DataFrame,
    contract: str | Mapping[str, Any] | ModelInputNumericContract,
    *,
    require_finite: bool | None = None,
) -> pd.DataFrame:
    """Reproduce the fitted matrix's clip -> float16 -> float32 round trip."""
    name = _numeric_contract_name(contract).strip()
    if not name:
        return frame
    if name != FLOAT16_CLIPPED_THEN_FLOAT32_V1:
        raise ValueError(f"Unsupported model input numeric contract: {name}")
    if isinstance(contract, ModelInputNumericContract):
        payload: Mapping[str, Any] = contract.asdict()
    elif isinstance(contract, Mapping):
        payload = contract
    else:
        payload = {}
    if payload:
        contract_hash = str(payload.get("contract_hash") or "")
        if contract_hash and contract_hash != compute_contract_hash(dict(payload)):
            raise ValueError("Model input numeric contract hash mismatch")
        expected_features_hash = str(payload.get("feature_names_hash") or "")
        if not expected_features_hash:
            raise ValueError("Model input numeric feature order hash is missing")
        if expected_features_hash != ordered_names_hash(
            [str(col) for col in frame.columns]
        ):
            raise ValueError("Model input numeric feature order mismatch")
        if float(payload.get("clip_abs", np.nan)) != float(np.finfo(np.float16).max):
            raise ValueError("Model input numeric clip bound mismatch")
    if frame.empty:
        return frame.astype(np.float32, copy=False)
    values = frame.to_numpy(dtype=np.float32, copy=True)
    finite_required = (
        bool(require_finite)
        if require_finite is not None
        else bool(contract.get("require_finite", True))
        if isinstance(contract, Mapping)
        else bool(getattr(contract, "require_finite", True))
    )
    if finite_required and not np.isfinite(values).all():
        raise ValueError("Model input numeric contract received non-finite values")
    limit = np.float32(np.finfo(np.float16).max)
    np.clip(values, -limit, limit, out=values)
    values = values.astype(np.float16).astype(np.float32)
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def model_matrix_hash(
    frame: pd.DataFrame, *, row_ids: pd.DataFrame | None = None
) -> str:
    """Hash ordered row identity, column identity, dtype, shape, and exact bytes."""
    values = frame.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Cannot hash a non-finite model matrix")
    digest = sha256()
    digest.update(
        ordered_names_hash([str(col) for col in frame.columns]).encode("ascii")
    )
    digest.update(str(values.shape).encode("ascii"))
    digest.update(str(values.dtype).encode("ascii"))
    if row_ids is not None:
        if len(row_ids) != len(frame):
            raise ValueError("row_ids length does not match model matrix")
        row_payload = row_ids.astype(str).to_csv(index=False, lineterminator="\n")
    else:
        row_payload = (
            pd.Series(frame.index, dtype=object)
            .astype(str)
            .to_csv(index=False, header=False, lineterminator="\n")
        )
    digest.update(row_payload.encode("utf-8"))
    digest.update(np.ascontiguousarray(values).tobytes())
    return "sha256:" + digest.hexdigest()


def compare_model_matrices_exact(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    row_ids: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return exact parity metrics and the first row/feature divergence."""
    if list(expected.columns) != list(actual.columns):
        return {
            "ok": False,
            "error": "column_order_mismatch",
            "expected_columns_hash": ordered_names_hash(list(expected.columns)),
            "actual_columns_hash": ordered_names_hash(list(actual.columns)),
        }
    if expected.shape != actual.shape:
        return {
            "ok": False,
            "error": "shape_mismatch",
            "expected": expected.shape,
            "actual": actual.shape,
        }
    exp = expected.to_numpy(dtype=np.float32, copy=False)
    got = actual.to_numpy(dtype=np.float32, copy=False)
    equal = (exp == got) | (np.isnan(exp) & np.isnan(got))
    mismatch_positions = np.argwhere(~equal)
    finite_delta = np.abs(exp.astype(np.float64) - got.astype(np.float64))
    finite_delta[~np.isfinite(finite_delta)] = np.nan
    report: dict[str, Any] = {
        "ok": not bool(mismatch_positions.size),
        "rows": int(exp.shape[0]),
        "features": int(exp.shape[1]),
        "cells": int(exp.size),
        "mismatched_cells": int((~equal).sum()),
        "exact_cell_rate": float(equal.mean()) if equal.size else 1.0,
        "mean_abs_delta": float(np.nanmean(finite_delta)) if finite_delta.size else 0.0,
        "max_abs_delta": float(np.nanmax(finite_delta)) if finite_delta.size else 0.0,
        "expected_matrix_hash": model_matrix_hash(expected, row_ids=row_ids),
        "actual_matrix_hash": model_matrix_hash(actual, row_ids=row_ids),
    }
    if mismatch_positions.size:
        row_pos, col_pos = map(int, mismatch_positions[0])
        first = {
            "row_position": row_pos,
            "feature": str(expected.columns[col_pos]),
            "expected": float(exp[row_pos, col_pos]),
            "actual": float(got[row_pos, col_pos]),
            "abs_delta": float(finite_delta[row_pos, col_pos]),
        }
        if row_ids is not None:
            first["row_id"] = {
                str(key): _jsonable(value)
                for key, value in row_ids.iloc[row_pos].items()
            }
        else:
            first["index"] = _jsonable(expected.index[row_pos])
        report["first_divergence"] = first
    return report


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _jsonable(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
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


def _is_passthrough_col(
    name: str, passthrough_cols: set[str], never_cols: set[str]
) -> bool:
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
    vals = pd.to_numeric(scoped.stack(dropna=False), errors="coerce").to_numpy(
        dtype=np.float64
    )
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
        if fit_scope is None and not bool(
            cfg.get("feature_transform_allow_full_fit", False)
        ):
            raise RuntimeError(
                "FeatureTransformContract requires a training-scope fit_scope. "
                "Set feature_transform_allow_full_fit=true only for diagnostics/tests."
            )
        if fit_scope is None:
            tprint(
                "WARNING: fitting feature transform contract on full available feature history."
            )
            fit_scope = {
                "stage_name": "full_available",
                "symbols": [],
                "allowed_periods": [],
            }

        kind = str(cfg.get("feature_transform_kind", "robust")).lower()
        if kind not in {"none", "standard", "robust"}:
            raise ValueError(f"Unsupported feature_transform_kind={kind!r}")
        clip_q = list(cfg.get("feature_transform_clip_quantiles", [0.005, 0.995]) or [])
        if len(clip_q) != 2:
            clip_q = [0.005, 0.995]
        impute = str(cfg.get("feature_transform_impute", "median")).lower()
        passthrough = set(DEFAULT_PASSTHROUGH_COLS) | set(
            cfg.get("feature_transform_passthrough_cols", []) or []
        )
        never = set(DEFAULT_NEVER_TRANSFORM_COLS) | set(
            cfg.get("feature_transform_never_transform_cols", []) or []
        )

        raw_cols = sorted(
            str(k) for k, v in (feats or {}).items() if isinstance(v, pd.DataFrame)
        )
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
                if isinstance(
                    cfg.get("feature_required_lookback_hours_by_feature"), dict
                )
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
            if c not in self.per_column_stats
            and c not in set(self.never_transform_cols)
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
        if (
            lo is not None
            and hi is not None
            and np.isfinite(float(lo))
            and np.isfinite(float(hi))
        ):
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

    def transform_panels(
        self, feats: dict[str, pd.DataFrame], strict: bool = True
    ) -> dict[str, pd.DataFrame]:
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
                if not np.isfinite(
                    raw_values[col].to_numpy(dtype=np.float32, copy=False)
                ).all()
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
            sub = np.minimum(
                np.maximum(sub, lows[transform_mask]), highs[transform_mask]
            )
            sub = (sub - centers[transform_mask]) / scales[transform_mask]
            arr[:, transform_mask] = sub

        out = pd.DataFrame(arr, index=matrix.index, columns=self.raw_feature_cols)
        return out.reindex(columns=self.transformed_feature_cols).astype(
            np.float32, copy=False
        )


def _contract_paths(data_root: str | Path, run_id: str) -> tuple[Path, Path, Path]:
    root = Path(data_root) / "artifacts" / str(run_id) / "features"
    return (
        root,
        root / "feature_transform_contract.joblib",
        root / "feature_transform_manifest.json",
    )


def save_feature_transform_contract(
    contract: FeatureTransformContract, data_root: str | Path, run_id: str
) -> dict[str, Any]:
    root, contract_path, manifest_path = _contract_paths(data_root, run_id)
    root.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix="feature_transform_contract.", suffix=".joblib", dir=str(root)
    )
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
    tprint(
        f"Saved feature transform contract: {contract.contract_hash} -> {contract_path}"
    )
    return manifest


def load_feature_transform_manifest(
    data_root: str | Path, run_id: str
) -> dict[str, Any] | None:
    _, _, manifest_path = _contract_paths(data_root, run_id)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def load_feature_transform_contract(
    data_root: str | Path, run_id: str
) -> tuple[FeatureTransformContract, dict[str, Any]]:
    _, contract_path, _ = _contract_paths(data_root, run_id)
    manifest = load_feature_transform_manifest(data_root, run_id)
    if not contract_path.exists():
        raise FileNotFoundError(
            f"Feature transform contract not found: {contract_path}"
        )
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
    if (
        cfg
        and cfg.get("market_mode")
        and contract.market_mode not in {"unknown", str(cfg.get("market_mode"))}
    ):
        raise RuntimeError(
            f"Feature transform market_mode mismatch: {contract.market_mode} != {cfg.get('market_mode')}"
        )
    required = {
        str(k)
        for k in (required_feature_keys or [])
        if isinstance(k, str)
        and k
        and not k.startswith("__")
        and not k.startswith("pred_")
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
