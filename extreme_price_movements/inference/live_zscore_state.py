from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from extreme_price_movements import fast_funcs as ff
from extreme_price_movements.feature_family_registry import (
    FeatureFamily,
    get_feature_family,
)


LIVE_ZSCORE_STATE_VERSION = 1
DEFAULT_LIVE_ZSCORE_STATE_FILE = "causal_zscore_state.npz"
LIVE_RAW_ROLLING_STATE_VERSION = 1
DEFAULT_LIVE_RAW_ROLLING_STATE_FILE = "raw_rolling_state.npz"


def live_zscore_state_path(data_root: str | Path, run_id: str) -> Path:
    return (
        Path(data_root)
        / "artifacts"
        / str(run_id)
        / "live_state"
        / DEFAULT_LIVE_ZSCORE_STATE_FILE
    )


def live_raw_rolling_state_path(data_root: str | Path, run_id: str) -> Path:
    return (
        Path(data_root)
        / "artifacts"
        / str(run_id)
        / "live_state"
        / DEFAULT_LIVE_RAW_ROLLING_STATE_FILE
    )


class RawRollingFeatureState:
    """Stateful latest-row rolling primitives for live append-only features."""

    VERSION = LIVE_RAW_ROLLING_STATE_VERSION
    SUPPORTED_OPS = {"sum", "mean", "std", "max", "min"}

    def __init__(
        self,
        *,
        op: str,
        name: str,
        symbols,
        window: int,
        last_timestamp: str | None = None,
    ):
        op = str(op)
        if op not in self.SUPPORTED_OPS:
            raise ValueError(f"Unsupported raw rolling op: {op}")
        self.op = op
        self.name = str(name)
        self.symbols = [str(s) for s in symbols]
        self.window = int(window)
        self.last_timestamp = last_timestamp
        if self.window <= 0:
            raise ValueError("RawRollingFeatureState window must be positive")
        n_symbols = len(self.symbols)
        self.buffer = np.empty((n_symbols, self.window), dtype=np.float32)
        self.valid = np.zeros((n_symbols, self.window), dtype=np.bool_)
        self.ptr = np.zeros(n_symbols, dtype=np.int32)
        self.count = np.zeros(n_symbols, dtype=np.int32)
        self.sum = np.zeros(n_symbols, dtype=np.float64)
        self.sum_sq = np.zeros(n_symbols, dtype=np.float64)

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "op": self.op,
            "name": self.name,
            "symbol_order": list(self.symbols),
            "window": int(self.window),
            "last_timestamp": self.last_timestamp,
        }

    def metadata_matches(
        self,
        *,
        op: str | None = None,
        name: str | None = None,
        symbols=None,
        window: int | None = None,
    ) -> bool:
        if op is not None and str(op) != self.op:
            return False
        if name is not None and str(name) != self.name:
            return False
        if symbols is not None and [str(s) for s in symbols] != self.symbols:
            return False
        if window is not None and int(window) != self.window:
            return False
        return True

    def update(self, raw_values, timestamp: Any | None = None) -> np.ndarray:
        arr = np.asarray(raw_values, dtype=np.float32).reshape(-1)
        if arr.shape[0] != len(self.symbols):
            raise ValueError(
                f"Rolling input {self.name!r} has {arr.shape[0]} symbols; "
                f"expected {len(self.symbols)}"
            )
        out = np.full(len(self.symbols), np.nan, dtype=np.float32)
        for j, val in enumerate(arr):
            slot = int(self.ptr[j])
            if bool(self.valid[j, slot]):
                old = float(self.buffer[j, slot])
                if np.isfinite(old):
                    self.sum[j] -= old
                    self.sum_sq[j] -= old * old
                    self.count[j] = max(int(self.count[j]) - 1, 0)
            if np.isfinite(float(val)):
                self.buffer[j, slot] = float(val)
                self.valid[j, slot] = True
                self.sum[j] += float(val)
                self.sum_sq[j] += float(val) * float(val)
                self.count[j] += 1
            else:
                self.buffer[j, slot] = np.nan
                self.valid[j, slot] = False
            self.ptr[j] = (slot + 1) % self.window
            n = int(self.count[j])
            if n <= 0:
                continue
            if self.op == "sum":
                out[j] = np.float32(self.sum[j])
            elif self.op == "mean":
                out[j] = np.float32(self.sum[j] / max(n, 1))
            elif self.op == "max":
                vals = self.buffer[j, self.valid[j, :]]
                if vals.size:
                    out[j] = np.float32(np.nanmax(vals))
            elif self.op == "min":
                vals = self.buffer[j, self.valid[j, :]]
                if vals.size:
                    out[j] = np.float32(np.nanmin(vals))
            elif n > 1:
                var = (self.sum_sq[j] - (self.sum[j] * self.sum[j] / n)) / (n - 1)
                out[j] = np.float32(np.sqrt(max(var, 0.0)))
        if timestamp is not None:
            self.last_timestamp = str(timestamp)
        return out

    def seed_from_frame(self, values, index) -> None:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("seed_from_frame expects a 2D array")
        if arr.shape[1] != len(self.symbols):
            raise ValueError(
                f"Rolling seed has {arr.shape[1]} symbols; expected {len(self.symbols)}"
            )
        start = max(0, arr.shape[0] - self.window)
        for pos in range(start, arr.shape[0]):
            ts = None
            if index is not None:
                ts = pd_timestamp_iso(index[pos])
            self.update(arr[pos, :], timestamp=ts)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(self.metadata(), sort_keys=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        np.savez_compressed(
            tmp_path,
            metadata=np.asarray(metadata),
            buffer=self.buffer,
            valid=self.valid,
            ptr=self.ptr,
            count=self.count,
            sum=self.sum,
            sum_sq=self.sum_sq,
        )
        npz_tmp = tmp_path
        if not npz_tmp.exists() and tmp_path.with_suffix(tmp_path.suffix + ".npz").exists():
            npz_tmp = tmp_path.with_suffix(tmp_path.suffix + ".npz")
        npz_tmp.replace(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        op: str | None = None,
        name: str | None = None,
        symbols=None,
        window: int | None = None,
    ) -> "RawRollingFeatureState | None":
        path = Path(path)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata"].item()))
                if int(meta.get("version", -1)) != cls.VERSION:
                    return None
                state = cls(
                    op=str(meta.get("op")),
                    name=str(meta.get("name")),
                    symbols=meta.get("symbol_order", []),
                    window=int(meta.get("window")),
                    last_timestamp=meta.get("last_timestamp"),
                )
                if not state.metadata_matches(
                    op=op,
                    name=name,
                    symbols=symbols,
                    window=window,
                ):
                    return None
                state.buffer[...] = data["buffer"]
                state.valid[...] = data["valid"]
                state.ptr[...] = data["ptr"]
                state.count[...] = data["count"]
                state.sum[...] = data["sum"]
                state.sum_sq[...] = data["sum_sq"]
                return state
        except Exception:
            return None


def pd_timestamp_iso(value: Any) -> str:
    try:
        import pandas as pd

        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


class RollingZScoreState:
    """Stateful latest-row equivalent of CausalFeatureTransformer's z-score path."""

    VERSION = LIVE_ZSCORE_STATE_VERSION

    def __init__(
        self,
        feature_keys,
        symbols,
        window,
        sigma_k,
        winsor_qt=0.02,
        last_timestamp: str | None = None,
    ):
        self.feature_keys = [str(k) for k in feature_keys]
        self.symbols = [str(s) for s in symbols]
        self.window = int(window)
        self.sigma_k = float(sigma_k)
        self.winsor_qt = float(winsor_qt)
        self.last_timestamp = last_timestamp
        if self.window <= 0:
            raise ValueError("RollingZScoreState window must be positive")

        self._feature_index = {k: i for i, k in enumerate(self.feature_keys)}
        n_features = len(self.feature_keys)
        n_symbols = len(self.symbols)
        self.buffer = np.empty(
            (n_features, n_symbols, self.window), dtype=np.float32
        )
        self.valid = np.zeros(
            (n_features, n_symbols, self.window), dtype=np.bool_
        )
        self.ptr = np.zeros((n_features, n_symbols), dtype=np.int32)
        self.count = np.zeros((n_features, n_symbols), dtype=np.int32)
        self.K = np.zeros((n_features, n_symbols), dtype=np.float64)
        self.K_set = np.zeros((n_features, n_symbols), dtype=np.bool_)
        self.sum_d = np.zeros((n_features, n_symbols), dtype=np.float64)
        self.sum_d_sq = np.zeros((n_features, n_symbols), dtype=np.float64)
        self._active_mask = np.asarray(
            [
                get_feature_family(key) == FeatureFamily.RISK_NORMALIZED_CONTINUOUS
                for key in self.feature_keys
            ],
            dtype=np.bool_,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "feature_order": list(self.feature_keys),
            "symbol_order": list(self.symbols),
            "roll_window": int(self.window),
            "winsor_qt": float(self.winsor_qt),
            "sigma_k": float(self.sigma_k),
            "code_cache_version": f"live_zscore_state_v{self.VERSION}",
            "last_timestamp": self.last_timestamp,
        }

    def metadata_matches(
        self,
        *,
        feature_keys=None,
        symbols=None,
        window=None,
        winsor_qt=None,
        sigma_k=None,
    ) -> bool:
        if feature_keys is not None and [str(k) for k in feature_keys] != self.feature_keys:
            return False
        if symbols is not None and [str(s) for s in symbols] != self.symbols:
            return False
        if window is not None and int(window) != self.window:
            return False
        if winsor_qt is not None and float(winsor_qt) != self.winsor_qt:
            return False
        if sigma_k is not None and float(sigma_k) != self.sigma_k:
            return False
        return True

    def update(
        self,
        raw_feature_values_by_key: Mapping[str, Any],
        timestamp: Any | None = None,
    ) -> dict[str, np.ndarray]:
        latest = np.zeros((len(self.feature_keys), len(self.symbols)), dtype=np.float32)
        present = np.zeros(len(self.feature_keys), dtype=np.bool_)
        passthrough: dict[str, np.ndarray] = {}

        for key, raw_values in raw_feature_values_by_key.items():
            key = str(key)
            idx = self._feature_index.get(key)
            if idx is None:
                continue
            family = get_feature_family(key)
            arr = np.asarray(raw_values, dtype=np.float32).reshape(-1)
            if arr.shape[0] != len(self.symbols):
                raise ValueError(
                    f"Feature {key!r} latest row has {arr.shape[0]} values; "
                    f"expected {len(self.symbols)}"
                )
            present[idx] = True
            if family == FeatureFamily.RISK_NORMALIZED_CONTINUOUS:
                latest[idx, :] = np.arcsinh(arr).astype(np.float32, copy=False)
            elif family == FeatureFamily.CATEGORICAL_OR_BUCKETED:
                passthrough[key] = arr.astype(np.float32, copy=True)
                latest[idx, :] = arr
            else:
                cleaned = np.nan_to_num(
                    arr.astype(np.float32, copy=True),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if family in {
                    FeatureFamily.ALREADY_STANDARDIZED,
                    FeatureFamily.BOUNDED_GEOMETRY,
                }:
                    np.clip(cleaned, -self.sigma_k, self.sigma_k, out=cleaned)
                passthrough[key] = cleaned
                latest[idx, :] = cleaned

        active = self._active_mask & present
        out = np.empty_like(latest, dtype=np.float32)
        ff._numba_live_zscore_update(
            latest,
            active,
            self.buffer,
            self.valid,
            self.ptr,
            self.count,
            self.K,
            self.K_set,
            self.sum_d,
            self.sum_d_sq,
            out,
        )

        result: dict[str, np.ndarray] = {}
        for key in raw_feature_values_by_key:
            key = str(key)
            idx = self._feature_index.get(key)
            if idx is None:
                continue
            if self._active_mask[idx]:
                row = out[idx, :].astype(np.float32, copy=True)
                np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                np.clip(row, -self.sigma_k, self.sigma_k, out=row)
                result[key] = row
            else:
                result[key] = passthrough[key].astype(np.float32, copy=False)

        if timestamp is not None:
            self.last_timestamp = str(timestamp)
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(self.metadata(), sort_keys=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        np.savez_compressed(
            tmp_path,
            metadata=np.asarray(metadata),
            buffer=self.buffer,
            valid=self.valid,
            ptr=self.ptr,
            count=self.count,
            K=self.K,
            K_set=self.K_set,
            sum_d=self.sum_d,
            sum_d_sq=self.sum_d_sq,
        )
        npz_tmp = tmp_path
        if not npz_tmp.exists() and tmp_path.with_suffix(tmp_path.suffix + ".npz").exists():
            npz_tmp = tmp_path.with_suffix(tmp_path.suffix + ".npz")
        npz_tmp.replace(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        feature_keys=None,
        symbols=None,
        window=None,
        winsor_qt=None,
        sigma_k=None,
    ) -> "RollingZScoreState | None":
        path = Path(path)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata"].item()))
                if int(meta.get("version", -1)) != cls.VERSION:
                    return None
                state = cls(
                    meta.get("feature_order", []),
                    meta.get("symbol_order", []),
                    int(meta.get("roll_window")),
                    float(meta.get("sigma_k")),
                    winsor_qt=float(meta.get("winsor_qt", 0.02)),
                    last_timestamp=meta.get("last_timestamp"),
                )
                if not state.metadata_matches(
                    feature_keys=feature_keys,
                    symbols=symbols,
                    window=window,
                    winsor_qt=winsor_qt,
                    sigma_k=sigma_k,
                ):
                    return None
                state.buffer[...] = data["buffer"]
                state.valid[...] = data["valid"]
                state.ptr[...] = data["ptr"]
                state.count[...] = data["count"]
                state.K[...] = data["K"]
                state.K_set[...] = data["K_set"]
                state.sum_d[...] = data["sum_d"]
                state.sum_d_sq[...] = data["sum_d_sq"]
                return state
        except Exception:
            return None
