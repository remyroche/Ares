from __future__ import annotations

import hashlib
import json
import sqlite3
import time
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
LIVE_CAUSAL_TRANSFORM_STATE_CONTAINER_VERSION = 1
DEFAULT_LIVE_CAUSAL_TRANSFORM_STATE_CONTAINER_SUFFIX = ".container.sqlite"
LIVE_RAW_ROLLING_STATE_VERSION = 1
DEFAULT_LIVE_RAW_ROLLING_STATE_FILE = "raw_rolling_state.npz"
LIVE_RAW_ROLLING_STATE_CONTAINER_VERSION = 1
DEFAULT_LIVE_RAW_ROLLING_STATE_CONTAINER_SUFFIX = ".container.sqlite"


class RawRollingStateContainerBusy(RuntimeError):
    """Another process owns the raw rolling state snapshot."""


class CausalTransformStateContainerBusy(RuntimeError):
    """Another process owns the causal transform state snapshot."""


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


def raw_rolling_state_container_path(path: str | Path) -> Path:
    """Return the appendable container path associated with a legacy state root."""
    target = Path(path)
    if target.suffix:
        return target.with_suffix(DEFAULT_LIVE_RAW_ROLLING_STATE_CONTAINER_SUFFIX)
    return target.with_name(target.name + DEFAULT_LIVE_RAW_ROLLING_STATE_CONTAINER_SUFFIX)


def causal_transform_state_container_path(path: str | Path) -> Path:
    """Return the per-feature SQLite container for a legacy z-score state root."""
    target = Path(path)
    if target.suffix:
        return target.with_suffix(DEFAULT_LIVE_CAUSAL_TRANSFORM_STATE_CONTAINER_SUFFIX)
    return target.with_name(
        target.name + DEFAULT_LIVE_CAUSAL_TRANSFORM_STATE_CONTAINER_SUFFIX
    )


class RawRollingFeatureState:
    """Stateful latest-row rolling primitives for live append-only features."""

    VERSION = LIVE_RAW_ROLLING_STATE_VERSION
    SUPPORTED_OPS = {"sum", "mean", "std", "max", "min"}
    _OP_CODES = {"sum": 0, "mean": 1, "std": 2, "max": 3, "min": 4}

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
        ff._numba_raw_rolling_state_update(
            self.buffer,
            self.valid,
            self.ptr,
            self.count,
            self.sum,
            self.sum_sq,
            arr,
            self._OP_CODES[self.op],
            out,
        )
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


class RawRollingStateContainer:
    """Atomic, namespaced storage for raw rolling primitive states.

    A feature cycle can update many independent mean/std/sum/min/max worksets.
    Storing them as individual compressed NPZ files is correct but incurs a
    disproportionate amount of filesystem metadata and compression work.  This
    container keeps each exact state namespace as one SQLite row, so state
    reads remain lazy and all updates from a compute pass commit atomically.
    """

    VERSION = LIVE_RAW_ROLLING_STATE_CONTAINER_VERSION
    _TABLE = "raw_rolling_states"

    def __init__(
        self,
        path: str | Path,
        connection: sqlite3.Connection,
        *,
        transaction_active: bool,
    ):
        self.path = Path(path)
        self._connection = connection
        self._dirty_keys: set[str] = set()
        self._transaction_active = bool(transaction_active)
        self._closed = False

    @classmethod
    def _connect(cls, target: Path, *, timeout_seconds: float) -> sqlite3.Connection:
        connection = sqlite3.connect(str(target), timeout=float(timeout_seconds))
        try:
            connection.execute(
                f"PRAGMA busy_timeout={max(0, int(timeout_seconds * 1000))}"
            )
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {cls._TABLE} (
                    state_key TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    buffer BLOB NOT NULL,
                    valid BLOB NOT NULL,
                    ptr BLOB NOT NULL,
                    count BLOB NOT NULL,
                    sum BLOB NOT NULL,
                    sum_sq BLOB NOT NULL,
                    updated_at_ns INTEGER NOT NULL
                )
                """
            )
            connection.commit()
            return connection
        except Exception:
            connection.close()
            raise

    @staticmethod
    def _quarantine_corrupt_container(target: Path) -> Path:
        """Preserve a corrupt state file and return its forensic backup path."""
        backup = target.with_name(
            f"{target.name}.corrupt.{time.time_ns()}"
        )
        target.replace(backup)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(target) + suffix)
            if sidecar.exists():
                sidecar.replace(backup.with_name(f"{backup.name}{suffix}"))
        return backup

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        lock_timeout_seconds: float = 30.0,
    ) -> "RawRollingStateContainer":
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        timeout_seconds = max(float(lock_timeout_seconds), 0.0)
        try:
            connection = cls._connect(target, timeout_seconds=timeout_seconds)
        except sqlite3.OperationalError as exc:
            # A live process owns the coherent snapshot.  Do not mistake its
            # lock for corruption or rename its active state file.
            if any(token in str(exc).lower() for token in ("locked", "busy")):
                raise RawRollingStateContainerBusy(
                    f"raw rolling state container is busy: {target}"
                ) from exc
            raise
        except sqlite3.DatabaseError:
            # Preserve evidence for diagnosis, then create a clean container.
            # Feature code falls back to legacy/vectorized state while this
            # recovery happens, so a bad cache never changes feature math.
            cls._quarantine_corrupt_container(target)
            connection = cls._connect(target, timeout_seconds=timeout_seconds)
        try:
            # Hold one writer reservation from all state reads through the
            # final commit.  This prevents two overlapping feature passes from
            # reading the same old namespace and silently overwriting each
            # other's append-only updates.
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            connection.close()
            if any(token in str(exc).lower() for token in ("locked", "busy")):
                raise RawRollingStateContainerBusy(
                    f"raw rolling state container is busy: {target}"
                ) from exc
            raise
        except Exception:
            connection.close()
            raise
        return cls(target, connection, transaction_active=True)

    @staticmethod
    def _array_blob(values: np.ndarray) -> sqlite3.Binary:
        return sqlite3.Binary(np.ascontiguousarray(values).tobytes())

    @staticmethod
    def _array_from_blob(
        blob: bytes,
        *,
        dtype,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if len(blob) != expected:
            raise ValueError("raw rolling state blob has an invalid size")
        return np.frombuffer(blob, dtype=dtype).copy().reshape(shape)

    @staticmethod
    def _state_arrays_are_valid(state: RawRollingFeatureState) -> bool:
        if np.any(state.ptr < 0) or np.any(state.ptr >= state.window):
            return False
        if np.any(state.count < 0) or np.any(state.count > state.window):
            return False
        valid_counts = state.valid.sum(axis=1, dtype=np.int64)
        if not np.array_equal(valid_counts, state.count.astype(np.int64, copy=False)):
            return False
        valid_values = state.buffer[state.valid]
        if not np.isfinite(valid_values).all():
            return False
        expected_sum = np.where(state.valid, state.buffer, 0.0).sum(
            axis=1, dtype=np.float64
        )
        expected_sum_sq = np.where(
            state.valid, state.buffer.astype(np.float64) ** 2, 0.0
        ).sum(axis=1, dtype=np.float64)
        return bool(
            np.isfinite(state.sum).all()
            and np.isfinite(state.sum_sq).all()
            and np.all(state.sum_sq >= 0.0)
            and np.allclose(state.sum, expected_sum, rtol=1e-6, atol=1e-9)
            and np.allclose(state.sum_sq, expected_sum_sq, rtol=1e-6, atol=1e-7)
        )

    def _begin_write_transaction(self) -> None:
        if not self._transaction_active:
            self._connection.execute("BEGIN IMMEDIATE")
            self._transaction_active = True

    def get(
        self,
        state_key: str,
        *,
        op: str,
        name: str,
        symbols,
        window: int,
    ) -> RawRollingFeatureState | None:
        if self._closed:
            return None
        try:
            row = self._connection.execute(
                f"""
                SELECT version, metadata_json, buffer, valid, ptr, count, sum, sum_sq
                FROM {self._TABLE} WHERE state_key = ?
                """,
                (str(state_key),),
            ).fetchone()
            if row is None or int(row[0]) != self.VERSION:
                return None
            metadata = json.loads(str(row[1]))
            if (
                int(metadata.get("version", -1)) != RawRollingFeatureState.VERSION
                or str(metadata.get("container_state_key") or "") != str(state_key)
            ):
                return None
            state = RawRollingFeatureState(
                op=str(metadata.get("op")),
                name=str(metadata.get("name")),
                symbols=metadata.get("symbol_order", []),
                window=int(metadata.get("window")),
                last_timestamp=metadata.get("last_timestamp"),
            )
            if not state.metadata_matches(
                op=op,
                name=name,
                symbols=symbols,
                window=window,
            ):
                return None
            n_symbols = len(state.symbols)
            state.buffer[...] = self._array_from_blob(
                row[2], dtype=np.float32, shape=(n_symbols, state.window)
            )
            state.valid[...] = self._array_from_blob(
                row[3], dtype=np.bool_, shape=(n_symbols, state.window)
            )
            state.ptr[...] = self._array_from_blob(
                row[4], dtype=np.int32, shape=(n_symbols,)
            )
            state.count[...] = self._array_from_blob(
                row[5], dtype=np.int32, shape=(n_symbols,)
            )
            state.sum[...] = self._array_from_blob(
                row[6], dtype=np.float64, shape=(n_symbols,)
            )
            state.sum_sq[...] = self._array_from_blob(
                row[7], dtype=np.float64, shape=(n_symbols,)
            )
            if not self._state_arrays_are_valid(state):
                return None
            return state
        except Exception:
            return None

    def put(self, state_key: str, state: RawRollingFeatureState) -> None:
        if self._closed:
            raise RuntimeError("RawRollingStateContainer is closed")
        if not self._state_arrays_are_valid(state):
            raise ValueError("refusing to persist invalid raw rolling state")
        self._begin_write_transaction()
        metadata_payload = state.metadata()
        metadata_payload["container_state_key"] = str(state_key)
        metadata = json.dumps(metadata_payload, sort_keys=True)
        self._connection.execute(
            f"""
            INSERT INTO {self._TABLE} (
                state_key, version, metadata_json, buffer, valid, ptr, count, sum, sum_sq,
                updated_at_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(state_key) DO UPDATE SET
                version = excluded.version,
                metadata_json = excluded.metadata_json,
                buffer = excluded.buffer,
                valid = excluded.valid,
                ptr = excluded.ptr,
                count = excluded.count,
                sum = excluded.sum,
                sum_sq = excluded.sum_sq,
                updated_at_ns = excluded.updated_at_ns
            """,
            (
                str(state_key),
                self.VERSION,
                metadata,
                self._array_blob(state.buffer),
                self._array_blob(state.valid),
                self._array_blob(state.ptr),
                self._array_blob(state.count),
                self._array_blob(state.sum),
                self._array_blob(state.sum_sq),
                time.time_ns(),
            ),
        )
        self._dirty_keys.add(str(state_key))

    def flush(self) -> int:
        if self._closed or not self._transaction_active:
            return 0
        self._connection.commit()
        count = len(self._dirty_keys)
        self._dirty_keys.clear()
        self._transaction_active = False
        return count

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._connection.close()
            self._closed = True

    def abort(self) -> None:
        """Discard unflushed updates after an interrupted feature pass."""
        if self._closed:
            return
        try:
            if self._transaction_active:
                self._connection.rollback()
        finally:
            self._connection.close()
            self._closed = True
            self._transaction_active = False

    def __del__(self) -> None:
        # Feature generation normally closes the container explicitly.  This
        # keeps a failed compute pass from holding a SQLite write lock until
        # process teardown without committing a partial state update.
        try:
            self.abort()
        except Exception:
            pass


class CausalTransformStateContainer:
    """Atomic per-feature storage for causal rolling z-score state.

    A z-score workset is only a vectorization choice.  Persisting one state per
    workset makes overlapping feature requests fork their histories.  This
    container instead keys each feature by a canonical transform namespace and
    reconstructs a contiguous ``RollingZScoreState`` for each requested workset.
    The SQLite writer reservation covers the read/update/write lifecycle, so
    append-only passes cannot lose an overlapping feature update.
    """

    VERSION = LIVE_CAUSAL_TRANSFORM_STATE_CONTAINER_VERSION
    _TABLE = "causal_transform_feature_states"

    def __init__(
        self,
        path: str | Path,
        connection: sqlite3.Connection,
        *,
        transaction_active: bool,
    ):
        self.path = Path(path)
        self._connection = connection
        self._dirty_keys: set[tuple[str, str]] = set()
        self._transaction_active = bool(transaction_active)
        self._closed = False

    @staticmethod
    def _symbols_hash(symbols) -> str:
        payload = json.dumps(
            [str(symbol) for symbol in symbols],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    @classmethod
    def namespace_key(
        cls,
        *,
        scope: str,
        transform_contract: str,
        symbols,
        window: int,
        winsor_qt: float,
        sigma_k: float,
    ) -> str:
        """Return the canonical namespace; requested feature names are excluded."""
        symbol_order = [str(symbol) for symbol in symbols]
        if not symbol_order or len(set(symbol_order)) != len(symbol_order):
            raise ValueError("causal transform state symbols must be unique and nonempty")
        if int(window) <= 0:
            raise ValueError("causal transform state window must be positive")
        if not np.isfinite(float(winsor_qt)) or not 0.0 <= float(winsor_qt) < 0.5:
            raise ValueError("causal transform state winsor_qt must be in [0, 0.5)")
        if not np.isfinite(float(sigma_k)) or float(sigma_k) <= 0.0:
            raise ValueError("causal transform state sigma_k must be positive")
        payload = {
            "container_version": cls.VERSION,
            "scope": str(scope),
            "transform_contract": str(transform_contract),
            "symbol_hash": cls._symbols_hash(symbol_order),
            "window": int(window),
            "winsor_qt": float(winsor_qt),
            "sigma_k": float(sigma_k),
        }
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _namespace_metadata(
        cls,
        namespace: str,
        *,
        symbols,
        window: int,
        winsor_qt: float,
        sigma_k: float,
    ) -> dict[str, Any]:
        try:
            metadata = json.loads(str(namespace))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "namespace must be created by CausalTransformStateContainer.namespace_key"
            ) from exc
        expected = {
            "container_version": cls.VERSION,
            "symbol_hash": cls._symbols_hash(symbols),
            "window": int(window),
            "winsor_qt": float(winsor_qt),
            "sigma_k": float(sigma_k),
        }
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {
                "container_version",
                "scope",
                "transform_contract",
                "symbol_hash",
                "window",
                "winsor_qt",
                "sigma_k",
            }
            or any(metadata.get(key) != value for key, value in expected.items())
            or not str(metadata.get("scope", ""))
            or not str(metadata.get("transform_contract", ""))
        ):
            raise ValueError("causal transform state namespace metadata is incompatible")
        return metadata

    @classmethod
    def _connect(cls, target: Path, *, timeout_seconds: float) -> sqlite3.Connection:
        connection = sqlite3.connect(str(target), timeout=float(timeout_seconds))
        try:
            connection.execute(
                f"PRAGMA busy_timeout={max(0, int(timeout_seconds * 1000))}"
            )
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {cls._TABLE} (
                    namespace TEXT NOT NULL,
                    feature_key TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    buffer BLOB NOT NULL,
                    valid BLOB NOT NULL,
                    ptr BLOB NOT NULL,
                    count BLOB NOT NULL,
                    k BLOB NOT NULL,
                    k_set BLOB NOT NULL,
                    sum_d BLOB NOT NULL,
                    sum_d_sq BLOB NOT NULL,
                    updated_at_ns INTEGER NOT NULL,
                    PRIMARY KEY (namespace, feature_key)
                )
                """
            )
            connection.commit()
            return connection
        except Exception:
            connection.close()
            raise

    @staticmethod
    def _quarantine_corrupt_container(target: Path) -> Path:
        backup = target.with_name(f"{target.name}.corrupt.{time.time_ns()}")
        target.replace(backup)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(target) + suffix)
            if sidecar.exists():
                sidecar.replace(backup.with_name(f"{backup.name}{suffix}"))
        return backup

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        lock_timeout_seconds: float = 30.0,
    ) -> "CausalTransformStateContainer":
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        timeout_seconds = max(float(lock_timeout_seconds), 0.0)
        try:
            connection = cls._connect(target, timeout_seconds=timeout_seconds)
        except sqlite3.OperationalError as exc:
            if any(token in str(exc).lower() for token in ("locked", "busy")):
                raise CausalTransformStateContainerBusy(
                    f"causal transform state container is busy: {target}"
                ) from exc
            raise
        except sqlite3.DatabaseError:
            cls._quarantine_corrupt_container(target)
            connection = cls._connect(target, timeout_seconds=timeout_seconds)
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            connection.close()
            if any(token in str(exc).lower() for token in ("locked", "busy")):
                raise CausalTransformStateContainerBusy(
                    f"causal transform state container is busy: {target}"
                ) from exc
            raise
        except Exception:
            connection.close()
            raise
        return cls(target, connection, transaction_active=True)

    @staticmethod
    def _array_blob(values: np.ndarray) -> sqlite3.Binary:
        return sqlite3.Binary(np.ascontiguousarray(values).tobytes())

    @staticmethod
    def _array_from_blob(blob: bytes, *, dtype, shape: tuple[int, ...]) -> np.ndarray:
        expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if len(blob) != expected:
            raise ValueError("causal transform state blob has an invalid size")
        return np.frombuffer(blob, dtype=dtype).copy().reshape(shape)

    @staticmethod
    def _feature_arrays_are_valid(state: "RollingZScoreState", index: int) -> bool:
        if np.any(state.ptr[index] < 0) or np.any(state.ptr[index] >= state.window):
            return False
        if np.any(state.count[index] < 0) or np.any(state.count[index] > state.window):
            return False
        valid = state.valid[index]
        if not np.array_equal(
            valid.sum(axis=1, dtype=np.int64),
            state.count[index].astype(np.int64, copy=False),
        ):
            return False
        if not np.isfinite(state.buffer[index][valid]).all():
            return False
        return bool(
            np.isfinite(state.K[index]).all()
            and np.isfinite(state.sum_d[index]).all()
            and np.isfinite(state.sum_d_sq[index]).all()
        )

    @staticmethod
    def _requested_feature_keys(feature_keys) -> list[str]:
        keys = [str(key) for key in feature_keys]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("causal transform state feature keys must be unique and nonempty")
        return keys

    def _begin_write_transaction(self) -> None:
        if not self._transaction_active:
            self._connection.execute("BEGIN IMMEDIATE")
            self._transaction_active = True

    @staticmethod
    def _set_aggregate_last_timestamp(state: "RollingZScoreState") -> None:
        timestamps = [state.feature_last_timestamps.get(key) for key in state.feature_keys]
        if timestamps and all(timestamp is not None for timestamp in timestamps):
            first = str(timestamps[0])
            state.last_timestamp = first if all(str(value) == first for value in timestamps) else None
        else:
            state.last_timestamp = None

    def get_many(
        self,
        namespace: str,
        *,
        feature_keys,
        symbols,
        window: int,
        winsor_qt: float,
        sigma_k: float,
    ) -> "RollingZScoreState | None":
        """Load requested features into one vectorized state.

        Missing feature rows remain empty while present rows retain their own
        timestamps in ``feature_last_timestamps``.  A mixed timestamp workset
        has ``last_timestamp=None`` deliberately: callers must advance each
        feature from its own causal cursor rather than replay a shared row.
        """
        if self._closed:
            return None
        requested = self._requested_feature_keys(feature_keys)
        symbol_order = [str(symbol) for symbol in symbols]
        namespace_metadata = self._namespace_metadata(
            namespace,
            symbols=symbol_order,
            window=window,
            winsor_qt=winsor_qt,
            sigma_k=sigma_k,
        )
        placeholders = ",".join("?" for _ in requested)
        try:
            rows = self._connection.execute(
                f"""
                SELECT feature_key, version, metadata_json, buffer, valid, ptr, count,
                       k, k_set, sum_d, sum_d_sq
                FROM {self._TABLE}
                WHERE namespace = ? AND feature_key IN ({placeholders})
                """,
                (str(namespace), *requested),
            ).fetchall()
            if not rows:
                return None
            state = RollingZScoreState(
                requested,
                symbol_order,
                int(window),
                float(sigma_k),
                winsor_qt=float(winsor_qt),
            )
            row_by_key = {str(row[0]): row for row in rows}
            for feature_index, feature_key in enumerate(requested):
                row = row_by_key.get(feature_key)
                if row is None:
                    continue
                if int(row[1]) != self.VERSION:
                    raise ValueError("causal transform state container version is incompatible")
                metadata = json.loads(str(row[2]))
                if (
                    metadata.get("namespace") != namespace_metadata
                    or str(metadata.get("feature_key")) != feature_key
                    or int(metadata.get("state_version", -1))
                    != RollingZScoreState.VERSION
                ):
                    raise ValueError("causal transform state row metadata is incompatible")
                n_symbols = len(symbol_order)
                state.buffer[feature_index] = self._array_from_blob(
                    row[3], dtype=np.float32, shape=(n_symbols, int(window))
                )
                state.valid[feature_index] = self._array_from_blob(
                    row[4], dtype=np.bool_, shape=(n_symbols, int(window))
                )
                state.ptr[feature_index] = self._array_from_blob(
                    row[5], dtype=np.int32, shape=(n_symbols,)
                )
                state.count[feature_index] = self._array_from_blob(
                    row[6], dtype=np.int32, shape=(n_symbols,)
                )
                state.K[feature_index] = self._array_from_blob(
                    row[7], dtype=np.float64, shape=(n_symbols,)
                )
                state.K_set[feature_index] = self._array_from_blob(
                    row[8], dtype=np.bool_, shape=(n_symbols,)
                )
                state.sum_d[feature_index] = self._array_from_blob(
                    row[9], dtype=np.float64, shape=(n_symbols,)
                )
                state.sum_d_sq[feature_index] = self._array_from_blob(
                    row[10], dtype=np.float64, shape=(n_symbols,)
                )
                if not self._feature_arrays_are_valid(state, feature_index):
                    raise ValueError("causal transform state row arrays are invalid")
                last_timestamp = metadata.get("last_timestamp")
                state.feature_last_timestamps[feature_key] = (
                    None if last_timestamp is None else str(last_timestamp)
                )
            self._set_aggregate_last_timestamp(state)
            return state
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("causal transform state row could not be loaded") from exc

    def put_many(
        self,
        namespace: str,
        state: "RollingZScoreState",
        *,
        feature_keys=None,
    ) -> None:
        """Atomically persist only the supplied feature rows; never delete others."""
        if self._closed:
            raise RuntimeError("CausalTransformStateContainer is closed")
        selected = self._requested_feature_keys(
            state.feature_keys if feature_keys is None else feature_keys
        )
        if any(key not in state._feature_index for key in selected):
            raise ValueError("cannot persist a feature absent from the z-score state")
        namespace_metadata = self._namespace_metadata(
            namespace,
            symbols=state.symbols,
            window=state.window,
            winsor_qt=state.winsor_qt,
            sigma_k=state.sigma_k,
        )
        self._begin_write_transaction()
        for feature_key in selected:
            feature_index = state._feature_index[feature_key]
            if not self._feature_arrays_are_valid(state, feature_index):
                raise ValueError(
                    f"refusing to persist invalid causal transform state for {feature_key!r}"
                )
            metadata = json.dumps(
                {
                    "namespace": namespace_metadata,
                    "feature_key": feature_key,
                    "state_version": RollingZScoreState.VERSION,
                    "last_timestamp": state.feature_last_timestamps.get(
                        feature_key, state.last_timestamp
                    ),
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            self._connection.execute(
                f"""
                INSERT INTO {self._TABLE} (
                    namespace, feature_key, version, metadata_json, buffer, valid, ptr,
                    count, k, k_set, sum_d, sum_d_sq, updated_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, feature_key) DO UPDATE SET
                    version = excluded.version,
                    metadata_json = excluded.metadata_json,
                    buffer = excluded.buffer,
                    valid = excluded.valid,
                    ptr = excluded.ptr,
                    count = excluded.count,
                    k = excluded.k,
                    k_set = excluded.k_set,
                    sum_d = excluded.sum_d,
                    sum_d_sq = excluded.sum_d_sq,
                    updated_at_ns = excluded.updated_at_ns
                """,
                (
                    str(namespace),
                    feature_key,
                    self.VERSION,
                    metadata,
                    self._array_blob(state.buffer[feature_index]),
                    self._array_blob(state.valid[feature_index]),
                    self._array_blob(state.ptr[feature_index]),
                    self._array_blob(state.count[feature_index]),
                    self._array_blob(state.K[feature_index]),
                    self._array_blob(state.K_set[feature_index]),
                    self._array_blob(state.sum_d[feature_index]),
                    self._array_blob(state.sum_d_sq[feature_index]),
                    time.time_ns(),
                ),
            )
            self._dirty_keys.add((str(namespace), feature_key))

    def flush(self) -> int:
        if self._closed or not self._transaction_active:
            return 0
        self._connection.commit()
        count = len(self._dirty_keys)
        self._dirty_keys.clear()
        self._transaction_active = False
        return count

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._connection.close()
            self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        try:
            if self._transaction_active:
                self._connection.rollback()
        finally:
            self._connection.close()
            self._closed = True
            self._transaction_active = False

    def __del__(self) -> None:
        try:
            self.abort()
        except Exception:
            pass


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
        self.feature_last_timestamps: dict[str, str | None] = {
            key: None if last_timestamp is None else str(last_timestamp)
            for key in self.feature_keys
        }
        if self.window <= 0:
            raise ValueError("RollingZScoreState window must be positive")

        self._feature_index = {k: i for i, k in enumerate(self.feature_keys)}
        n_features = len(self.feature_keys)
        n_symbols = len(self.symbols)
        # Invalid slots are ignored numerically, but deterministic contents
        # matter for reproducible serialization and state-hash comparisons.
        self.buffer = np.zeros(
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
            "feature_last_timestamps": dict(self.feature_last_timestamps),
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
            timestamp_text = str(timestamp)
            self.last_timestamp = timestamp_text
            for key in raw_feature_values_by_key:
                key = str(key)
                if key in self._feature_index:
                    self.feature_last_timestamps[key] = timestamp_text
        return result

    def feature_last_timestamp(self, feature_key: str) -> str | None:
        """Return the causal cursor for one feature in a partial workset."""
        return self.feature_last_timestamps.get(str(feature_key))

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
                stored_feature_timestamps = meta.get("feature_last_timestamps")
                if isinstance(stored_feature_timestamps, Mapping):
                    state.feature_last_timestamps = {
                        key: (
                            None
                            if stored_feature_timestamps.get(key) is None
                            else str(stored_feature_timestamps.get(key))
                        )
                        for key in state.feature_keys
                    }
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
