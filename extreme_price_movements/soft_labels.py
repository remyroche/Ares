import numpy as np
from extreme_price_movements.utils import tprint



CLASS_ORDER = ("SL", "TO", "TP")

def convert_class_order(arr, src_order, dst_order):
    if src_order == dst_order:
        return arr
    mapping = [src_order.index(l) for l in dst_order]
    return arr[:, mapping]



class DynamicSoftLabels:
    def __init__(self, mfe, mae, t_mfe, t_mae, h, atr_1h):
        self.mfe = np.asarray(mfe, dtype=float)
        self.mae = np.asarray(mae, dtype=float)
        self.t_mfe = np.asarray(t_mfe, dtype=float)
        self.t_mae = np.asarray(t_mae, dtype=float)
        self.h = float(h)
        atr_1h = np.asarray(atr_1h, dtype=float)
        self.atr_1h = np.where(np.isfinite(atr_1h) & (atr_1h > 1e-9), atr_1h, 0.005)

        # ATR scaling assertion and comment:
        # ATR_1h is computed purely on backward-looking EWMA True Range logic in `numba_atr_no_norm` (features.py).
        # ATR_h = ATR_1h * sqrt(h) inherently preserves this strict causality.
        # No centered windows, future bars, or post-entry info leakage exists in this derivation.
        assert len(self.atr_1h) == len(self.mfe), "ATR length must match target length to guarantee causal alignment."
        self.atr_h = self.atr_1h * np.sqrt(self.h)

        self.N = len(self.mfe)
        self.ndim = 2  # To satisfy MetaClassifierModel checks

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if isinstance(idx, slice) or isinstance(idx, (list, np.ndarray)):
            # return a subset object
            return DynamicSoftLabels(
                self.mfe[idx], self.mae[idx], self.t_mfe[idx], self.t_mae[idx],
                self.h, self.atr_1h[idx]
            )
        return self

    def get_retained_geometries(self):
        candidates = []
        for sl_mult in [0.75, 1.0, 1.25]:
            for tp_mult in [0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
                candidates.append((tp_mult, sl_mult))
        candidates = list(set(candidates))

        retained = []
        reasons = {"base_rates": 0, "value_bounds": 0, "zero_samples": 0, "excursion_stats": 0}

        for tp_m, sl_m in candidates:
            TP_v = tp_m * self.atr_h
            SL_v = sl_m * self.atr_h

            _hit_tp = (self.mfe >= TP_v) & (self.t_mfe <= self.h + 1e-5)
            _hit_sl = (self.mae >= SL_v) & (self.t_mae <= self.h + 1e-5)
            _both = _hit_tp & _hit_sl

            y_g = np.ones(self.N, dtype=np.int8)
            y_g[_hit_sl & ~_hit_tp] = 0
            y_g[_hit_tp & ~_hit_sl] = 2
            y_g[_both & (self.t_mfe < self.t_mae)] = 2
            y_g[_both & (self.t_mfe >= self.t_mae)] = 0

            rate_TP = np.mean(y_g == 2)
            rate_SL = np.mean(y_g == 0)
            rate_TO = np.mean(y_g == 1)

            if not (rate_TP >= 0.02 and rate_SL >= 0.02):
                reasons["base_rates"] += 1
                continue
            if not (1.0 <= tp_m / sl_m <= 3.0 and sl_m < tp_m):
                reasons["value_bounds"] += 1
                continue
            if np.sum(y_g == 2) == 0 or np.sum(y_g == 0) == 0:
                reasons["zero_samples"] += 1
                continue

            mfe_tp = np.mean(self.mfe[y_g == 2])
            mfe_sl = np.mean(self.mfe[y_g == 0])
            mae_tp = np.mean(self.mae[y_g == 2])
            mae_sl = np.mean(self.mae[y_g == 0])

            if not (mfe_tp > 1.2 * mfe_sl) or not (mae_tp < 0.85 * mae_sl):
                reasons["excursion_stats"] += 1
                continue
            retained.append((tp_m, sl_m))

        if len(retained) == 0:
            tprint(
                f"WARNING: No valid geometries retained. Horizon={self.h}. "
                f"Total candidates={len(candidates)}. Rejections: {reasons}. "
                f"Returning empty list — soft-label head will be skipped."
            )
            return []
        if len(retained) < 3:
            tprint(f"WARNING: Retained geometries < 3 for horizon={self.h}. Retained count={len(retained)}.")

        tprint(f"Horizon {self.h} geometries: Candidates={len(candidates)}, Retained={len(retained)}. Rejections: {reasons}")
        return retained

    def build_soft_labels_with(self, retained):
        soft = np.zeros((self.N, 3), dtype=np.float64)
        for tp_m, sl_m in retained:
            TP_v = tp_m * self.atr_h
            SL_v = sl_m * self.atr_h

            _hit_tp = (self.mfe >= TP_v) & (self.t_mfe <= self.h + 1e-5)
            _hit_sl = (self.mae >= SL_v) & (self.t_mae <= self.h + 1e-5)
            _both = _hit_tp & _hit_sl

            y_g = np.ones(self.N, dtype=np.int8)
            y_g[_hit_sl & ~_hit_tp] = 0
            y_g[_hit_tp & ~_hit_sl] = 2
            y_g[_both & (self.t_mfe < self.t_mae)] = 2
            y_g[_both & (self.t_mfe >= self.t_mae)] = 0

            soft[:, 0] += (y_g == 0)
            soft[:, 1] += (y_g == 1)
            soft[:, 2] += (y_g == 2)

        soft /= len(retained)
        out_soft = np.zeros((self.N, 3), dtype=np.float64)
        # Column 0: TP, 1: SL, 2: TO
        out_soft[:, 0] = soft[:, 2]
        out_soft[:, 1] = soft[:, 0]
        out_soft[:, 2] = soft[:, 1]

        out_soft = validate_probability_simplex(out_soft, f"built_soft_labels_h{self.h}")
        summarize_soft_labels(out_soft, f"horizon_{self.h}_soft_targets")

        # Check for degeneracy
        # If one class mass > 0.95 or entropy is very low
        eps = 1e-12
        p_safe = np.clip(out_soft, eps, 1 - eps)
        entropy = -np.sum(p_safe * np.log2(p_safe), axis=1).mean()
        if entropy < 0.2:
            tprint(f"WARNING: Degenerate soft labels generated for horizon={self.h}. Entropy={entropy:.3f}")

        return out_soft

    def to_numpy(self):
        retained = self.get_retained_geometries()
        if not retained:
            return None
        return self.build_soft_labels_with(retained)

def validate_probability_simplex(arr: np.ndarray, name: str, atol: float = 1e-6) -> np.ndarray:
    """Validates and enforces probability simplex constraints."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: contains non-finite values.")

    if not np.all(arr >= -1e-8):
        raise ValueError(f"{name}: contains material negative probabilities.")

    # Fix floating point minor negative values
    if np.any(arr < 0):
        tprint(f"{name}: Correcting tiny negative probabilities.")
        arr = np.clip(arr, 0.0, None)

    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError(f"{name}: row sums do not equal 1.0 within tolerance.")

    # Re-normalize to exactly 1.0 to avoid precision accumulation
    arr = arr / np.clip(arr.sum(axis=1, keepdims=True), 1e-12, None)
    return arr

def summarize_soft_labels(arr: np.ndarray, name: str):
    """Logs target diagnostic entropy and class mass distributions."""
    mass = arr.mean(axis=0)
    eps = 1e-12
    p_safe = np.clip(arr, eps, 1 - eps)

    # Measure entropy profile
    entropies = -np.sum(p_safe * np.log2(p_safe), axis=1)
    mean_entropy = entropies.mean()
    var_entropy = entropies.var()

    tprint(f"Soft labels summary [{name}]: Entropy Mean={mean_entropy:.3f}, Var={var_entropy:.4f}, Class masses: {mass}")

    if np.any(mass < 0.01):
        tprint(f"WARNING: {name} has an almost empty class support (mass < 1%). Model might be unstable.")
    if mean_entropy < 0.2:
        tprint(f"WARNING: Overconfidence detected in {name} soft targets (Entropy={mean_entropy:.3f}). Model may become indecisive/overfit.")
    elif mean_entropy > 1.4:
        tprint(f"WARNING: Uncertainty overload detected in {name} soft targets (Entropy={mean_entropy:.3f}). Model may become indecisive.")
