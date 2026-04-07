import numpy as np
from extreme_price_movements.utils import tprint

class DynamicSoftLabels:
    def __init__(self, mfe, mae, t_mfe, t_mae, h, atr_1h):
        self.mfe = np.asarray(mfe, dtype=float)
        self.mae = np.asarray(mae, dtype=float)
        self.t_mfe = np.asarray(t_mfe, dtype=float)
        self.t_mae = np.asarray(t_mae, dtype=float)
        self.h = float(h)
        atr_1h = np.asarray(atr_1h, dtype=float)
        self.atr_1h = np.where(np.isfinite(atr_1h) & (atr_1h > 1e-9), atr_1h, 0.005)
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

            if not (rate_TP >= 0.05 and rate_SL >= 0.05 and 0.10 <= rate_TO <= 0.80):
                continue
            if not (1.0 <= tp_m / sl_m <= 3.0 and sl_m < tp_m):
                continue
            if np.sum(y_g == 2) == 0 or np.sum(y_g == 0) == 0:
                continue

            mfe_tp = np.mean(self.mfe[y_g == 2])
            mfe_sl = np.mean(self.mfe[y_g == 0])
            mae_tp = np.mean(self.mae[y_g == 2])
            mae_sl = np.mean(self.mae[y_g == 0])

            if not (mfe_tp > 1.2 * mfe_sl) or not (mae_tp < 0.85 * mae_sl):
                continue
            retained.append((tp_m, sl_m))

        if len(retained) == 0:
            raise ValueError(f"No valid geometries retained for horizon={self.h}")
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
        return out_soft

    def to_numpy(self):
        retained = self.get_retained_geometries()
        return self.build_soft_labels_with(retained)
