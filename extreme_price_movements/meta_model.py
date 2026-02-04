import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit
from scipy.special import logit
from extreme_price_movements.utils import tprint
from extreme_price_movements.feature_transforms import CausalFeatureTransformer
from extreme_price_movements.feature_selection_extreme_events import mdi_feature_selection_v3

class MetaModel:
    def __init__(self):
        tprint(f"Entering function: __init__ in meta_model.py")
        self.model = Ridge(alpha=1.0)
        self.feature_names = [
            "pred_tf_logit", "pred_mr_logit",
            "realized_vol", "vol_z", "log_volume",
            "norm_momentum", "dist_ma_z",
            "atr_slope", "dist_vwap_norm", "mom_accel"
        ]
        self.transformer = CausalFeatureTransformer(winsor_qt=0.02, roll_window=24*30)
        self.selected_features = None

    def prepare_meta_features(self, preds_tf, preds_mr, feats_df):
        """
        Constructs X_meta.
        Applies logit to predictions.
        Applies CausalTransform to other features.
        """
        tprint(f"Entering function: prepare_meta_features in meta_model.py")
        tprint(f"prepare_meta_features: Input feats_df shape: {feats_df.shape}")
        meta_data = pd.DataFrame(index=feats_df.index)

        eps = 1e-4
        p_tf = np.clip(preds_tf, eps, 1 - eps)
        p_mr = np.clip(preds_mr, eps, 1 - eps)

        meta_data["pred_tf_logit"] = logit(p_tf)
        meta_data["pred_mr_logit"] = logit(p_mr)
        tprint("prepare_meta_features: Logit transformation complete.")

        # Mapping
        meta_data["realized_vol"] = feats_df.get("a_rv24", 0.0)
        meta_data["vol_z"] = feats_df.get("a_volz", 0.0)
        meta_data["log_volume"] = feats_df.get("a_volz", 0.0)
        meta_data["norm_momentum"] = feats_df.get("a_rsi", 0.0)
        meta_data["dist_ma_z"] = feats_df.get("dist_ema_fast", 0.0)

        meta_data["atr_slope"] = feats_df.get("atr_slope", 0.0)
        meta_data["dist_vwap_norm"] = feats_df.get("dist_vwap_norm", 0.0)
        meta_data["mom_accel"] = feats_df.get("momentum_accel", 0.0)

        tprint(f"prepare_meta_features: Returning meta_data with shape: {meta_data[self.feature_names].shape}")
        return meta_data[self.feature_names].fillna(0.0)

    def _calc_metrics(self, y_true, y_pred):
        # PnL (Assumes y_pred is signal strength * direction, y_true is returns)
        # If y_pred is return prediction:
        pnl_curve = np.cumsum(y_pred * np.sign(y_pred) * np.sign(y_true))
        # Wait. PnL = Position * Return.
        # Position ~ y_pred. Return = y_true.
        # So trade_returns = y_pred * y_true?
        # Or simply y_pred is the weight.

        trade_rets = y_pred * y_true
        total_pnl = np.sum(trade_rets)

        # Sortino
        mean_ret = np.mean(trade_rets)
        downside = trade_rets[trade_rets < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-9
        sortino = mean_ret / (downside_std + 1e-9)

        # MaxDD
        cum = np.cumsum(trade_rets)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = np.max(dd) if len(dd) > 0 else 0.0

        return total_pnl, sortino, max_dd

    def fit(self, X_meta, y):
        tprint(f"Entering function: fit in meta_model.py")
        tprint(f"fit: Starting with X_meta shape: {X_meta.shape}, y shape: {y.shape}")

        # 1. Feature Selection
        n_samples = len(X_meta)
        n_select = min(20, max(1, n_samples // 100))
        tprint(f"MetaModel: Running MDI selection. Target={n_select}")

        base_selector = ExtraTreesRegressor(
            n_estimators=500, # Increased per v3
            max_depth=None,   # Let v3 suggest depth
            min_samples_leaf=20,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )

        sel_res = mdi_feature_selection_v3(
            X=X_meta,
            y=y,
            base_model=base_selector,
            analysis_n_estimators=500
        )

        self.selected_features = sel_res.selected_features[:n_select]
        tprint(f"MetaModel: Selected {len(self.selected_features)} features.")
        X_sel = X_meta[self.selected_features]

        alphas = [0.1, 0.3, 0.6, 1.0, 3.0]
        tscv = TimeSeriesSplit(n_splits=3)

        results = []

        X_arr = X_sel.values
        y_arr = y

        for alpha in alphas:
            tprint(f"fit: Evaluating alpha={alpha}")
            scores = {"pnl": [], "sortino": [], "maxdd": []}

            for train_idx, val_idx in tscv.split(X_arr):
                X_train, X_val = X_arr[train_idx], X_arr[val_idx]
                y_train, y_val = y_arr[train_idx], y_arr[val_idx]

                m = Ridge(alpha=alpha)
                m.fit(X_train, y_train)
                preds = m.predict(X_val)

                p, s, d = self._calc_metrics(y_val, preds)
                scores["pnl"].append(p)
                scores["sortino"].append(s)
                scores["maxdd"].append(d)

            # Average metrics
            avg_pnl = np.mean(scores["pnl"])
            avg_sort = np.mean(scores["sortino"])
            avg_dd = np.mean(scores["maxdd"])

            results.append({
                "alpha": alpha,
                "pnl": avg_pnl,
                "sortino": avg_sort,
                "maxdd": avg_dd
            })

        tprint("fit: Grid search results computed. Calculating best alpha.")

        # Pareto Selection
        # Score = 0.6 * Rank(PnL) + 0.3 * Rank(Sortino) + 0.1 * Rank(-MaxDD)
        res_df = pd.DataFrame(results)
        res_df["r_pnl"] = res_df["pnl"].rank(pct=True)
        res_df["r_sort"] = res_df["sortino"].rank(pct=True)
        res_df["r_dd"] = (-res_df["maxdd"]).rank(pct=True) # Minimize DD -> Maximize -DD

        res_df["score"] = 0.6 * res_df["r_pnl"] + 0.3 * res_df["r_sort"] + 0.1 * res_df["r_dd"]

        best_row = res_df.loc[res_df["score"].idxmax()]
        best_alpha = best_row["alpha"]

        tprint(f"MetaModel Grid Search: Best Alpha={best_alpha}, Score={best_row['score']:.4f}")

        # Refit on full data
        self.model = Ridge(alpha=best_alpha)
        self.model.fit(X_sel, y)

        return self

    def predict(self, X_meta):
        tprint(f"Entering function: predict in meta_model.py")
        tprint(f"predict: Input X_meta shape: {X_meta.shape}")
        if self.selected_features is None:
             # If fit not called? or loaded model without selected_features?
             # Fallback to all if not set (legacy support?)
             # Or assume we must have them.
             # If X_meta has all cols, we can try to use them all, but if model trained on subset it will fail.
             # If model trained on subset, we must subset.
             pass
        else:
             X_meta = X_meta[self.selected_features]

        tprint("predict: Generating predictions.")
        return self.model.predict(X_meta)
