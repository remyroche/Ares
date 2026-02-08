import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.special import logit
from extreme_price_movements.utils import tprint
from extreme_price_movements.feature_transforms import CausalFeatureTransformer
from extreme_price_movements.feature_selection_extreme_events import mdi_feature_selection_v3
from extreme_price_movements.purged_cv import PurgedKFold
from extreme_price_movements.config import CFG

class MetaModel:
    def __init__(self, strategy_name=None):
        tprint(f"Entering function: __init__ in meta_model.py")
        self.strategy_name = strategy_name
        self.model = Ridge(alpha=1.0, fit_intercept=True)
        self.feature_names = CFG.get("meta_feature_keys", [])
        self.transformer = CausalFeatureTransformer(winsor_qt=0.02, roll_window=24*30)
        self.scaler = StandardScaler()
        self.selected_features = None
        self.oof_probs = None

    def prepare_meta_features(self, preds, feats_df, pred_col_name="pred_logit"):
        """
        Constructs X_meta.
        Applies logit to predictions.
        Combines with feats_df (which should contain the meta features).
        preds: np.array of probabilities (0..1)
        """
        meta_data = pd.DataFrame(index=feats_df.index)

        eps = 1e-4
        p = np.clip(preds, eps, 1 - eps)

        meta_data[pred_col_name] = logit(p)

        # Merge with feats_df
        # feats_df should only contain the requested keys
        # We assume feats_df is aligned

        # If feats_df has missing columns, fill 0?
        # Ideally feats_df IS X_meta from training collection

        # Merge
        meta_data = pd.concat([meta_data, feats_df], axis=1)

        # Ensure only valid numeric cols
        return meta_data.fillna(0.0)

    def _calc_metrics(self, y_true, y_pred):
        # PnL (Assumes y_pred is signal strength * direction, y_true is returns)
        # If y_pred is return prediction:
        # pnl_curve = np.cumsum(y_pred * np.sign(y_pred) * np.sign(y_true))

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

        # 1. Feature Selection (relaxed for small meta feature sets)
        n_samples = len(X_meta)
        n_select = min(20, max(1, n_samples // 100))
        tprint(f"MetaModel: Running MDI selection. Target={n_select}")

        sel_res = mdi_feature_selection_v3(
            X=X_meta,
            y=y,
            analysis_n_estimators=500,
            end_features=n_select,
            cumulative_cap=0.99,
            min_share=0.0001,
            min_features=5,
            max_features_pct=0.5
        )

        self.selected_features = sel_res.selected_features
        tprint(f"MetaModel: Selected {len(self.selected_features)} features.")
        X_sel = X_meta[self.selected_features]

        # Expanded alpha grid for all strategies (wider range for better regularization tuning)
        alphas = [0.01, 0.1, 0.3, 1.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]
        tprint(f"MetaModel: Using expanded alpha grid: {alphas}")
        
        pkf = PurgedKFold(n_splits=3, purge=5, embargo=2)

        results = []

        X_arr = X_sel.values
        y_arr = y

        for alpha in alphas:
            scores = {"pnl": [], "sortino": [], "maxdd": []}

            for train_idx, val_idx in pkf.split(X_arr):
                X_train, X_val = X_arr[train_idx], X_arr[val_idx]
                y_train, y_val = y_arr[train_idx], y_arr[val_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)

                m = Ridge(alpha=alpha, fit_intercept=True)
                m.fit(X_train_s, y_train)
                preds = m.predict(X_val_s)

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

        # Pareto Selection
        res_df = pd.DataFrame(results)
        res_df["r_pnl"] = res_df["pnl"].rank(pct=True)
        res_df["r_sort"] = res_df["sortino"].rank(pct=True)
        res_df["r_dd"] = (-res_df["maxdd"]).rank(pct=True)

        res_df["score"] = 0.6 * res_df["r_pnl"] + 0.3 * res_df["r_sort"] + 0.1 * res_df["r_dd"]

        best_row = res_df.loc[res_df["score"].idxmax()]
        best_alpha = best_row["alpha"]

        tprint(f"MetaModel Grid Search: Best Alpha={best_alpha}, Score={best_row['score']:.4f}")

        # Generate OOF predictions for the best alpha
        tprint(f"MetaModel: Generating OOF predictions for best alpha={best_alpha}...")
        self.oof_probs = np.zeros(len(y), dtype=np.float32)

        for train_idx, val_idx in pkf.split(X_arr):
             X_train, X_val = X_arr[train_idx], X_arr[val_idx]
             y_train, y_val = y_arr[train_idx], y_arr[val_idx]

             scaler = StandardScaler()
             X_train_s = scaler.fit_transform(X_train)
             X_val_s = scaler.transform(X_val)

             m = Ridge(alpha=best_alpha, fit_intercept=True)
             m.fit(X_train_s, y_train)
             preds = m.predict(X_val_s)

             self.oof_probs[val_idx] = preds

        oof_std = np.std(self.oof_probs)
        tprint(f"MetaModel OOF: mean={np.mean(self.oof_probs):.6f}, std={oof_std:.6f}")
        if oof_std < 1e-6:
            tprint("WARNING: MetaModel OOF predictions have near-zero variance — model may be degenerate.")

        # Refit on full data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_sel)
        self.model = Ridge(alpha=best_alpha, fit_intercept=True)
        self.model.fit(X_scaled, y)

        return self

    def predict(self, X_meta):
        if self.selected_features is None:
             pass
        else:
             # Ensure columns exist
             cols = [c for c in self.selected_features if c in X_meta.columns]
             if len(cols) < len(self.selected_features):
                 missing = set(self.selected_features) - set(cols)
                 tprint(f"WARNING: MetaModel.predict missing {len(missing)} features: {list(missing)[:5]}")
             X_meta = X_meta[cols]

        X_scaled = self.scaler.transform(X_meta)
        preds = self.model.predict(X_scaled)
        return preds
