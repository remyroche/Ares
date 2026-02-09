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
from extreme_price_movements.model_scoring import meta_objective_huber, ic_cross_sectional, topk_mask


class MetaModel:
    def __init__(self, strategy_name=None):
        tprint(f"Entering function: __init__ in meta_model.py")
        self.strategy_name = strategy_name
        self.model = Ridge(alpha=1.0, fit_intercept=True)
        self.feature_names = CFG.get("meta_feature_keys", [])
        self.transformer = CausalFeatureTransformer(winsor_qt=0.02, roll_window=24 * 30)
        self.scaler = StandardScaler()
        self.selected_features = None
        self.oof_probs = None

    def prepare_meta_features(self, preds, feats_df, pred_col_name="pred_logit"):
        meta_data = pd.DataFrame(index=feats_df.index)
        eps = 1e-4
        p = np.clip(preds, eps, 1 - eps)
        meta_data[pred_col_name] = logit(p)
        meta_data = pd.concat([meta_data, feats_df], axis=1)
        return meta_data.fillna(0.0)

    def _meta_assess(self, y_true, y_pred, groups=None):
        ic = ic_cross_sectional(y_pred, y_true, groups=groups)
        m = topk_mask(y_pred, 0.10, groups=groups)
        top = float(np.mean(y_true[m])) if np.any(m) else 0.0
        base = float(np.mean(y_true))
        return {"MetaIC": float(ic), "TopKNetRet": top, "TopKUplift": float(top - base)}

    def fit(self, X_meta, y, sample_weight=None):
        tprint(f"Entering function: fit in meta_model.py")

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
            max_features_pct=0.5,
        )

        self.selected_features = sel_res.selected_features
        tprint(f"MetaModel: Selected {len(self.selected_features)} features.")
        X_sel = X_meta[self.selected_features]

        alphas = [0.01, 0.1, 0.3, 1.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]
        tprint(f"MetaModel: Using alpha grid: {alphas}")

        pkf = PurgedKFold(n_splits=3, purge=5, embargo=2)
        X_arr = X_sel.values
        y_arr = y
        results = []

        for alpha in alphas:
            losses = []
            for train_idx, val_idx in pkf.split(X_arr):
                X_train, X_val = X_arr[train_idx], X_arr[val_idx]
                y_train, y_val = y_arr[train_idx], y_arr[val_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)

                m = Ridge(alpha=alpha, fit_intercept=True)
                if sample_weight is not None:
                    w_train = np.asarray(sample_weight)[train_idx]
                    m.fit(X_train_s, y_train, sample_weight=w_train)
                else:
                    m.fit(X_train_s, y_train)
                preds = m.predict(X_val_s)
                losses.append(meta_objective_huber(y_val, preds, delta=1.0))

            results.append({"alpha": alpha, "huber": float(np.mean(losses))})

        res_df = pd.DataFrame(results)
        best_row = res_df.loc[res_df["huber"].idxmin()]
        best_alpha = float(best_row["alpha"])
        tprint(f"MetaModel Grid Search: Best Alpha={best_alpha}, Huber={best_row['huber']:.6f}")

        self.oof_probs = np.zeros(len(y), dtype=np.float32)
        for train_idx, val_idx in pkf.split(X_arr):
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train = y_arr[train_idx]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            m = Ridge(alpha=best_alpha, fit_intercept=True)
            if sample_weight is not None:
                w_train = np.asarray(sample_weight)[train_idx]
                m.fit(X_train_s, y_train, sample_weight=w_train)
            else:
                m.fit(X_train_s, y_train)
            self.oof_probs[val_idx] = m.predict(X_val_s)

        assess = self._meta_assess(y_arr, self.oof_probs)
        tprint(f"MetaModel OOF assess: IC={assess['MetaIC']:.4f} Top10Ret={assess['TopKNetRet']:.6f} Uplift={assess['TopKUplift']:.6f}")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_sel)
        self.model = Ridge(alpha=best_alpha, fit_intercept=True)
        if sample_weight is not None:
            self.model.fit(X_scaled, y, sample_weight=np.asarray(sample_weight))
        else:
            self.model.fit(X_scaled, y)
        return self

    def predict(self, X_meta):
        if self.selected_features is not None:
            cols = [c for c in self.selected_features if c in X_meta.columns]
            if len(cols) < len(self.selected_features):
                missing = set(self.selected_features) - set(cols)
                tprint(f"WARNING: MetaModel.predict missing {len(missing)} features: {list(missing)[:5]}")
            X_meta = X_meta[cols]

        X_scaled = self.scaler.transform(X_meta)
        preds = self.model.predict(X_scaled)
        return preds
