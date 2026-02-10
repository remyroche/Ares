import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
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
        meta_data[pred_col_name] = np.clip(logit(p), -4.0, 4.0)
        meta_data = pd.concat([meta_data, feats_df], axis=1)
        return meta_data.fillna(0.0)

    def _meta_assess(self, y_true, y_pred, groups=None):
        ic = ic_cross_sectional(y_pred, y_true, groups=groups)
        m = topk_mask(y_pred, 0.10, groups=groups)
        top = float(np.mean(y_true[m])) if np.any(m) else 0.0
        base = float(np.mean(y_true))
        return {"MetaIC": float(ic), "TopKNetRet": top, "TopKUplift": float(top - base)}

    def _cv_evaluate_model(self, model_fn, X_arr, y_arr, sample_weight, groups, pkf, scale=True):
        """Evaluate a model factory via PurgedKFold, returning (mean_neg_ic, oof_preds)."""
        oof = np.zeros(len(y_arr), dtype=np.float32)
        losses = []
        for train_idx, val_idx in pkf.split(X_arr):
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]
            if scale:
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)
            else:
                X_train_s, X_val_s = X_train, X_val
            m = model_fn()
            if sample_weight is not None:
                m.fit(X_train_s, y_train, sample_weight=np.asarray(sample_weight)[train_idx])
            else:
                m.fit(X_train_s, y_train)
            preds = m.predict(X_val_s)
            oof[val_idx] = preds
            g_val = np.asarray(groups)[val_idx] if groups is not None else None
            losses.append(-ic_cross_sectional(preds, y_val, groups=g_val))
        return float(np.mean(losses)), oof

    def fit(self, X_meta, y, sample_weight=None, groups=None):
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
            min_features=10,
            max_features_pct=0.5,
        )

        self.selected_features = sel_res.selected_features
        tprint(f"MetaModel: Selected {len(self.selected_features)} features.")
        X_sel = X_meta[self.selected_features]

        pkf = PurgedKFold(n_splits=3, purge=5, embargo=2)
        X_arr = X_sel.values
        y_arr = y

        # --- Ridge grid search ---
        alphas = [0.01, 0.1, 0.3, 1.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
        ridge_results = []
        for alpha in alphas:
            neg_ic, _ = self._cv_evaluate_model(
                lambda a=alpha: Ridge(alpha=a, fit_intercept=True),
                X_arr, y_arr, sample_weight, groups, pkf, scale=True
            )
            ridge_results.append({"alpha": alpha, "loss": neg_ic})
        ridge_df = pd.DataFrame(ridge_results)
        best_ridge = ridge_df.loc[ridge_df["loss"].idxmin()]
        best_ridge_alpha = float(best_ridge["alpha"])
        best_ridge_loss = float(best_ridge["loss"])
        tprint(f"MetaModel Ridge: Best Alpha={best_ridge_alpha}, NegIC={best_ridge_loss:.6f}")

        # --- ExtraTrees evaluation ---
        et_loss, _ = self._cv_evaluate_model(
            lambda: ExtraTreesRegressor(
                n_estimators=200, max_depth=4, min_samples_leaf=20,
                max_features=0.7, random_state=42, n_jobs=-1
            ),
            X_arr, y_arr, sample_weight, groups, pkf, scale=False
        )
        tprint(f"MetaModel ExtraTrees: NegIC={et_loss:.6f}")

        # --- Pick winner ---
        use_et = et_loss < best_ridge_loss
        winner = "ExtraTrees" if use_et else "Ridge"
        tprint(f"MetaModel: Winner={winner} (Ridge NegIC={best_ridge_loss:.4f}, ET NegIC={et_loss:.4f})")

        # --- Generate OOF with winner ---
        if use_et:
            _, self.oof_probs = self._cv_evaluate_model(
                lambda: ExtraTreesRegressor(
                    n_estimators=200, max_depth=4, min_samples_leaf=20,
                    max_features=0.7, random_state=42, n_jobs=-1
                ),
                X_arr, y_arr, sample_weight, groups, pkf, scale=False
            )
        else:
            _, self.oof_probs = self._cv_evaluate_model(
                lambda a=best_ridge_alpha: Ridge(alpha=a, fit_intercept=True),
                X_arr, y_arr, sample_weight, groups, pkf, scale=True
            )

        assess = self._meta_assess(y_arr, self.oof_probs, groups=groups)
        tprint(f"MetaModel OOF assess: IC_target={assess['MetaIC']:.4f} Top10Ret={assess['TopKNetRet']:.6f} Uplift={assess['TopKUplift']:.6f}")

        # --- Final fit on all data ---
        if use_et:
            self.scaler = None  # ET doesn't need scaling
            self.model = ExtraTreesRegressor(
                n_estimators=200, max_depth=4, min_samples_leaf=20,
                max_features=0.7, random_state=42, n_jobs=-1
            )
        else:
            self.scaler = StandardScaler()
            X_sel = pd.DataFrame(self.scaler.fit_transform(X_sel),
                                 columns=self.selected_features, index=X_meta.index)
            self.model = Ridge(alpha=best_ridge_alpha, fit_intercept=True)

        if sample_weight is not None:
            self.model.fit(X_sel.values if isinstance(X_sel, pd.DataFrame) else X_sel,
                           y, sample_weight=np.asarray(sample_weight))
        else:
            self.model.fit(X_sel.values if isinstance(X_sel, pd.DataFrame) else X_sel, y)
        self._model_type = winner
        return self

    def predict(self, X_meta):
        if self.selected_features is not None:
            cols = [c for c in self.selected_features if c in X_meta.columns]
            if len(cols) < len(self.selected_features):
                missing = set(self.selected_features) - set(cols)
                tprint(f"WARNING: MetaModel.predict missing {len(missing)} features: {list(missing)[:5]}")
            X_meta = X_meta[cols]

        if self.scaler is not None:
            X_input = self.scaler.transform(X_meta)
        else:
            X_input = X_meta.values if isinstance(X_meta, pd.DataFrame) else X_meta
        preds = self.model.predict(X_input)
        return preds
