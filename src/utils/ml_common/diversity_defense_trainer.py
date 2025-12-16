"""
Diversity Defense Trainer

Implements the advanced 10-model diversity ensemble logic:
- Group A (4 Models): Robust Regression (Quantile/Huber) on log1p(return)
- Group B (3 Models): Rank-Based Smooth Losses (Tanh/Fair) on gauss_rank(return)
- Group C (3 Models): Portfolio-Aware Sharpe Optimization (Baseline/Reg/Noisy)
"""

import copy
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List, Optional, Tuple, Callable
from scipy.stats import rankdata, norm

from src.utils.ml_common.objectives import (
    get_sharpe_objective,
    get_tanh_objective,
    get_huber_objective,
    get_fair_objective
)

class DiversityDefenseTrainer:
    def __init__(
        self,
        base_params: Dict[str, Any],
        n_estimators: int = 150,
        decay_rate: float = 0.0,
        random_state: int = 42
    ):
        """
        Args:
            base_params: "Vanilla" LGBM parameters from HPO.
            n_estimators: Number of boosting rounds per model.
            decay_rate: Exponential decay lambda for sample weights.
            random_state: Base seed.
        """
        self.base_params = copy.deepcopy(base_params)
        self.n_estimators = n_estimators
        self.decay_rate = decay_rate
        self.random_state = random_state
        self.models: Dict[str, lgb.Booster] = {}

        # Enforce diversity defaults
        # Groups will override these
        self.base_params.update({
            "boosting_type": "gbdt",
            "n_estimators": n_estimators,
            "verbose": -1,
            "n_jobs": 1
        })

    def _apply_target_transform(self, y: pd.Series, transform: str) -> np.ndarray:
        """
        Apply target transformation.

        Args:
            y: Target series (returns)
            transform: 'log1p', 'gauss_rank', 'clip_10', 'clip_7.5', 'none'
        """
        y_vals = y.values.astype(float)

        if transform == 'log1p':
            # Sign-preserving log1p: sign(x) * log(1 + |x|)
            # "Robust Regression on log(1 + Return)" usually implies this for financial returns
            return np.sign(y_vals) * np.log1p(np.abs(y_vals))

        elif transform == 'gauss_rank':
            # Gauss Rank: Map ranks to Gaussian distribution
            # Rank data (1..N) -> Uniform (0..1) -> Gaussian (-inf..inf)
            # Use 'average' for ties
            ranks = rankdata(y_vals, method='average')
            # Normalize to (0, 1) exclusive to avoid inf in ppf
            n = len(y_vals)
            uniform = (ranks - 0.5) / n
            return norm.ppf(uniform)

        elif transform == 'clip_10':
            # Clip +/- 10%
            return np.clip(y_vals, -0.10, 0.10)

        elif transform == 'clip_7.5':
            # Clip +/- 7.5%
            return np.clip(y_vals, -0.075, 0.075)

        else:
            return y_vals

    def _generate_diversity_configs(self) -> List[Dict[str, Any]]:
        """
        Generates the 10 specific configurations for Groups A, B, C.
        """
        configs = []

        # Extract base HPO params for scaling
        hpo_lambda_l1 = float(self.base_params.get("lambda_l1", 0.1))
        hpo_lambda_l2 = float(self.base_params.get("lambda_l2", 0.1))
        hpo_min_gain = float(self.base_params.get("min_gain_to_split", 0.0))
        hpo_min_data = int(self.base_params.get("min_data_in_leaf", 20))
        # Assume optimal huber_delta is 1.0 if not known, or derive?
        # User said "huber_delta: 0.5x HPO-optimal". Assuming HPO didn't tune it (vanilla), use 1.0 baseline.
        hpo_huber_delta = 1.0

        # --- GROUP A: Robust Regression (log1p) ---
        # A1: Quantile (Alpha 0.3)
        configs.append({
            "name": "A1_Quantile_Downside",
            "group": "A",
            "transform": "log1p",
            "objective_type": "quantile",
            "alpha": 0.30,
            "params": {
                "objective": "quantile",
                "alpha": 0.30,
                "lambda_l1": 2.0 * hpo_lambda_l1,
                "min_gain_to_split": 1.5 * hpo_min_gain,
                "bagging_fraction": 0.6,
                "bagging_freq": 1,
                "extra_trees": False
            }
        })
        # A2: Quantile (Alpha 0.5)
        configs.append({
            "name": "A2_Quantile_Median",
            "group": "A",
            "transform": "log1p",
            "objective_type": "quantile",
            "alpha": 0.50,
            "params": {
                "objective": "quantile",
                "alpha": 0.50,
                "lambda_l1": 2.0 * hpo_lambda_l1,
                "min_gain_to_split": 1.0 * hpo_min_gain,
                "bagging_fraction": 0.6,
                "bagging_freq": 1,
                "extra_trees": False
            }
        })
        # A3: Huber (Tight)
        configs.append({
            "name": "A3_Huber_Tight",
            "group": "A",
            "transform": "log1p",
            "objective_factory": lambda: get_huber_objective(delta=0.5 * hpo_huber_delta, alpha_asym=0.5),
            "params": {
                "lambda_l1": 2.0 * hpo_lambda_l1,
                "min_gain_to_split": 1.5 * hpo_min_gain,
                "bagging_fraction": 0.6,
                "bagging_freq": 1,
                "extra_trees": False
            }
        })
        # A4: Huber (Loose)
        configs.append({
            "name": "A4_Huber_Loose",
            "group": "A",
            "transform": "log1p",
            "objective_factory": lambda: get_huber_objective(delta=1.5 * hpo_huber_delta, alpha_asym=0.5),
            "params": {
                "lambda_l1": 2.0 * hpo_lambda_l1,
                "min_gain_to_split": 1.0 * hpo_min_gain,
                "bagging_fraction": 0.6,
                "bagging_freq": 1,
                "extra_trees": False
            }
        })

        # --- GROUP B: Rank-Based Smooth Losses (Gauss Rank) ---
        # B1: Tanh (Aggressive) c=0.7
        configs.append({
            "name": "B1_Tanh_Aggressive",
            "group": "B",
            "transform": "gauss_rank",
            "objective_factory": lambda: get_tanh_objective(scale=1.0/0.7), # Tanh obj takes scale. if loss is tanh(c*(p-y))?
            # Request says "tanh_c: 0.7". Usually implies Loss(r/c) or similar.
            # My get_tanh_objective implements log(cosh(scale * r)).
            # If c is scale? "aggressive" implies high sensitivity to errors?
            # Standard "tanh estimators" influence function is tanh(r).
            # Let's assume scale = 1/c for similarity with Huber?
            # If c is small (0.7), scale is large -> saturates faster -> more robust (aggressive filtering).
            "params": {
                "extra_trees": True,
                "min_gain_to_split": 0.5 * hpo_min_gain,
                "lambda_l2": 1.5 * hpo_lambda_l2,
                "bagging_fraction": 0.6,
                "bagging_freq": 1
            }
        })
        # B2: Tanh (Conservative) c=1.5
        configs.append({
            "name": "B2_Tanh_Conservative",
            "group": "B",
            "transform": "gauss_rank",
            "objective_factory": lambda: get_tanh_objective(scale=1.0/1.5),
            "params": {
                "extra_trees": True,
                "min_gain_to_split": 1.0 * hpo_min_gain,
                "lambda_l2": 2.0 * hpo_lambda_l2,
                "bagging_fraction": 0.6,
                "bagging_freq": 1
            }
        })
        # B3: Fair Loss c=1.0
        configs.append({
            "name": "B3_Fair_Loss",
            "group": "B",
            "transform": "gauss_rank",
            "objective_factory": lambda: get_fair_objective(c=1.0),
            "params": {
                "extra_trees": True,
                "min_gain_to_split": 1.0 * hpo_min_gain,
                "lambda_l2": 2.0 * hpo_lambda_l2,
                "bagging_fraction": 0.6,
                "bagging_freq": 1
            }
        })

        # --- GROUP C: Portfolio-Aware Sharpe Optimization (Clip) ---
        # Group C disables bagging (fraction=1.0, freq=0)

        # C1: Sharpe Baseline
        configs.append({
            "name": "C1_Sharpe_Baseline",
            "group": "C",
            "transform": "clip_10",
            "objective_factory": lambda: get_sharpe_objective(lambda_reg=0.1), # Default reg
            "params": {
                "max_depth": 6,
                "min_data_in_leaf": int(2.0 * hpo_min_data),
                "path_smooth": 4.0,
                "min_gain_to_split": 2.0 * hpo_min_gain,
                "bagging_fraction": 1.0,
                "bagging_freq": 0,
                "extra_trees": False
            }
        })
        # C2: Sharpe Regularized
        configs.append({
            "name": "C2_Sharpe_Regularized",
            "group": "C",
            "transform": "clip_10",
            "objective_factory": lambda: get_sharpe_objective(lambda_reg=1.0), # Strong reg
            "params": {
                "max_depth": 5,
                "min_data_in_leaf": int(3.0 * hpo_min_data),
                "path_smooth": 6.0,
                "min_gain_to_split": 3.0 * hpo_min_gain,
                "bagging_fraction": 1.0,
                "bagging_freq": 0,
                "extra_trees": False
            }
        })
        # C3: Sharpe Noisy/Stochastic
        # "Noisy" implies extra_trees=True for randomness?
        configs.append({
            "name": "C3_Sharpe_Noisy",
            "group": "C",
            "transform": "clip_7.5",
            "objective_factory": lambda: get_sharpe_objective(lambda_reg=0.1),
            "params": {
                "max_depth": 6,
                "min_data_in_leaf": int(2.0 * hpo_min_data),
                "path_smooth": 4.0,
                "min_gain_to_split": 2.0 * hpo_min_gain,
                "extra_trees": True,
                "bagging_fraction": 1.0,
                "bagging_freq": 0
            }
        })

        return configs

    def _compute_sample_weights(self, n_samples: int, decay_rate: float) -> np.ndarray:
        """
        Compute time-decay sample weights.
        """
        if decay_rate <= 0:
            return np.ones(n_samples)

        indices = np.arange(n_samples)
        age = n_samples - 1.0 - indices
        weights = np.exp(-decay_rate * age)
        weights = weights / np.mean(weights)
        return weights

    def train_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        feature_name: str = 'auto'
    ) -> Dict[str, lgb.Booster]:

        configs = self._generate_diversity_configs()
        base_weights = self._compute_sample_weights(len(y), self.decay_rate)

        self.models = {}

        for i, cfg in enumerate(configs):
            name = cfg["name"]
            seed = self.random_state + i

            # Prepare Target
            y_trans = self._apply_target_transform(y, cfg["transform"])

            # Prepare Params
            params = copy.deepcopy(self.base_params)
            params.update(cfg.get("params", {}))
            params['random_state'] = seed
            params['seed'] = seed

            # Prepare Dataset
            # Note: Group C disables bagging in params, so we pass full data.
            # Group A/B rely on LGBM internal bagging (fraction < 1.0).
            # We do NOT manually subsample rows here anymore, relying on LGBM's bagging
            # to handle the diversity via seeds, as per standard LGBM usage.

            dtrain = lgb.Dataset(X, label=y_trans, weight=base_weights)

            valid_sets = [dtrain]
            if eval_set:
                X_val, y_val = eval_set
                y_val_trans = self._apply_target_transform(y_val, cfg["transform"])
                dval = lgb.Dataset(X_val, label=y_val_trans, reference=dtrain)
                valid_sets.append(dval)

            # Determine Objective
            fobj = cfg.get("objective_factory", None)
            if fobj:
                fobj = fobj() # Instantiate

            # If config has "objective" string (e.g. quantile), use it.
            # If fobj is provided, it overrides string objective (LGBM convention: specify None in params?)
            # Actually, if we pass fobj to train(), params['objective'] is ignored or should be 'custom'?
            # We keep params['objective'] if it's 'quantile'. If custom, we rely on fobj.

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                fobj=fobj,
                valid_sets=valid_sets,
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)] if eval_set else None
            )

            self.models[name] = model

        return self.models

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate predictions for all models.
        Returns Dictionary {model_name: raw_scores}.
        """
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X)
        return preds
