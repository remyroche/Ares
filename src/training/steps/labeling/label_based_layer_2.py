"""
Label-Based Layer 2: Regime-Conditional Geometry Optimization & Meta-Labeling
-----------------------------------------------------------------------------
This module implements the Layer 2 architecture for meta-labeling, focusing on
regime-conditional barrier optimization and ML learnability probes.

It performs:
1. Event generation based on volatility-scaled returns.
2. Regime-conditional barrier family assignment.
3. Independent optimization of barrier geometries (TP/SL/Horizon) per family using Optuna.
4. Learnability assessment using cheap ML probes (Shallow LGBM, Linear) with multiple metrics.
5. Selection of diverse, high-quality geometries with stability constraints.
6. Bagged output generation with family-level cap checks.
"""

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
import copy

# Import compute_realized_returns from the existing module
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    compute_realized_returns
)

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class GeometryTrial:
    family: str
    params: Dict[str, Any]  # TP, SL, Horizon
    final_score: float
    learnability: float
    robust_magnitude: float
    stability: float
    balance: float
    raw_metrics: Dict[str, float]
    uuid: str

class LabelBasedLayer2:
    """
    Layer 2: Regime-Conditional Geometry Optimization & Meta-Labeling.
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,  # 10bps default (0.1%)
        n_trials: int = 50,
        n_splits: int = 3,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize Layer 2.

        Args:
            transaction_cost: Trading cost (slippage + fees) per side.
            n_trials: Number of Optuna trials per barrier family.
            n_splits: Number of TimeSeriesSplit folds for ML probes.
            random_state: Seed for reproducibility.
            verbose: Logging verbosity.
        """
        self.transaction_cost = transaction_cost
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose

        # Internal state
        self.selected_geometries: List[GeometryTrial] = []
        self.family_weights: Dict[str, float] = {}

        # Suppress Optuna logging if not verbose
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the Layer 2 pipeline with OOF generation.

        This method performs:
        1. Full Optimization (to get production geometries).
        2. K-Fold OOF Optimization (to get unbiased analytics/artifacts).

        Args:
            df: Input DataFrame containing 'close', 'vwap', 'volatility_1d',
                'trend_regime', 'vol_regime', etc.

        Returns:
            Dict containing:
            - 'oof_labels': OOF Weighted Consensus Labels (Series)
            - 'oof_returns': OOF Weighted Consensus Returns (Series)
            - 'weights': OOF Weights (Series)
            - 'individual_geometries': OOF predictions per geometry channel (Dict[str, Series])
            - 'events_df': Events DataFrame
            - 'selected_trials': List[Dict] (Production geometries from full fit)
        """
        logger.info("Starting Layer 2 Pipeline...")

        # Step 0: Preparation
        self._validate_inputs(df)
        events_df = self._generate_events(df)

        if events_df.empty:
            logger.warning("No events generated in Layer 2. Skipping.")
            return {}

        # ---------------------------------------------------------------------
        # Part A: Full Optimization (Production Artifacts)
        # ---------------------------------------------------------------------
        logger.info(">>> Layer 2: Running Full Optimization (Production)...")
        full_results = self._optimize_families(df, events_df)
        production_geometries = self._select_best_geometries(df, events_df, full_results)

        # Store for reference
        self.selected_geometries = production_geometries

        # ---------------------------------------------------------------------
        # Part B: OOF Optimization (Analytics Artifacts)
        # ---------------------------------------------------------------------
        logger.info(">>> Layer 2: Running OOF Optimization (Analytics)...")

        # Initialize storage for OOF results
        indices = df.index
        oof_labels = pd.Series(np.nan, index=indices)
        oof_returns = pd.Series(np.nan, index=indices)
        oof_weights = pd.Series(np.nan, index=indices)

        # Derive families dynamically to avoid hardcoding
        # We need standardize keys for consistent channels.
        # We'll use the unique families found in the full events generation.
        # However, events_df generation maps regimes to families.
        # Let's dry run the mapping once to get all potential families.
        # Or better, just use the set of families that *could* be generated.
        # But _get_barrier_family logic is static.
        families = ['Trend Continuation', 'Momentum', 'Mean Reversion']
        max_rank = 4
        oof_geo_preds = {}
        for fam in families:
            for r in range(max_rank):
                key = f"{fam}_Rank{r}"
                oof_geo_preds[key] = pd.Series(np.nan, index=indices)

        # K-Fold Split
        kf = KFold(n_splits=5, shuffle=False)

        # Iterate folds
        fold_idx = 0
        for train_idx, test_idx in kf.split(df):
            fold_idx += 1
            logger.info(f"   > Processing Fold {fold_idx}/5...")

            # Create Train Slice
            df_train = df.iloc[train_idx]

            # Subset events
            events_train = events_df.loc[events_df.index.intersection(df_train.index)]
            events_test = events_df.loc[events_df.index.intersection(df.index[test_idx])]

            if events_train.empty:
                logger.warning(f"Fold {fold_idx}: No training events. Skipping.")
                continue

            # Optimize on Train
            fold_results = self._optimize_families(df_train, events_train)
            if not fold_results:
                continue

            fold_geometries = self._select_best_geometries(df_train, events_train, fold_results)
            if not fold_geometries:
                continue

            # Rename/Standardize Geometries for consistent channels
            geo_by_fam = {}
            for g in fold_geometries:
                geo_by_fam.setdefault(g.family, []).append(g)

            standardized_geos = []
            for fam, geos in geo_by_fam.items():
                # Sort by final_score descending
                geos_sorted = sorted(geos, key=lambda x: x.final_score, reverse=True)
                for rank, g in enumerate(geos_sorted):
                    # Assign standardized UUID
                    g_copy = copy.deepcopy(g)
                    g_copy.uuid = f"{fam}_Rank{rank}"
                    standardized_geos.append(g_copy)

            # Predict on Test (Bagged Labeling)
            if not events_test.empty:
                # CRITICAL FIX: Pass FULL `df` to _bagged_labeling, but only process `events_test`.
                # This ensures compute_realized_returns can access price data beyond the fold boundary
                # for proper barrier hits (lookahead).
                fold_output = self._bagged_labeling(df, events_test, standardized_geos)

                # Assign to OOF arrays
                target_idx = events_test.index

                oof_labels.loc[target_idx] = fold_output['oof_labels']
                oof_returns.loc[target_idx] = fold_output['oof_returns']
                oof_weights.loc[target_idx] = fold_output['weights']

                # Assign individual geometry preds
                for uuid, series in fold_output['individual_geometries'].items():
                    if uuid in oof_geo_preds:
                        oof_geo_preds[uuid].loc[target_idx] = series

        # ---------------------------------------------------------------------
        # Final Packaging
        # ---------------------------------------------------------------------
        final_geo_preds = {k: v for k, v in oof_geo_preds.items() if v.notna().any()}

        logger.info("Layer 2 Pipeline Complete.")

        return {
            "oof_labels": oof_labels,
            "oof_returns": oof_returns,
            "weights": oof_weights,
            "individual_geometries": final_geo_preds,
            "events_df": events_df,
            "selected_trials": [asdict(t) for t in production_geometries]
        }

    def _validate_inputs(self, df: pd.DataFrame):
        """Ensure required columns exist."""
        required = ['close', 'volatility_1d']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in df: {missing}")

        # Check for regime columns, if missing create dummies
        if 'trend_regime' not in df.columns:
            logger.warning("'trend_regime' missing. Creating dummy 'Low' regime.")
            df['trend_regime'] = 'Low'
        if 'vol_regime' not in df.columns:
            logger.warning("'vol_regime' missing. Creating dummy 'Low' regime.")
            df['vol_regime'] = 'Low'

    def _generate_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 0: Generate events where |r_t| / sigma_t > 0.5.
        Returns a DataFrame of event timestamps.
        """
        returns = df['close'].pct_change()
        # Avoid division by zero
        vol = df['volatility_1d'].replace(0, np.nan)

        # Signal to Noise Ratio
        snr = returns.abs() / vol

        # Event Trigger
        trigger_mask = snr > 0.5

        events = df.index[trigger_mask]
        logger.info(f"Generated {len(events)} events from {len(df)} bars.")

        # Create events dataframe (index=timestamp)
        # We can store regime info here for easy lookup
        events_df = df.loc[events, ['trend_regime', 'vol_regime', 'volatility_1d']].copy()

        return events_df

    def _get_barrier_family(self, trend_regime: str, vol_regime: str) -> str:
        """
        Map regimes to barrier families.

        High Trend -> Trend Continuation
        Low Trend / High Vol -> Momentum
        Low Trend / Low Vol -> Mean Reversion
        """
        # Normalize inputs (handle int/float/string)
        t_reg = str(trend_regime).lower()
        v_reg = str(vol_regime).lower()

        is_high_trend = 'high' in t_reg or t_reg == '1' or t_reg == '1.0'
        is_high_vol = 'high' in v_reg or v_reg == '1' or v_reg == '1.0'

        if is_high_trend:
            return 'Trend Continuation'
        elif is_high_vol:
            # Low Trend + High Vol
            return 'Momentum'
        else:
            # Low Trend + Low Vol
            return 'Mean Reversion'

    def _compute_labels(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        tp_mult: float,
        sl_mult: float,
        horizon: int,
        family: str
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute triple barrier labels for the given parameters.

        Returns:
            binary_labels: Series (1 for profit, 0 for loss/timeout)
            realized_returns: Series of returns
        """
        # Recalculate returns to determine direction
        returns = df['close'].pct_change()
        directions = np.sign(returns.loc[events_df.index])
        # If 0 (rare), assume 1
        directions[directions == 0] = 1

        # Adjust direction based on family strategy
        if family == 'Mean Reversion':
            # In mean reversion, we bet AGAINST the move
            directions = -directions

        signals = pd.DataFrame(index=df.index)
        signals['consensus'] = 0
        signals.loc[events_df.index, 'consensus'] = directions

        # Adaptive thresholds: k * volatility
        vol_aligned = df['volatility_1d'].fillna(0)
        tp_series = vol_aligned * tp_mult
        sl_series = vol_aligned * sl_mult

        (
            realized_returns,
            binary_labels,
            _, event_durations, _, _, _, _
        ) = compute_realized_returns(
            df=df,
            signals=signals,
            profit_threshold=tp_series,
            stop_threshold=sl_series,
            horizon=horizon,
            transaction_cost=self.transaction_cost,
            min_event_spacing=0,
            volatility_series=vol_aligned
        )

        # Filter to our specific events
        # Note: compute_realized_returns might return labels for all signals
        subset_labels = binary_labels.reindex(events_df.index)
        subset_returns = realized_returns.reindex(events_df.index)
        subset_durations = event_durations.reindex(events_df.index)

        # Filter events shorter than 1 bar (noise reduction) - requested in review
        # Note: durations < 1 are theoretically impossible if horizon >= 1, but check anyway
        valid_duration = subset_durations >= 1.0
        subset_labels.loc[~valid_duration] = np.nan
        subset_returns.loc[~valid_duration] = np.nan

        return subset_labels, subset_returns

    def _train_probes(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Step 4: Cheap ML learnability probes.
        Train Shallow LGBM and Linear Model.
        """
        valid = y.notna()
        X_clean = X.loc[valid]
        y_clean = y.loc[valid]

        if len(y_clean) < 50 or y_clean.nunique() < 2:
            return {'auc': 0.5, 'log_loss': 1.0, 'ic': 0.0, 'passed': False}

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        # Models
        lgbm = lgb.LGBMClassifier(
            max_depth=3,
            n_estimators=50,
            num_leaves=8,
            learning_rate=0.1,
            verbose=-1,
            random_state=self.random_state,
            n_jobs=1
        )

        linear = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=200,
            n_jobs=1
        )

        scaler = StandardScaler()

        metrics = {
            'lgbm_auc': [], 'lgbm_ic': [], 'lgbm_ll': [],
            'lin_auc': [], 'lin_ic': [], 'lin_ll': []
        }

        try:
            for train_index, test_index in tscv.split(X_clean):
                X_train, X_test = X_clean.iloc[train_index], X_clean.iloc[test_index]
                y_train, y_test = y_clean.iloc[train_index], y_clean.iloc[test_index]

                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    continue

                # LGBM
                lgbm.fit(X_train, y_train)
                p_lgbm = lgbm.predict_proba(X_test)[:, 1]

                try:
                    metrics['lgbm_auc'].append(roc_auc_score(y_test, p_lgbm))
                    metrics['lgbm_ll'].append(log_loss(y_test, p_lgbm))
                    ic, _ = spearmanr(y_test, p_lgbm)
                    metrics['lgbm_ic'].append(ic if not np.isnan(ic) else 0.0)
                except:
                    pass

                # Linear
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                linear.fit(X_train_scaled, y_train)
                p_linear = linear.predict_proba(X_test_scaled)[:, 1]

                try:
                    metrics['lin_auc'].append(roc_auc_score(y_test, p_linear))
                    metrics['lin_ll'].append(log_loss(y_test, p_linear))
                    ic, _ = spearmanr(y_test, p_linear)
                    metrics['lin_ic'].append(ic if not np.isnan(ic) else 0.0)
                except:
                    pass

        except Exception as e:
            logger.warning(f"Probe failure: {e}")
            return {'auc': 0.5, 'log_loss': 1.0, 'ic': 0.0, 'passed': False}

        if not metrics['lgbm_auc']:
            return {'auc': 0.5, 'log_loss': 1.0, 'ic': 0.0, 'passed': False}

        avg_auc_lgbm = np.mean(metrics['lgbm_auc'])
        avg_auc_linear = np.mean(metrics['lin_auc'])

        avg_ic_lgbm = np.mean(metrics['lgbm_ic'])
        avg_ic_linear = np.mean(metrics['lin_ic'])

        avg_ll_lgbm = np.mean(metrics['lgbm_ll'])
        avg_ll_linear = np.mean(metrics['lin_ll'])

        final_auc = np.median([avg_auc_lgbm, avg_auc_linear])

        return {
            'auc': final_auc,
            'ic': np.mean([avg_ic_lgbm, avg_ic_linear]),
            'log_loss': np.mean([avg_ll_lgbm, avg_ll_linear]),
            'auc_lgbm': avg_auc_lgbm,
            'auc_linear': avg_auc_linear,
            'passed': (avg_auc_lgbm >= 0.52) and (avg_auc_linear >= 0.52)
        }

    def _check_stability(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        trial_params: Dict[str, Any],
        base_score: float,
        family: str
    ) -> bool:
        """
        A. Stability under perturbation
        Perturb TP/SL/H by ±10-20% and require graceful degradation.
        Checks:
        1. Returns don't flip sign
        2. Mean return doesn't drop by > 30%
        3. Variance doesn't explode (optional, checked via Sharpe proxy)
        """
        perturbations = [0.9, 1.1]

        base_labels, base_returns = self._compute_labels(df, events_df, family=family, **trial_params)
        base_mean_ret = base_returns.mean()

        if np.isnan(base_mean_ret): return False

        # If base return is negative, it's already bad, but maybe stable.
        # But we only select good geometries.
        if base_mean_ret < 0: return False

        for p in perturbations:
            perturbed_params = copy.deepcopy(trial_params)
            perturbed_params['tp_mult'] *= p
            perturbed_params['sl_mult'] *= p
            # Horizon is int
            perturbed_params['horizon'] = int(max(1, perturbed_params['horizon'] * p))

            labels, returns = self._compute_labels(df, events_df, family=family, **perturbed_params)

            mean_ret = returns.mean()
            if np.isnan(mean_ret): return False

            # Check 1: Sign flip
            if mean_ret < 0:
                logger.debug(f"Stability failed: Returns flipped sign (base={base_mean_ret:.5f}, pert={mean_ret:.5f})")
                return False

            # Check 2: Relative drop threshold (max 30% reduction)
            # If mean_ret is significantly lower than base_mean_ret
            # Allow some noise, but drop > 30% is suspicious
            if mean_ret < base_mean_ret * 0.7:
                logger.debug(f"Stability failed: Returns dropped > 30% (base={base_mean_ret:.5f}, pert={mean_ret:.5f})")
                return False

        return True

    def _optimize_families(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame
    ) -> Dict[str, List[GeometryTrial]]:
        """
        Run Optuna optimization for each barrier family.
        """
        results: Dict[str, List[GeometryTrial]] = {}

        events_df['family'] = events_df.apply(
            lambda x: self._get_barrier_family(x['trend_regime'], x['vol_regime']),
            axis=1
        )

        unique_families = events_df['family'].unique()

        # Enhanced Probe Features
        probe_features = pd.DataFrame(index=df.index)
        probe_features['vol_1d'] = df['volatility_1d']
        probe_features['ret_1'] = df['close'].pct_change().fillna(0)
        # Add lagged returns and momentum as requested
        probe_features['ret_5'] = df['close'].pct_change(5).fillna(0)
        probe_features['ret_20'] = df['close'].pct_change(20).fillna(0)
        # Rolling volatility
        probe_features['vol_5'] = probe_features['ret_1'].rolling(5).std().fillna(0)

        for family in unique_families:
            logger.info(f"Optimizing family: {family}")

            family_mask = events_df['family'] == family
            family_events = events_df[family_mask]

            if len(family_events) < 50:
                logger.warning(f"Not enough events for family {family} ({len(family_events)}). Skipping.")
                continue

            study = optuna.create_study(direction="maximize")
            trial_results = []

            def objective(trial):
                # Suggest parameters based on family
                if family == 'Trend Continuation':
                    tp_mult = trial.suggest_float('tp_mult', 1.5, 4.0)
                    sl_mult = trial.suggest_float('sl_mult', 0.5, 1.5)
                    horizon = trial.suggest_int('horizon', 12, 48)
                elif family == 'Momentum':
                    tp_mult = trial.suggest_float('tp_mult', 1.0, 2.5)
                    sl_mult = trial.suggest_float('sl_mult', 0.5, 1.2)
                    horizon = trial.suggest_int('horizon', 6, 18)
                else: # Mean Reversion
                    tp_mult = trial.suggest_float('tp_mult', 0.5, 1.5)
                    sl_mult = trial.suggest_float('sl_mult', 0.3, 1.0)
                    horizon = trial.suggest_int('horizon', 4, 12)

                # Compute labels
                labels, returns = self._compute_labels(df, family_events, tp_mult, sl_mult, horizon, family)

                # Metrics
                mean_ret = returns.mean()
                if np.isnan(mean_ret): mean_ret = -1.0

                count = labels.notna().sum()
                if count < 20: return -1.0

                # Align features to events
                X_probe = probe_features.loc[labels.index]
                probe_res = self._train_probes(X_probe, labels)

                if not probe_res['passed']:
                    learnability = 0.0
                else:
                    learnability = probe_res['auc']

                pos_ratio = (labels == 1).mean()
                balance = 1.0 - abs(pos_ratio - 0.5) * 2

                robust_magnitude = mean_ret * 1000

                final_score = robust_magnitude * np.log1p(count) * balance * learnability

                t_obj = GeometryTrial(
                    family=family,
                    params={'tp_mult': tp_mult, 'sl_mult': sl_mult, 'horizon': horizon},
                    final_score=final_score,
                    learnability=learnability,
                    robust_magnitude=robust_magnitude,
                    stability=np.log1p(count),
                    balance=balance,
                    raw_metrics=probe_res,
                    uuid=f"{family}_{trial.number}"
                )
                trial_results.append(t_obj)

                return final_score

            study.optimize(objective, n_trials=self.n_trials)
            results[family] = trial_results

        return results

    def _select_best_geometries(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        family_results: Dict[str, List[GeometryTrial]]
    ) -> List[GeometryTrial]:
        """Step 3.2 & 3.3: Prune families and select diverse geometries."""
        # 3.2 Discard poorer barrier families
        family_medians = {}
        for fam, trials in family_results.items():
            trials_sorted = sorted(trials, key=lambda x: x.final_score, reverse=True)
            top_k = trials_sorted[:10]
            if not top_k:
                continue
            median_score = np.median([t.final_score for t in top_k])
            family_medians[fam] = median_score

        sorted_families = sorted(family_medians.items(), key=lambda x: x[1], reverse=True)
        keep_families = [f[0] for f in sorted_families[:3]]

        selected = []

        # 3.3 Keep diverse geometries per family
        for fam in keep_families:
            trials = family_results[fam]
            trials.sort(key=lambda x: x.final_score, reverse=True)
            n_top = max(2, int(len(trials) * 0.2))
            top_tier = trials[:n_top]

            fam_selected = []

            # Helper to normalize params for distance calculation
            tp_vals = [t.params['tp_mult'] for t in top_tier]
            sl_vals = [t.params['sl_mult'] for t in top_tier]
            h_vals = [t.params['horizon'] for t in top_tier]

            tp_range = max(tp_vals) - min(tp_vals) + 1e-6
            sl_range = max(sl_vals) - min(sl_vals) + 1e-6
            h_range = max(h_vals) - min(h_vals) + 1e-6

            def get_norm_vec(t):
                return np.array([
                    (t.params['tp_mult'] - min(tp_vals)) / tp_range,
                    (t.params['sl_mult'] - min(sl_vals)) / sl_range,
                    (t.params['horizon'] - min(h_vals)) / h_range
                ])

            # Pick best first (if stable)
            for cand in top_tier:
                fam_events = events_df[events_df['family'] == fam]
                if self._check_stability(df, fam_events, cand.params, cand.final_score, fam):
                    fam_selected.append(cand)
                    break

            if not fam_selected: continue

            # Pick others maximizing normalized distance
            # Try to get at least 2 geometries per family if possible (Requirement 3.3)
            # Loop until we have 4 or run out of candidates

            candidate_pool = [t for t in top_tier if t not in fam_selected]

            while len(fam_selected) < 4 and candidate_pool:
                best_cand = None
                max_dist = -1.0

                for cand in candidate_pool:
                    dists = [np.linalg.norm(get_norm_vec(cand) - get_norm_vec(s)) for s in fam_selected]
                    min_d = min(dists)

                    if min_d > max_dist:
                        max_dist = min_d
                        best_cand = cand

                if best_cand:
                    # Stability check
                    fam_events = events_df[events_df['family'] == fam]
                    if self._check_stability(df, fam_events, best_cand.params, best_cand.final_score, fam):
                        fam_selected.append(best_cand)

                    candidate_pool.remove(best_cand)
                else:
                    break

            # Require at least 2 geometries? Or just prefer?
            # "Select 2-4 complementary geometries per family"
            # If we only found 1 stable one, we keep it.
            selected.extend(fam_selected)

        return selected

    def _bagged_labeling(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        geometries: List[GeometryTrial]
    ) -> Dict[str, Any]:
        """
        Step 3.4: Generate final bagged outputs with advanced weighting checks.

        Outputs:
        - Weighted Consensus Labels
        - Weighted Consensus Returns
        - Event Weights (capped and normalized)
        """

        # Ensure family assignment is up to date
        events_df['family'] = events_df.apply(
            lambda x: self._get_barrier_family(x['trend_regime'], x['vol_regime']),
            axis=1
        )

        # Organize geometries by family
        geo_by_fam = {}
        for g in geometries:
            geo_by_fam.setdefault(g.family, []).append(g)

        # Storage for aggregation
        composite_labels = pd.Series(index=events_df.index, dtype=float)
        composite_returns = pd.Series(index=events_df.index, dtype=float)
        composite_weights = pd.Series(index=events_df.index, dtype=float)
        oof_preds = {} # Store individual geometry predictions

        # Iterate by family (since events are disjoint by family)
        for family, fam_geos in geo_by_fam.items():
            fam_mask = events_df['family'] == family
            fam_events = events_df[fam_mask]

            if fam_events.empty: continue

            # Temporary storage for this family's calculations
            # Dimensions: (n_events, n_geometries)
            n_events = len(fam_events)
            n_geos = len(fam_geos)

            geo_labels_mat = np.zeros((n_events, n_geos))
            geo_returns_mat = np.zeros((n_events, n_geos))
            geo_scores_mat = np.zeros((n_events, n_geos))
            valid_mask_mat = np.zeros((n_events, n_geos), dtype=bool)

            for i, g in enumerate(fam_geos):
                # Compute labels/returns for this geometry
                lbls, rets = self._compute_labels(df, fam_events, family=family, **g.params)

                # Store individual geometry output
                oof_preds[g.uuid] = lbls

                # Align to fam_events index
                lbls_aligned = lbls.reindex(fam_events.index)
                rets_aligned = rets.reindex(fam_events.index)

                # Identify valid labels (not NaN)
                not_na = lbls_aligned.notna()

                # Fill matrices
                geo_labels_mat[not_na, i] = lbls_aligned[not_na]
                geo_returns_mat[not_na, i] = rets_aligned[not_na]
                geo_scores_mat[not_na, i] = g.final_score
                valid_mask_mat[not_na, i] = True

            # --- Per-Geometry Capping Logic ---
            # Raw total score per event
            # Sum of scores of VALID geometries for each event
            event_total_score = np.sum(geo_scores_mat * valid_mask_mat, axis=1)

            # Max contribution per geometry: 30% of event total
            max_contrib = 0.3 * event_total_score

            # Broadcast max_contrib to match geometry dimension
            max_contrib_mat = max_contrib[:, np.newaxis]

            # Cap the weights: min(score, max_contrib)
            # Only apply to valid geometries (invalid have score 0 anyway in calculation,
            # but let's be explicit: capped_weight should be 0 if invalid)
            capped_weights_mat = np.minimum(geo_scores_mat, max_contrib_mat)
            capped_weights_mat[~valid_mask_mat] = 0.0

            # Final Event Weight (sum of capped weights)
            final_event_weights = np.sum(capped_weights_mat, axis=1)

            # Avoid division by zero
            safe_weights = final_event_weights.copy()
            safe_weights[safe_weights == 0] = 1.0 # arbitrary, will be 0 in result anyway

            # Weighted Consensus Calculation
            # Weighted Average Label
            w_labels_sum = np.sum(geo_labels_mat * capped_weights_mat, axis=1)
            consensus_labels = w_labels_sum / safe_weights

            # Weighted Average Return
            w_returns_sum = np.sum(geo_returns_mat * capped_weights_mat, axis=1)
            consensus_returns = w_returns_sum / safe_weights

            # Handle events with no valid geometries
            no_valid_geo = final_event_weights == 0
            consensus_labels[no_valid_geo] = np.nan
            consensus_returns[no_valid_geo] = np.nan

            # Assign to main storage
            composite_labels.loc[fam_events.index] = consensus_labels
            composite_returns.loc[fam_events.index] = consensus_returns
            composite_weights.loc[fam_events.index] = final_event_weights

        # --- Global Family Normalization (Max 60% of total mass) ---
        # "weights[event.family == fam] = np.minimum(weights[event.family == fam], family_cap)"

        # Fill NaNs in weights with 0
        composite_weights = composite_weights.fillna(0.0)

        total_weight_global = composite_weights.sum()

        if total_weight_global > 0:
            for family in geo_by_fam.keys():
                fam_mask = events_df['family'] == family
                fam_total_weight = composite_weights[fam_mask].sum()

                # Cap at 60% of GLOBAL total
                family_cap = 0.6 * total_weight_global

                if fam_total_weight > family_cap:
                    scale_factor = family_cap / fam_total_weight
                    logger.info(f"Scaling down family {family} by {scale_factor:.4f} (Total: {fam_total_weight:.2f} > Cap: {family_cap:.2f})")
                    composite_weights.loc[fam_mask] *= scale_factor

        # Normalize final weights to mean=1.0 for stability
        mean_weight = composite_weights.mean()
        if mean_weight > 0:
            composite_weights /= mean_weight

        return {
            "oof_labels": composite_labels,
            "oof_returns": composite_returns,
            "weights": composite_weights,
            "individual_geometries": oof_preds,
            "selected_trials": [asdict(t) for t in geometries]
        }
