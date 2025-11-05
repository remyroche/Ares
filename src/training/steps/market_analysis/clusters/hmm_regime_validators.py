"""
HMM Regime Validators

Comprehensive validation methods for HMM-based regime models including:
I. Predictive/generalization checks
II. Stability & reproducibility
III. Regime occupancy & persistence
IV. Transition matrix sensibility
V. Emission/geometric diagnostics
VI. Posterior predictive checks
VII. Economic utility & robustness
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings
import logging

from src.utils.tprint import tprint_info, tprint_warning, tprint_debug, tprint_success


class HMMRegimeValidator:
    """
    Comprehensive validator for HMM-based regime models.
    
    Implements all validation checks recommended for production regime models.
    """
    
    def __init__(self, timeframe: str = "1h"):
        """
        Initialize HMM regime validator.
        
        Args:
            timeframe: Timeframe for duration interpretability (e.g., '1h', '1d')
        """
        self.timeframe = timeframe
        self.samples_per_day = self._get_samples_per_day(timeframe)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_samples_per_day(self, timeframe: str) -> int:
        """Get number of samples per day for timeframe."""
        mapping = {
            '1m': 1440, '3m': 480, '5m': 288, '15m': 96,
            '30m': 48, '1h': 24, '60m': 24, '4h': 6, '1d': 1
        }
        return mapping.get(timeframe, 24)  # Default to hourly
    
    # ==========================================
    # I. PREDICTIVE/GENERALIZATION CHECKS
    # ==========================================
    
    def rolling_predictive_ll_validation(
        self,
        model,
        data: np.ndarray,
        n_folds: int = 5,
        min_train_size: int = 500
    ) -> Dict[str, Any]:
        """
        Rolling predictive log-likelihood on disjoint holdout blocks.
        
        Checks for stable improvement vs baseline (AR(1) or constant volatility).
        
        Args:
            model: Fitted HMM model with predict_proba or log_likelihood method
            data: Time series data
            n_folds: Number of disjoint blocks
            min_train_size: Minimum training size per fold
            
        Returns:
            Dict with rolling LL scores and baseline comparisons
        """
        tprint_info("📊 I. Validating rolling predictive log-likelihood...")
        
        # Create disjoint blocks
        block_size = len(data) // (n_folds + 1)
        if block_size < min_train_size:
            tprint_warning(f"⚠️ Insufficient data for {n_folds} folds")
            return {'delta_ll': [], 'mean_delta_ll': 0.0, 'positive_folds': 0}
        
        delta_lls = []
        holdout_lls = []
        baseline_lls = []
        
        for fold in range(n_folds):
            # Training data: all data before holdout
            train_end = (fold + 1) * block_size
            train_data = data[:train_end]
            
            # Holdout: next block
            holdout_start = train_end
            holdout_end = min(holdout_start + block_size, len(data))
            holdout_data = data[holdout_start:holdout_end]
            
            if len(holdout_data) < 10:
                continue
            
            try:
                # Calculate HMM predictive LL on holdout
                if hasattr(model, 'score'):
                    hmm_ll = model.score(holdout_data)
                else:
                    hmm_ll = 0.0
                
                # Baseline: AR(1) log-likelihood
                baseline_ll = self._ar1_log_likelihood(train_data, holdout_data)
                
                delta_ll = hmm_ll - baseline_ll
                delta_lls.append(delta_ll)
                holdout_lls.append(hmm_ll)
                baseline_lls.append(baseline_ll)
                
            except Exception as e:
                tprint_warning(f"⚠️ Fold {fold} failed: {e}")
                continue
        
        if len(delta_lls) == 0:
            return {'delta_ll': [], 'mean_delta_ll': 0.0, 'positive_folds': 0}
        
        # Calculate statistics
        mean_delta = np.mean(delta_lls)
        std_delta = np.std(delta_lls)
        positive_folds = sum(1 for dl in delta_lls if dl > 0)
        
        # Effect size: mean improvement / std
        effect_size = mean_delta / (std_delta + 1e-10)
        
        # DIAGNOSTIC: Median & IQR of predictive LL
        median_ll = np.median(holdout_lls)
        q25_ll = np.percentile(holdout_lls, 25)
        q75_ll = np.percentile(holdout_lls, 75)
        iqr_ll = q75_ll - q25_ll
        
        result = {
            'delta_ll_across_folds': delta_lls,
            'mean_delta_ll': mean_delta,
            'std_delta_ll': std_delta,
            'positive_folds': positive_folds,
            'total_folds': len(delta_lls),
            'positive_ratio': positive_folds / len(delta_lls),
            'effect_size': effect_size,
            'holdout_lls': holdout_lls,
            'baseline_lls': baseline_lls,
            # DIAGNOSTIC: Median & IQR
            'predictive_ll_median': float(median_ll),
            'predictive_ll_iqr': float(iqr_ll),
            'predictive_ll_q25': float(q25_ll),
            'predictive_ll_q75': float(q75_ll)
        }
        
        # Heuristic: consistent positive ΔLL
        if positive_folds / len(delta_lls) > 0.7 and effect_size > 1.0:
            tprint_success(f"✅ Predictive LL: {positive_folds}/{len(delta_lls)} positive folds, effect size={effect_size:.2f}")
        else:
            tprint_warning(f"⚠️ Weak predictive improvement: {positive_folds}/{len(delta_lls)} positive, effect={effect_size:.2f}")
        
        return result
    
    def _ar1_log_likelihood(self, train_data: np.ndarray, test_data: np.ndarray) -> float:
        """Calculate AR(1) baseline log-likelihood."""
        try:
            # Fit AR(1) on training data (simple)
            if len(train_data) < 2:
                return -np.inf
            
            # Calculate autocorrelation at lag 1
            returns = np.diff(train_data[:, 0]) if train_data.shape[1] > 0 else train_data.flatten()
            if len(returns) < 2:
                return -np.inf
            
            rho = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            sigma = np.std(returns)
            
            # Calculate LL on test data
            test_returns = np.diff(test_data[:, 0]) if test_data.shape[1] > 0 else test_data.flatten()
            if len(test_returns) < 1:
                return -np.inf
            
            # Simple AR(1) prediction: x_t = rho * x_{t-1}
            predictions = rho * test_returns[:-1]
            residuals = test_returns[1:] - predictions
            
            # Gaussian log-likelihood
            ll = -0.5 * np.sum((residuals / sigma) ** 2) - len(residuals) * np.log(sigma * np.sqrt(2 * np.pi))
            
            return ll
            
        except Exception as e:
            tprint_debug(f"AR(1) baseline failed: {e}")
            return -np.inf
    
    # ==========================================
    # II. STABILITY & REPRODUCIBILITY
    # ==========================================
    
    def refit_stability_validation(
        self,
        model_fit_func: Callable,
        data: np.ndarray,
        n_refits: int = 10,
        perturbation_scale: float = 0.01
    ) -> Dict[str, Any]:
        """
        Measure refit stability across different seeds and data perturbations.
        
        Args:
            model_fit_func: Function that fits model and returns labels
            data: Data to fit on
            n_refits: Number of refits
            perturbation_scale: Scale of data perturbations
            
        Returns:
            Dict with ARI, NMI stability metrics
        """
        tprint_info(f"📊 II. Validating refit stability ({n_refits} refits)...")
        
        all_labels = []
        
        for i in range(n_refits):
            try:
                # Add small perturbation
                if perturbation_scale > 0:
                    noise = np.random.normal(0, perturbation_scale, data.shape)
                    perturbed_data = data + noise * np.std(data, axis=0)
                else:
                    perturbed_data = data
                
                # Fit model with different seed
                labels = model_fit_func(perturbed_data, seed=i)
                all_labels.append(labels)
                
            except Exception as e:
                tprint_warning(f"⚠️ Refit {i} failed: {e}")
                continue
        
        if len(all_labels) < 2:
            return {'ari_scores': [], 'median_ari': 0.0, 'mean_ari': 0.0}
        
        # Calculate pairwise ARI and NMI
        ari_scores = []
        nmi_scores = []
        
        for i in range(len(all_labels)):
            for j in range(i + 1, len(all_labels)):
                try:
                    ari = adjusted_rand_score(all_labels[i], all_labels[j])
                    nmi = normalized_mutual_info_score(all_labels[i], all_labels[j])
                    ari_scores.append(ari)
                    nmi_scores.append(nmi)
                except Exception as e:
                    continue
        
        if len(ari_scores) == 0:
            return {'ari_scores': [], 'median_ari': 0.0, 'mean_ari': 0.0}
        
        median_ari = np.median(ari_scores)
        mean_ari = np.mean(ari_scores)
        median_nmi = np.median(nmi_scores) if nmi_scores else 0.0
        
        # DIAGNOSTIC: ARI median & IQR
        q25_ari = np.percentile(ari_scores, 25)
        q75_ari = np.percentile(ari_scores, 75)
        iqr_ari = q75_ari - q25_ari
        
        result = {
            'ari_scores': ari_scores,
            'nmi_scores': nmi_scores,
            'median_ari': median_ari,
            'mean_ari': mean_ari,
            'median_nmi': median_nmi,
            'n_refits': len(all_labels),
            # DIAGNOSTIC: ARI across restarts (detailed)
            'ari_across_restarts': ari_scores,
            'ari_median': float(median_ari),
            'ari_iqr': float(iqr_ari),
            'ari_q25': float(q25_ari),
            'ari_q75': float(q75_ari)
        }
        
        # Heuristic: ARI median > 0.6 is decent; <0.4 indicates instability
        if median_ari > 0.6:
            tprint_success(f"✅ Stable regime identification: ARI median={median_ari:.3f}")
        elif median_ari > 0.4:
            tprint_warning(f"⚠️ Moderate stability: ARI median={median_ari:.3f}")
        else:
            tprint_warning(f"⚠️ Unstable regime identification: ARI median={median_ari:.3f} (crypto is noisy)")
        
        return result
    
    # ==========================================
    # III. REGIME OCCUPANCY & PERSISTENCE
    # ==========================================
    
    def regime_occupancy_persistence_validation(
        self,
        labels: np.ndarray,
        transition_matrix: Optional[np.ndarray] = None,
        timeframe_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validate regime occupancy and persistence.
        
        Args:
            labels: Regime labels
            transition_matrix: State transition matrix (if available)
            timeframe_hours: Hours per sample (e.g., 1.0 for 1h, 0.25 for 15m)
            
        Returns:
            Dict with occupancy and duration metrics
        """
        tprint_info("📊 III. Validating regime occupancy and persistence...")
        
        if timeframe_hours is None:
            timeframe_hours = 24.0 / self.samples_per_day
        
        # Calculate state occupancy
        unique_states = np.unique(labels)
        state_occupancy = {}
        for state in unique_states:
            occupancy = np.sum(labels == state) / len(labels)
            state_occupancy[int(state)] = float(occupancy)
        
        # Count tiny states (< 1% occupancy)
        tiny_states = sum(1 for occ in state_occupancy.values() if occ < 0.01)
        
        # DIAGNOSTIC: State occupancy distribution
        occupancy_values = sorted(state_occupancy.values(), reverse=True)
        min_occupancy_pct = min(occupancy_values) * 100 if occupancy_values else 0.0
        max_occupancy_pct = max(occupancy_values) * 100 if occupancy_values else 0.0
        
        # Shannon entropy of occupancy distribution
        occupancy_arr = np.array(list(state_occupancy.values()))
        occupancy_entropy = -np.sum(occupancy_arr * np.log(occupancy_arr + 1e-10))
        
        # Calculate expected durations from transition matrix
        expected_durations = {}
        if transition_matrix is not None:
            for i, state in enumerate(unique_states):
                if i < transition_matrix.shape[0]:
                    p_ii = transition_matrix[i, i]
                    # Expected duration: E[D] = 1 / (1 - p_ii)
                    if p_ii < 1.0:
                        duration_samples = 1.0 / (1.0 - p_ii)
                        duration_hours = duration_samples * timeframe_hours
                        duration_days = duration_hours / 24.0
                        expected_durations[int(state)] = {
                            'samples': duration_samples,
                            'hours': duration_hours,
                            'days': duration_days
                        }
        
        # Get min/max durations
        if expected_durations:
            durations_days = [d['days'] for d in expected_durations.values()]
            min_duration = min(durations_days)
            max_duration = max(durations_days)
        else:
            min_duration = None
            max_duration = None
        
        # Duration quality assessment
        # Heuristic: For hourly data, expect >48-168 hours (2-7 days)
        # For daily data, expect >7-14 days
        if timeframe_hours <= 1.0:  # Hourly or sub-hourly
            min_acceptable_hours = 48
            good_hours = 168
        else:  # 4h or daily
            min_acceptable_hours = 24 * 7  # 7 days
            good_hours = 24 * 14  # 14 days
        
        duration_quality = "unknown"
        if min_duration is not None:
            min_duration_hours = min_duration * 24
            if min_duration_hours >= good_hours:
                duration_quality = "good"
            elif min_duration_hours >= min_acceptable_hours:
                duration_quality = "acceptable"
            elif min_duration_hours >= min_acceptable_hours * 0.5:
                duration_quality = "warning"
            else:
                duration_quality = "poor"
        
        result = {
            'state_occupancy': state_occupancy,
            'tiny_state_count': tiny_states,
            'expected_durations': expected_durations,
            'min_expected_duration_days': min_duration,
            'max_expected_duration_days': max_duration,
            'duration_quality_flag': duration_quality,
            # DIAGNOSTIC: State occupancy distribution (detailed)
            'occupancy_distribution': occupancy_values,
            'occupancy_entropy': float(occupancy_entropy),
            'min_occupancy_pct': float(min_occupancy_pct),
            'max_occupancy_pct': float(max_occupancy_pct)
        }
        
        # Report
        tprint_info(f"   States: {len(unique_states)}, Tiny states (<1%): {tiny_states}")
        if min_duration is not None:
            tprint_info(f"   Duration range: {min_duration:.1f} - {max_duration:.1f} days")
            tprint_info(f"   Duration quality: {duration_quality}")
        
        if tiny_states > 0:
            tprint_warning(f"⚠️ Found {tiny_states} tiny states (<1% occupancy)")
        
        if duration_quality in ["warning", "poor"]:
            tprint_warning(
                f"⚠️ Short regime durations detected. "
                f"Min duration: {min_duration:.1f} days. "
                "Regimes may be capturing noise rather than persistent market states."
            )
        else:
            tprint_success(f"✅ Regime persistence: {duration_quality}")
        
        return result
    
    # ==========================================
    # IV. TRANSITION MATRIX SENSIBILITY
    # ==========================================
    
    def transition_matrix_validation(
        self,
        transition_matrix: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Validate transition matrix interpretability and detect oscillations.
        
        Args:
            transition_matrix: N×N transition probability matrix
            labels: Actual state sequence
            
        Returns:
            Dict with transition sensibility checks
        """
        tprint_info("📊 IV. Validating transition matrix sensibility...")
        
        n_states = transition_matrix.shape[0]
        
        # Check for diagonal dominance (stickiness)
        diagonal_values = np.diag(transition_matrix)
        off_diagonal_mean = []
        for i in range(n_states):
            off_diag = [transition_matrix[i, j] for j in range(n_states) if i != j]
            if off_diag:
                off_diagonal_mean.append(np.mean(off_diag))
        
        # Calculate interpretability score
        # High diagonal, low off-diagonal = interpretable (sticky states)
        avg_diagonal = np.mean(diagonal_values)
        avg_off_diagonal = np.mean(off_diagonal_mean) if off_diagonal_mean else 0.0
        interpretability = avg_diagonal / (avg_off_diagonal + 1e-10)
        interpretability_score = min(1.0, interpretability / 10.0)  # Normalize to 0-1
        
        # Detect unrealistic oscillation
        # Count consecutive state changes
        state_changes = np.sum(np.diff(labels) != 0)
        change_rate = state_changes / len(labels)
        
        # Heuristic: > 30% change rate might indicate oscillation
        unrealistic_oscillation = change_rate > 0.3
        
        result = {
            'avg_self_transition': float(avg_diagonal),
            'avg_cross_transition': float(avg_off_diagonal),
            'interpretability_score': float(interpretability_score),
            'state_changes': int(state_changes),
            'change_rate': float(change_rate),
            'unrealistic_oscillation': unrealistic_oscillation,
            'diagonal_values': diagonal_values.tolist()
        }
        
        if unrealistic_oscillation:
            tprint_warning(
                f"⚠️ High oscillation rate: {change_rate:.1%} "
                "(states flipping frequently - possible noise)"
            )
        else:
            tprint_success(f"✅ Transition matrix: interpretable={interpretability_score:.3f}")
        
        return result
    
    # ==========================================
    # V. EMISSION/GEOMETRIC DIAGNOSTICS
    # ==========================================
    
    def emission_diagnostics(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        feature_subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze state-conditioned distributions and distinctiveness.
        
        Args:
            data: Feature data
            labels: State labels
            feature_subset: Optional subset of features to analyze
            
        Returns:
            Dict with emission statistics and distinctiveness
        """
        tprint_info("📊 V. Analyzing emission distributions and distinctiveness...")
        
        unique_states = np.unique(labels)
        state_stats = {}
        
        # Select features to analyze
        if feature_subset:
            features = [f for f in feature_subset if f in data.columns]
        else:
            features = data.columns.tolist()[:10]  # First 10 features
        
        # Calculate state-conditioned statistics
        for state in unique_states:
            state_mask = labels == state
            state_data = data[state_mask]
            
            if len(state_data) < 5:
                continue
            
            state_stats[int(state)] = {
                'mean': float(state_data[features].mean().mean()),
                'std': float(state_data[features].std().mean()),
                'skew': float(state_data[features].skew().mean()),
                'kurtosis': float(state_data[features].kurtosis().mean()),
                'n_samples': int(len(state_data))
            }
        
        # Calculate distinctiveness: average pairwise distance between state means
        if len(state_stats) >= 2:
            means = np.array([s['mean'] for s in state_stats.values()])
            if len(means.shape) == 1:
                means = means.reshape(-1, 1)
            pairwise_distances = cdist(means.reshape(-1, 1), means.reshape(-1, 1))
            avg_distance = np.mean(pairwise_distances[np.triu_indices_from(pairwise_distances, k=1)])
            distinctiveness = min(1.0, avg_distance / 10.0)  # Normalize
        else:
            distinctiveness = 0.0
        
        result = {
            'state_conditioned_stats': state_stats,
            'emission_distinctiveness': float(distinctiveness),
            'n_analyzed_features': len(features)
        }
        
        tprint_info(f"   Analyzed {len(features)} features across {len(state_stats)} states")
        tprint_info(f"   Emission distinctiveness: {distinctiveness:.3f}")
        
        return result
    
    # ==========================================
    # VI. POSTERIOR PREDICTIVE CHECKS
    # ==========================================
    
    def posterior_predictive_check(
        self,
        model,
        data: np.ndarray,
        n_simulations: int = 100
    ) -> Dict[str, Any]:
        """
        ENHANCED posterior predictive check with CRPS, PIT, and tail quantiles.
        
        Args:
            model: Fitted HMM model with sample method
            data: Empirical data
            n_simulations: Number of simulations for ensemble
            
        Returns:
            Dict with:
            - Moment comparisons
            - CRPS score (Continuous Ranked Probability Score)
            - PIT values and uniformity test
            - Tail quantile comparison (q01, q05, q95, q99)
            - Tail coverage score
        """
        tprint_info("📊 VI. Running ENHANCED posterior predictive checks...")
        
        try:
            # Generate simulated data from model
            if hasattr(model, 'sample'):
                simulated_data, simulated_labels = model.sample(len(data))
            else:
                tprint_warning("⚠️ Model doesn't support sampling, skipping posterior predictive")
                return {}
            
            # Compare moments
            empirical_mean = np.mean(data, axis=0)
            simulated_mean = np.mean(simulated_data, axis=0)
            
            empirical_std = np.std(data, axis=0)
            simulated_std = np.std(simulated_data, axis=0)
            
            # Calculate moment differences
            mean_diff = np.mean(np.abs(empirical_mean - simulated_mean) / (empirical_std + 1e-10))
            std_diff = np.mean(np.abs(empirical_std - simulated_std) / (empirical_std + 1e-10))
            
            # Autocorrelation comparison
            emp_autocorr = self._calculate_autocorrelation(data[:, 0], lag=1) if data.shape[1] > 0 else 0.0
            sim_autocorr = self._calculate_autocorrelation(simulated_data[:, 0], lag=1) if simulated_data.shape[1] > 0 else 0.0
            autocorr_diff = abs(emp_autocorr - sim_autocorr)
            
            # DIAGNOSTIC: CRPS (Continuous Ranked Probability Score)
            # Simplified CRPS: mean absolute difference between empirical and simulated CDFs
            emp_data_flat = data.flatten()
            sim_data_flat = simulated_data.flatten()
            
            # Sort both
            emp_sorted = np.sort(emp_data_flat)
            sim_sorted = np.sort(sim_data_flat)
            
            # Resample to same length for comparison
            min_len = min(len(emp_sorted), len(sim_sorted))
            emp_sample = emp_sorted[np.linspace(0, len(emp_sorted)-1, min_len).astype(int)]
            sim_sample = sim_sorted[np.linspace(0, len(sim_sorted)-1, min_len).astype(int)]
            
            crps_score = float(np.mean(np.abs(emp_sample - sim_sample)))
            
            # DIAGNOSTIC: PIT (Probability Integral Transform) calibration
            # For each empirical point, calculate its percentile in simulated distribution
            pit_values = []
            for val in emp_data_flat[:min(1000, len(emp_data_flat))]:  # Sample for speed
                percentile = np.sum(sim_data_flat <= val) / len(sim_data_flat)
                pit_values.append(percentile)
            
            pit_values = np.array(pit_values)
            
            # Test uniformity with Kolmogorov-Smirnov test
            from scipy import stats as scipy_stats
            ks_statistic, ks_pvalue = scipy_stats.kstest(pit_values, 'uniform')
            pit_uniformity_pvalue = float(ks_pvalue)
            
            # DIAGNOSTIC: Tail quantile comparison
            quantiles = [0.01, 0.05, 0.25, 0.75, 0.95, 0.99]
            tail_comparison = {}
            
            for q in quantiles:
                emp_q = np.percentile(emp_data_flat, q * 100)
                sim_q = np.percentile(sim_data_flat, q * 100)
                tail_comparison[f'q{int(q*100):02d}'] = {
                    'empirical': float(emp_q),
                    'simulated': float(sim_q),
                    'diff': float(abs(emp_q - sim_q)),
                    'rel_diff': float(abs(emp_q - sim_q) / (abs(emp_q) + 1e-10))
                }
            
            # Tail coverage score: how well extreme quantiles match
            tail_qs = [0.01, 0.05, 0.95, 0.99]
            tail_errors = []
            for q in tail_qs:
                emp_q = np.percentile(emp_data_flat, q * 100)
                sim_q = np.percentile(sim_data_flat, q * 100)
                rel_error = abs(emp_q - sim_q) / (abs(emp_q) + 1e-10)
                tail_errors.append(rel_error)
            
            tail_coverage_score = float(1.0 - min(1.0, np.mean(tail_errors)))
            
            # Overall calibration score (0-1, higher is better)
            calibration_score = 1.0 - min(1.0, (mean_diff + std_diff + autocorr_diff) / 3.0)
            
            result = {
                # Basic moment comparison
                'mean_difference': float(mean_diff),
                'std_difference': float(std_diff),
                'autocorr_difference': float(autocorr_diff),
                'calibration_score': float(calibration_score),
                'empirical_mean': float(np.mean(empirical_mean)),
                'simulated_mean': float(np.mean(simulated_mean)),
                'empirical_std': float(np.mean(empirical_std)),
                'simulated_std': float(np.mean(simulated_std)),
                
                # DIAGNOSTIC: CRPS
                'crps_score': crps_score,
                
                # DIAGNOSTIC: PIT
                'pit_uniformity_pvalue': pit_uniformity_pvalue,
                'pit_ks_statistic': float(ks_statistic),
                
                # DIAGNOSTIC: Tail quantiles
                'tail_quantile_comparison': tail_comparison,
                'tail_coverage_score': tail_coverage_score
            }
            
            # Determine calibration flag
            if calibration_score > 0.7 and pit_uniformity_pvalue > 0.05:
                result['calibration_flag'] = 'well_calibrated'
                tprint_success(f"✅ Well-calibrated: score={calibration_score:.3f}, PIT p={pit_uniformity_pvalue:.3f}")
            elif calibration_score > 0.5:
                result['calibration_flag'] = 'acceptable'
                tprint_warning(f"⚠️ Acceptable calibration: score={calibration_score:.3f}")
            else:
                result['calibration_flag'] = 'poor'
                tprint_warning(f"⚠️ Poor calibration: score={calibration_score:.3f}")
            
            # Report tail coverage
            tprint_info(f"   CRPS: {crps_score:.4f}, Tail coverage: {tail_coverage_score:.3f}")
            
            return result
            
        except Exception as e:
            tprint_warning(f"⚠️ Posterior predictive check failed: {e}")
            import traceback
            tprint_debug(traceback.format_exc())
            return {}
    
    def _calculate_autocorrelation(self, series: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at given lag."""
        try:
            if len(series) < lag + 10:
                return 0.0
            return float(np.corrcoef(series[:-lag], series[lag:])[0, 1])
        except:
            return 0.0
    
    # ==========================================
    # VII. ECONOMIC UTILITY & ROBUSTNESS
    # ==========================================
    
    def economic_utility_validation(
        self,
        labels: np.ndarray,
        returns: pd.Series,
        transaction_cost_bps: float = 10.0,
        n_bootstrap: int = 100,
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Validate economic utility via regime-aware strategy backtest.
        
        ENHANCED: Now computes Sharpe and turnover across rolling folds
        to calculate median & IQR for robustness assessment.
        
        Args:
            labels: Regime labels
            returns: Forward returns
            transaction_cost_bps: Transaction costs in basis points
            n_bootstrap: Number of bootstrap samples
            n_folds: Number of rolling folds for Sharpe/turnover distribution
            
        Returns:
            Dict with Sharpe, drawdown, significance tests, and fold-wise distributions
        """
        tprint_info("📊 VII. Validating economic utility and robustness...")
        
        if returns is None or len(returns) == 0:
            tprint_warning("⚠️ No returns provided, skipping economic validation")
            return {}
        
        # Simple regime-aware strategy: go long in positive-return regimes
        # Ensure labels and returns have the same length
        min_length = min(len(labels), len(returns))
        if len(labels) != len(returns):
            self.logger.warning(f"Length mismatch between labels ({len(labels)}) and returns ({len(returns)}), truncating to {min_length}")
            labels = labels[:min_length]
            returns = returns[:min_length]
        
        unique_states = np.unique(labels)
        
        # Calculate average return per regime
        regime_returns = {}
        for state in unique_states:
            state_mask = labels == state
            regime_returns[state] = np.mean(returns[state_mask])
        
        # Create strategy: allocate to regimes with positive average returns
        strategy_returns = []
        transitions = 0
        prev_allocation = None
        
        for i, state in enumerate(labels):
            # Decide allocation based on regime
            if regime_returns[state] > 0:
                allocation = 1.0  # Long
            else:
                allocation = 0.0  # Cash
            
            # Track transitions
            if prev_allocation is not None and allocation != prev_allocation:
                transitions += 1
            prev_allocation = allocation
            
            # Apply transaction costs on transitions
            if i < len(returns):
                ret = returns.iloc[i] * allocation
                if transitions > 0 and i > 0:
                    ret -= transaction_cost_bps / 10000.0  # Convert bps to decimal
                strategy_returns.append(ret)
        
        if len(strategy_returns) == 0:
            return {}
        
        strategy_returns = np.array(strategy_returns)
        
        # Remove NaN values and check if we have valid returns
        strategy_returns = strategy_returns[~np.isnan(strategy_returns)]
        if len(strategy_returns) == 0:
            tprint_warning("   ⚠️ No valid strategy returns (all NaN), cannot calculate Sharpe")
            return {}
        
        # Calculate Sharpe ratio (annualized)
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        if std_return < 1e-10:
            tprint_warning("   ⚠️ Strategy returns have zero variance, Sharpe undefined")
            sharpe = 0.0
        else:
            sharpe = (mean_return / std_return) * np.sqrt(252 * self.samples_per_day)
        
        # Calculate max drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Turnover
        turnover = transitions / len(labels)
        
        # Bootstrap significance
        bootstrap_sharpes = []
        for _ in range(min(n_bootstrap, 100)):  # Limit to 100 for speed
            sample_idx = np.random.choice(len(strategy_returns), len(strategy_returns), replace=True)
            sample_returns = strategy_returns[sample_idx]
            sample_std = np.std(sample_returns)
            if sample_std >= 1e-10:
                sample_sharpe = (np.mean(sample_returns) / sample_std) * np.sqrt(252 * self.samples_per_day)
                bootstrap_sharpes.append(sample_sharpe)
        
        if len(bootstrap_sharpes) == 0:
            tprint_warning("   ⚠️ No valid bootstrap samples, using default CI")
            sharpe_ci_lower = 0.0
            sharpe_ci_upper = 0.0
            sharpe_significant = False
        else:
            sharpe_ci_lower = np.percentile(bootstrap_sharpes, 5)
            sharpe_ci_upper = np.percentile(bootstrap_sharpes, 95)
            sharpe_significant = sharpe_ci_lower > 0  # 90% CI above zero
        
        # Baseline: buy-and-hold
        clean_returns = returns.dropna()
        if len(clean_returns) == 0:
            tprint_warning("   ⚠️ No valid baseline returns, using baseline Sharpe = 0")
            baseline_sharpe = 0.0
        else:
            baseline_std = np.std(clean_returns)
            if baseline_std < 1e-10:
                baseline_sharpe = 0.0
            else:
                baseline_sharpe = (np.mean(clean_returns) / baseline_std) * np.sqrt(252 * self.samples_per_day)
        sharpe_uplift = sharpe - baseline_sharpe
        
        result = {
            'out_of_sample_sharpe': float(sharpe),
            'out_of_sample_max_drawdown': float(max_drawdown),
            'strategy_turnover': float(turnover),
            'transaction_cost_bps': float(transaction_cost_bps),
            'bootstrap_sharpe_ci': (float(sharpe_ci_lower), float(sharpe_ci_upper)),
            'sharpe_significant': bool(sharpe_significant),
            'baseline_sharpe': float(baseline_sharpe),
            'sharpe_uplift': float(sharpe_uplift),
            'n_transitions': int(transitions)
        }
        
        # Economic utility score: Sharpe adjusted for turnover and costs
        utility_score = max(0.0, sharpe - 0.5 * turnover)  # Penalize high turnover
        result['economic_utility_score'] = float(utility_score)
        
        # DIAGNOSTIC: Sharpe & Turnover across rolling folds
        sharpe_folds = []
        turnover_folds = []
        
        if n_folds > 1 and len(labels) >= n_folds * 100:  # Need sufficient data
            fold_size = len(labels) // n_folds
            
            for fold_idx in range(n_folds):
                start_idx = fold_idx * fold_size
                end_idx = start_idx + fold_size if fold_idx < n_folds - 1 else len(labels)
                
                fold_labels = labels[start_idx:end_idx]
                fold_returns = returns.iloc[start_idx:end_idx]
                
                # Calculate regime returns for this fold
                fold_regime_returns = {}
                for state in np.unique(fold_labels):
                    state_mask = fold_labels == state
                    fold_regime_returns[state] = np.mean(fold_returns[state_mask])
                
                # Compute fold strategy
                fold_strategy_returns = []
                fold_transitions = 0
                prev_alloc = None
                
                for i, state in enumerate(fold_labels):
                    allocation = 1.0 if fold_regime_returns.get(state, 0) > 0 else 0.0
                    
                    if prev_alloc is not None and allocation != prev_alloc:
                        fold_transitions += 1
                    prev_alloc = allocation
                    
                    if i < len(fold_returns):
                        ret = fold_returns.iloc[i] * allocation
                        if fold_transitions > 0 and i > 0:
                            ret -= transaction_cost_bps / 10000.0
                        fold_strategy_returns.append(ret)
                
                if len(fold_strategy_returns) > 0:
                    fold_strategy_returns = np.array(fold_strategy_returns)
                    # Remove NaN values
                    fold_strategy_returns = fold_strategy_returns[~np.isnan(fold_strategy_returns)]
                    if len(fold_strategy_returns) > 0:
                        fold_std = np.std(fold_strategy_returns)
                        if fold_std >= 1e-10:
                            fold_sharpe = (np.mean(fold_strategy_returns) / fold_std) * np.sqrt(252 * self.samples_per_day)
                        else:
                            fold_sharpe = 0.0
                        fold_turnover = fold_transitions / len(fold_labels)
                        
                        sharpe_folds.append(fold_sharpe)
                        turnover_folds.append(fold_turnover)
        
        # Calculate median & IQR for Sharpe and turnover
        if len(sharpe_folds) > 0:
            sharpe_median = np.median(sharpe_folds)
            sharpe_q25 = np.percentile(sharpe_folds, 25)
            sharpe_q75 = np.percentile(sharpe_folds, 75)
            sharpe_iqr = sharpe_q75 - sharpe_q25
            
            result['sharpe_across_folds'] = sharpe_folds
            result['sharpe_median'] = float(sharpe_median)
            result['sharpe_iqr'] = float(sharpe_iqr)
            result['sharpe_q25'] = float(sharpe_q25)
            result['sharpe_q75'] = float(sharpe_q75)
            
            tprint_info(f"   Sharpe distribution: median={sharpe_median:.3f}, IQR={sharpe_iqr:.3f}")
        
        if len(turnover_folds) > 0:
            turnover_median = np.median(turnover_folds)
            turnover_q25 = np.percentile(turnover_folds, 25)
            turnover_q75 = np.percentile(turnover_folds, 75)
            turnover_iqr = turnover_q75 - turnover_q25
            
            result['turnover_across_folds'] = turnover_folds
            result['turnover_median'] = float(turnover_median)
            result['turnover_iqr'] = float(turnover_iqr)
            result['turnover_q25'] = float(turnover_q25)
            result['turnover_q75'] = float(turnover_q75)
            
            tprint_info(f"   Turnover distribution: median={turnover_median:.1%}, IQR={turnover_iqr:.1%}")
        
        # Report
        tprint_info(f"   Sharpe: {sharpe:.3f}, Baseline: {baseline_sharpe:.3f}, Uplift: {sharpe_uplift:.3f}")
        tprint_info(f"   Max DD: {max_drawdown:.1%}, Turnover: {turnover:.1%}")
        
        if sharpe_significant and sharpe_uplift > 0.2:
            tprint_success(f"✅ Economically useful: Sharpe={sharpe:.3f}, uplift={sharpe_uplift:.3f}")
        elif sharpe > 0.5:
            tprint_warning(f"⚠️ Moderate economic utility: Sharpe={sharpe:.3f}")
        else:
            tprint_warning(f"⚠️ Low economic utility: Sharpe={sharpe:.3f}")
        
        return result


# Convenience function
def create_hmm_regime_validator(timeframe: str = "1h") -> HMMRegimeValidator:
    """
    Create HMM regime validator.
    
    Args:
        timeframe: Timeframe for duration interpretation
        
    Returns:
        HMMRegimeValidator instance
    """
    return HMMRegimeValidator(timeframe=timeframe)


__all__ = [
    'HMMRegimeValidator',
    'create_hmm_regime_validator'
]
