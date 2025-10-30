#!/usr/bin/env python3
"""
MS-DR Auto-Tuner with Optuna

Automatically optimizes MS-DR clustering parameters to maximize regime separation.

Usage:
    python ms_dr_auto_tuner_script.py --n-trials 100 --timeout 60
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import json

sys.path.insert(0, 'src')

from src.training.steps.market_analysis.ms_dr_clustering.ms_dr_clusterer import MSDRClusterer, MSDRConfig
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import ClusterQualityAssessor
from improved_ms_dr_signal import create_improved_regime_signal

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    print("⚠️ Optuna not available. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False


class MSDRAutoTuner:
    """Auto-tuner for MS-DR clustering parameters."""
    
    def __init__(self, data: np.ndarray, verbose: bool = True):
        """
        Initialize auto-tuner.
        
        Args:
            data: Input data for clustering (n_samples, n_features)
            verbose: Print progress
        """
        self.data = data
        self.verbose = verbose
        self.best_score = float('-inf')
        self.best_params = None
        self.trial_history = []
        
        self.quality_assessor = ClusterQualityAssessor(
            enable_hardware_optimization=True,
            enable_vectorization=True
        )
    
    def objective(self, trial):
        """Optuna objective function."""
        
        # Sample hyperparameters
        params = {
            'n_regimes': trial.suggest_int('n_regimes', 2, 6),
            'order': trial.suggest_int('order', 1, 4),
            'method': trial.suggest_categorical('method', ['powell', 'bfgs', 'nm']),
            'max_iter': trial.suggest_int('max_iter', 1000, 5000, step=1000),
            'switching_variance': trial.suggest_categorical('switching_variance', [True, False]),
            'auto_select_regimes': False  # Fixed during tuning
        }
        
        if self.verbose:
            print(f"\n📊 Trial {trial.number}: Testing parameters...")
            print(f"   n_regimes={params['n_regimes']}, order={params['order']}, method={params['method']}")
        
        try:
            # Create config
            config = MSDRConfig(
                n_regimes=params['n_regimes'],
                order=params['order'],
                method=params['method'],
                max_iter=params['max_iter'],
                switching_variance=params['switching_variance'],
                auto_select_regimes=params['auto_select_regimes'],
                model_type='autoregression',
                enable_pca=False,
                random_state=42,
                use_memory_optimization=True,
                use_hardware_acceleration=True,
                show_progress=False  # Disable progress bar during tuning
            )
            
            # Run clustering
            clusterer = MSDRClusterer(config)
            result = clusterer.fit_predict(self.data)
            
            if not result.success:
                if self.verbose:
                    print(f"   ❌ Clustering failed: {result.error_message}")
                return float('-inf')
            
            # Check for degenerate clustering
            unique_labels = np.unique(result.cluster_labels)
            if len(unique_labels) == 1:
                if self.verbose:
                    print(f"   ❌ Degenerate clustering (all samples in one regime)")
                return float('-inf')
            
            # Assess quality
            feature_df = pd.DataFrame(self.data)
            quality_metrics = self.quality_assessor.assess_quality(
                regime_labels=result.cluster_labels,
                feature_data=feature_df,
                forward_returns=None,
                timestamps=None,
                min_regime_size=10
            )
            
            # Composite score: weighted combination of metrics
            composite_score = 0.0
            weights = {
                'silhouette': 0.3,
                'balance': 0.2,
                'temporal_smoothness': 0.2,
                'quality': 0.3
            }
            
            if quality_metrics.silhouette_score is not None:
                composite_score += weights['silhouette'] * quality_metrics.silhouette_score
            
            if quality_metrics.balance_score is not None:
                composite_score += weights['balance'] * quality_metrics.balance_score
            
            if quality_metrics.temporal_smoothness is not None:
                composite_score += weights['temporal_smoothness'] * quality_metrics.temporal_smoothness
            
            if quality_metrics.quality_score is not None:
                composite_score += weights['quality'] * quality_metrics.quality_score
            
            # Penalize if too many regimes with small clusters
            cluster_counts = np.bincount(result.cluster_labels)
            min_cluster_size = cluster_counts.min()
            if min_cluster_size < 10:
                composite_score *= 0.5  # Penalty for tiny clusters
            
            if self.verbose:
                print(f"   ✅ Score: {composite_score:.4f}")
                print(f"      Silhouette: {quality_metrics.silhouette_score:.4f}" if quality_metrics.silhouette_score else "      Silhouette: None")
                print(f"      Balance: {quality_metrics.balance_score:.4f}" if quality_metrics.balance_score else "      Balance: None")
                print(f"      Overall Quality: {quality_metrics.quality_score:.4f}" if quality_metrics.quality_score else "      Overall Quality: None")
            
            # Store trial
            self.trial_history.append({
                'trial': trial.number,
                'params': params,
                'composite_score': composite_score,
                'n_clusters': result.n_clusters,
                'silhouette': quality_metrics.silhouette_score,
                'balance': quality_metrics.balance_score,
                'quality': quality_metrics.quality_score
            })
            
            # Update best
            if composite_score > self.best_score:
                self.best_score = composite_score
                self.best_params = params
            
            return composite_score
        
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Error: {e}")
            return float('-inf')
    
    def optimize(self, n_trials: int = 50, timeout: Optional[float] = None) -> Dict:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of trials
            timeout: Timeout in seconds (optional)
            
        Returns:
            Dictionary with optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for auto-tuning. Install with: pip install optuna")
        
        print("=" * 80)
        print("🎯 MS-DR AUTO-TUNER")
        print("=" * 80)
        print(f"\n📊 Configuration:")
        print(f"   Data shape: {self.data.shape}")
        print(f"   N trials: {n_trials}")
        print(f"   Timeout: {timeout}s" if timeout else "   Timeout: None")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        print(f"\n🚀 Starting optimization...")
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        print(f"\n✅ Optimization complete!")
        print(f"   Best score: {study.best_value:.4f}")
        print(f"   Best parameters: {study.best_params}")
        
        # Prepare results
        results = {
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'trial_history': self.trial_history,
            'timestamp': datetime.now().isoformat()
        }
        
        return results


def run_ms_dr_auto_tuner(n_trials: int = 50, timeout: Optional[float] = None):
    """Run MS-DR auto-tuner with improved signal."""
    
    print("=" * 80)
    print("🎯 MS-DR AUTO-TUNER WITH IMPROVED SIGNAL")
    print("=" * 80)
    
    # === STEP 1: Create market data ===
    print("\n📊 STEP 1: Creating Market Data")
    
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
    base_price = 3000.0
    
    regime_lengths = [350, 300, 350]
    regime_params = [
        {'volatility': 0.02, 'trend': 0.001, 'volume': 1.5},
        {'volatility': 0.05, 'trend': -0.0005, 'volume': 0.8},
        {'volatility': 0.01, 'trend': 0.0, 'volume': 1.0}
    ]
    
    prices = [base_price]
    volumes = []
    regime_idx = 0
    regime_counter = 0
    
    for i in range(n_samples):
        if regime_counter >= regime_lengths[regime_idx]:
            regime_idx = (regime_idx + 1) % 3
            regime_counter = 0
        
        params = regime_params[regime_idx]
        price_change = np.random.normal(params['trend'], params['volatility'])
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        
        volume = np.random.uniform(500 * params['volume'], 2000 * params['volume'])
        volumes.append(volume)
        
        regime_counter += 1
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices[:-1],
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
        'close': prices[1:],
        'volume': volumes
    })
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ Created market data: {df.shape}")
    
    # === STEP 2: Create improved regime signal ===
    print("\n🔧 STEP 2: Creating Improved Regime Signal")
    
    regime_signal, signal_diagnostics = create_improved_regime_signal(
        df,
        use_nonlinear=True,
        use_multiscale=True,
        use_adaptive_weighting=True
    )
    
    data = regime_signal.values.reshape(-1, 1)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"✅ Prepared data for MS-DR: {data.shape}")
    
    # === STEP 3: Run auto-tuner ===
    print("\n🚀 STEP 3: Running Auto-Tuner")
    
    tuner = MSDRAutoTuner(data, verbose=True)
    results = tuner.optimize(n_trials=n_trials, timeout=timeout)
    
    # === STEP 4: Save results ===
    print("\n📝 STEP 4: Saving Results")
    
    outcomes_dir = Path("outcomes")
    outcomes_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = outcomes_dir / f"ms_dr_autotuner_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {results_file}")
    
    # === STEP 5: Test best parameters ===
    print("\n🎯 STEP 5: Testing Best Parameters")
    
    best_config = MSDRConfig(
        n_regimes=results['best_params']['n_regimes'],
        order=results['best_params']['order'],
        method=results['best_params']['method'],
        max_iter=results['best_params']['max_iter'],
        switching_variance=results['best_params']['switching_variance'],
        auto_select_regimes=False,
        model_type='autoregression',
        enable_pca=False,
        random_state=42,
        use_memory_optimization=True,
        use_hardware_acceleration=True,
        show_progress=True
    )
    
    clusterer = MSDRClusterer(best_config)
    result = clusterer.fit_predict(data)
    
    print(f"\n✅ Best Model Results:")
    print(f"   N clusters: {result.n_clusters}")
    print(f"   Success: {result.success}")
    
    unique, counts = np.unique(result.cluster_labels, return_counts=True)
    print(f"   Regime distribution:")
    for regime_id, count in zip(unique, counts):
        percentage = (count / len(result.cluster_labels)) * 100
        print(f"      Regime {regime_id}: {count} samples ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ AUTO-TUNER COMPLETE")
    print("=" * 80)
    
    return results, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MS-DR Auto-Tuner')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--timeout', type=float, default=None, help='Timeout in seconds')
    
    args = parser.parse_args()
    
    run_ms_dr_auto_tuner(n_trials=args.n_trials, timeout=args.timeout)

