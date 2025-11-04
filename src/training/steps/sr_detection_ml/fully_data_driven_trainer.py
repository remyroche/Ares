"""
Fully Data-Driven SR Training System

Orchestrates complete zero-heuristic training pipeline.
Combines all components into end-to-end automated system.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
import shap
import lightgbm as lgb

from src.training.steps.sr_detection_ml.sr_data_collector import SRDataCollector
from src.training.steps.sr_detection_ml.lgbm_shap_feature_selector import LGBMShapFeatureSelector
from src.training.steps.sr_detection_ml.fast_target_selector import FastTargetSelector
from src.training.steps.sr_detection_ml.hpo_trainer import HPOTrainer
from src.training.steps.sr_detection_ml.utils.report_generator import SRMLReportGenerator
from src.training.steps.sr_detection_ml.data_leakage_checker import DataLeakageChecker
from src.training.steps.sr_detection_ml.validation_safeguards import ValidationSafeguards
from src.training.steps.sr_detection_ml.multicollinearity_remover import MulticollinearityRemover

# tprint for progress tracking
try:
    from src.utils.tprint import tprint
except ImportError:
    tprint = print

logger = logging.getLogger(__name__)


class FullyDataDrivenSRSystem:
    """
    Complete 100% data-driven SR level ML system.
    
    Pipeline:
    1. Collect raw data with ALL candidates
    2. Generate exhaustive features (300-500)
    3. Generate all possible targets (50-100)
    4. Select features via LGBM+SHAP
    5. Find best target via AutoML
    6. Optimize hyperparameters via HPO
    7. Train final model with SHAP analysis
    
    Zero heuristics, zero manual tuning.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.collector = SRDataCollector(fast_mode=True)
        self.feature_selector = LGBMShapFeatureSelector(n_splits=5)
        self.target_selector = FastTargetSelector(n_splits=3, top_k=10)  # Fast: only eval top 10
        self.hpo_trainer = HPOTrainer(n_trials=200)
        self.report_generator = SRMLReportGenerator()
        self.leakage_checker = DataLeakageChecker()
        self.validation_safeguards = ValidationSafeguards()
        self.multicollinearity_remover = MulticollinearityRemover(perfect_threshold=0.999, high_threshold=0.95)
        
        # Storage for results
        self.results = {}
        self.raw_data = None  # Store for report generation
    
    def train_from_scratch(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        n_features: int = 50,
        sample_every_n_bars: int = 10
    ) -> Dict[str, Any]:
        """
        Train complete SR ML system from scratch.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            exchange: Exchange name (e.g., 'binance')
            timeframe: Timeframe (e.g., '1h')
            start_date: Start date for training data
            end_date: End date for training data
            n_features: Number of features to select via SHAP
            sample_every_n_bars: Sampling frequency
        
        Returns:
            Dictionary with trained model and analysis results
        """
        tprint("=" * 80)
        tprint("🚀 FULLY DATA-DRIVEN SR LEVEL ML SYSTEM")
        tprint("=" * 80)
        tprint(f"Symbol: {symbol} {exchange} {timeframe}")
        tprint(f"Period: {start_date} to {end_date}")
        tprint(f"Target features: {n_features}")
        tprint("=" * 80)
        
        # Step 1: Collect raw data
        tprint("\n📊 STEP 1: DATA COLLECTION")
        tprint("-" * 80)
        
        raw_data = self.collector.collect_training_data(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            sample_every_n_bars=sample_every_n_bars
        )
        
        # Store for report generation
        self.raw_data = raw_data
        
        # Step 2: Extract features and targets
        tprint("\n🔍 STEP 2: FEATURE & TARGET EXTRACTION")
        tprint("-" * 80)
        
        feature_cols, target_cols = self._identify_columns(raw_data)
        
        tprint(f"✅ Identified {len(feature_cols)} features, {len(target_cols)} targets")
        
        X_raw = raw_data[feature_cols].copy()
        targets_df = raw_data[target_cols].copy()
        
        # Handle missing values
        X_raw = X_raw.fillna(0)
        
        # CRITICAL: Check dataset size BEFORE proceeding
        if len(raw_data) < 500:
            self.logger.warning(
                f"⚠️ WARNING: Only {len(raw_data)} samples collected. "
                f"Recommended minimum: 500 samples for stable training. "
                f"Results may be unreliable!"
            )
        
        # Step 3: Remove multicollinear features
        tprint("\n🧹 STEP 3: MULTICOLLINEARITY REMOVAL")
        tprint("-" * 80)
        
        X_cleaned, mcol_report = self.multicollinearity_remover.detect_and_remove(
            X_raw, remove_perfect_only=True  # Remove perfect correlations only
        )
        
        if mcol_report['removed_count'] > 0:
            tprint(f"⚠️ Removed {mcol_report['removed_count']} features with perfect correlation (r >= 0.999)")
            tprint(f"   Perfect correlation pairs: {mcol_report['perfect_correlations']}")
        else:
            tprint("✅ No multicollinearity detected")
        
        # Step 4: Feature selection via LGBM+SHAP
        tprint("\n⚡ STEP 4: FEATURE SELECTION (LGBM+SHAP)")
        tprint("-" * 80)
        
        # Use first target for initial feature selection
        initial_target = targets_df.iloc[:, 0].fillna(0)
        
        selected_features, shap_importance = self.feature_selector.select_features_by_shap_importance(
            X_cleaned, initial_target, n_features=n_features
        )
        
        X_selected = X_cleaned[selected_features]
        
        # Step 5: Find best target via Fast ML-based selection
        tprint("\n🎯 STEP 5: FAST TARGET SELECTION (ML-Based)")
        tprint("-" * 80)
        
        best_target, target_analysis = self.target_selector.find_best_target_fast(
            X_selected, targets_df
        )
        
        y = targets_df[best_target].fillna(0)
        
        # Step 6: Train/val split (time series)
        tprint("\n✂️  STEP 6: TRAIN/VAL SPLIT")
        tprint("-" * 80)
        
        split_idx = int(len(X_selected) * 0.8)
        
        X_train = X_selected.iloc[:split_idx]
        X_val = X_selected.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]
        
        tprint(f"✅ Train: {len(X_train):,} samples")
        tprint(f"✅ Val:   {len(X_val):,} samples")
        
        # Step 7: HPO for optimal hyperparameters
        tprint("\n🔧 STEP 7: HYPERPARAMETER OPTIMIZATION")
        tprint("-" * 80)
        
        final_model, best_params = self.hpo_trainer.train_optimized_model(
            X_train, y_train, X_val, y_val
        )
        
        # Step 8: SHAP analysis
        tprint("\n🔬 STEP 8: SHAP INTERPRETABILITY ANALYSIS")
        tprint("-" * 80)
        
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_val)
        
        tprint(f"✅ SHAP analysis complete for {len(X_val)} validation samples")
        
        # CRITICAL: Run validation safeguards
        tprint("\n🛡️ STEP 8.5: VALIDATION SAFEGUARDS")
        tprint("-" * 80)
        
        # Check for data leakage
        leakage_report = self.leakage_checker.check_for_leakage(
            raw_data, feature_cols, target_cols, best_target
        )
        
        if leakage_report['critical_issues']:
            self.logger.error("🚨 DATA LEAKAGE DETECTED!")
            for issue in leakage_report['critical_issues']:
                self.logger.error(f"   {issue}")
        
        if leakage_report['warnings']:
            for warning in leakage_report['warnings']:
                self.logger.warning(f"   {warning}")
        
        # Check for suspicious results
        validation_report = self.validation_safeguards.validate_results(
            final_model, X_train, y_train, X_val, y_val,
            best_params, shap_values
        )
        
        if validation_report['critical_issues']:
            self.logger.error("🚨 VALIDATION ISSUES DETECTED!")
            for issue in validation_report['critical_issues']:
                self.logger.error(f"   {issue}")
        
        if validation_report['warnings']:
            for warning in validation_report['warnings']:
                self.logger.warning(f"   {warning}")
        
        # Store validation results
        validation_results = {
            'leakage_check': leakage_report,
            'validation_check': validation_report,
            'safe_to_use': (
                not leakage_report['has_critical_issues'] and
                validation_report['safe_to_use']
            )
        }
        
        # Calculate feature importance from SHAP
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Log top features
        top_10_idx = np.argsort(feature_importance)[-10:]
        self.logger.info(f"\n   Top 10 most important features (by SHAP):")
        for rank, idx in enumerate(reversed(top_10_idx), 1):
            feature = selected_features[idx]
            importance = feature_importance[idx]
            self.logger.info(f"      {rank}. {feature}: {importance:.6f}")
        
        # Step 9: Compile results
        self.logger.info("\n📦 STEP 9: COMPILING RESULTS")
        self.logger.info("-" * 80)
        
        results = {
            'model': final_model,
            'explainer': explainer,
            'selected_features': selected_features,
            'best_target': best_target,
            'target_analysis': target_analysis,
            'best_params': best_params,
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'metadata': {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'n_samples_total': len(raw_data),
                'n_samples_train': len(X_train),
                'n_samples_val': len(X_val),
                'n_features_raw': len(feature_cols),
                'n_features_after_mcol_removal': len(X_cleaned.columns),
                'n_features_removed_mcol': mcol_report['removed_count'],
                'n_features_selected': len(selected_features),
                'n_targets_evaluated': len(target_cols),
                'best_target_name': best_target,
                'best_target_r2': target_analysis[best_target]['mean_r2'],
                'training_timestamp': datetime.now().isoformat(),
                'val_r2': final_model.score(X_val, y_val),
                'train_r2': final_model.score(X_train, y_train)
            },
            'validation': validation_results
        }
        
        # Save results
        self._save_results(results, symbol, exchange, timeframe)
        
        # Generate comprehensive markdown report
        tprint("\n📝 STEP 10: GENERATING COMPREHENSIVE REPORT")
        tprint("-" * 80)
        
        try:
            report_path = self.report_generator.generate_comprehensive_report(
                results, self.raw_data
            )
            self.logger.info(f"✅ Comprehensive report generated: {report_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate report: {e}")
        
        tprint("=" * 80)
        tprint("✅ TRAINING COMPLETE!")
        tprint("=" * 80)
        tprint(f"   Model: {final_model.__class__.__name__}")
        tprint(f"   Best Target: {best_target}")
        tprint(f"   Val R²: {final_model.score(X_val, y_val):.4f}")
        tprint(f"   Features: {len(selected_features)}")
        tprint(f"   Hyperparameters: Optimized via {self.hpo_trainer.n_trials} trials")
        tprint("=" * 80)
        
        self.results = results
        return results
    
    def _identify_columns(self, df: pd.DataFrame) -> tuple:
        """
        Identify feature and target columns with strict separation to prevent leakage.
        
        Args:
            df: Training data DataFrame
        
        Returns:
            Tuple of (feature_cols, target_cols)
        """
        # Define target prefixes FIRST (more specific)
        target_prefixes = [
            'max_', 'touch_', 'break_', 'reversal_', 
            'vol_change', 'vol_spike',  # SPECIFIC vol_ targets
            'volume_surge', 'volume_spike',  # SPECIFIC volume_ targets
            'net_move', 'bars_to'
        ]
        
        # Feature prefixes (more general, but exclude targets)
        feature_prefixes = [
            'dist_', 'crosses_', 'ret_', 'range_', 
            'atr_', 'time_at_', 'close_', 'cross_rate',
            'vol_mean_', 'vol_std_', 'vol_median_', 'vol_min_', 'vol_max_',  # vol_ FEATURES only
            'vol_near_', 'vol_skew_', 'vol_kurt_', 'vol_ratio_',  # More vol_ FEATURES
            'volatility_ratio_', 'volatility_norm_'  # volatility features
        ]
        
        # First, identify ALL columns that match target prefixes
        target_cols = [
            c for c in df.columns 
            if any(c.startswith(p) for p in target_prefixes)
        ]
        
        # Then, identify feature columns EXCLUDING any that are targets
        feature_cols = [
            c for c in df.columns 
            if any(c.startswith(p) for p in feature_prefixes) and c not in target_cols
        ]
        
        # CRITICAL: Double-check no targets leaked into features
        leaked_targets = set(feature_cols) & set(target_cols)
        if leaked_targets:
            self.logger.error(f"🚨 CRITICAL: Target leakage detected! {leaked_targets}")
            raise ValueError(f"Target columns leaked into features: {leaked_targets}")
        
        self.logger.info(f"📊 Column identification: {len(feature_cols)} features, {len(target_cols)} targets")
        
        return feature_cols, target_cols
    
    def _save_results(
        self,
        results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ):
        """
        Save training results to disk.
        
        Args:
            results: Results dictionary
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
        """
        try:
            # Create output directory
            output_dir = Path("models/sr_ml")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prefix = f"sr_ml_{symbol}_{exchange}_{timeframe}_{timestamp}"
            
            # Save model
            model_path = output_dir / f"{prefix}_model.txt"
            results['model'].booster_.save_model(str(model_path))
            self.logger.info(f"✅ Model saved: {model_path}")
            
            # Save metadata
            metadata_path = output_dir / f"{prefix}_metadata.json"
            metadata_serializable = {
                k: v for k, v in results['metadata'].items()
                if not isinstance(v, (np.ndarray, pd.DataFrame))
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata_serializable, f, indent=2)
            self.logger.info(f"✅ Metadata saved: {metadata_path}")
            
            # Save feature list
            features_path = output_dir / f"{prefix}_features.json"
            with open(features_path, 'w') as f:
                json.dump({
                    'selected_features': results['selected_features'],
                    'feature_importance': results['feature_importance'].tolist()
                }, f, indent=2)
            self.logger.info(f"✅ Features saved: {features_path}")
            
            # Save target analysis
            target_analysis_path = output_dir / f"{prefix}_target_analysis.json"
            target_analysis_serializable = {
                k: {
                    metric: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for metric, v in metrics.items()
                }
                for k, metrics in results['target_analysis'].items()
            }
            with open(target_analysis_path, 'w') as f:
                json.dump(target_analysis_serializable, f, indent=2)
            self.logger.info(f"✅ Target analysis saved: {target_analysis_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save some results: {e}")

