"""
Stacked Outcome Predictor - Two-Stage Model

Stage 1: Classifier predicts outcome type (Bounce, Break, or Chop)
Stage 2: Specialized regressors predict specific metrics for each outcome type

This simplifies the learning task by training separate experts for each outcome type.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class StackedOutcomePredictor:
    """
    Two-stage stacked model for S/R level outcome prediction.
    
    Stage 1: Outcome Type Classifier
        - Predicts: "Bounce", "Break", or "Chop"
        - Based on: price behavior after level is reached
    
    Stage 2: Specialized Regressors
        - Bounce Regressor: Predicts reversal strength for bounce events
        - Break Regressor: Predicts breakout magnitude for break events
        - Chop Regressor: Predicts consolidation metrics for chop events
    
    Benefits:
    - Each model specializes in its outcome type (no confusion from other types)
    - More interpretable (SHAP shows what makes a bounce vs break)
    - Better performance on imbalanced data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Stage 1: Classifier
        self.classifier = None
        self.classifier_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'force_col_wise': True,
            'n_estimators': 200,
            'random_state': 42
        }
        
        # Stage 2: Specialized Regressors
        self.bounce_regressor = None
        self.break_regressor = None
        self.chop_regressor = None
        
        self.regressor_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'force_col_wise': True,
            'n_estimators': 200,
            'random_state': 42
        }
        
        # Label mappings
        self.outcome_labels = {0: 'Chop', 1: 'Bounce', 2: 'Break'}
        self.label_to_idx = {'Chop': 0, 'Bounce': 1, 'Break': 2}
    
    def _classify_outcomes(
        self, 
        targets_df: pd.DataFrame
    ) -> pd.Series:
        """
        Classify each sample into Bounce, Break, or Chop.
        
        Logic:
        - Break: If max_abs_move > 2% in any 50-bar window
        - Bounce: If reversal_strength > 1% and no break
        - Chop: Otherwise (low volatility, no clear direction)
        
        Args:
            targets_df: DataFrame with all target columns
        
        Returns:
            Series with outcome classifications (0=Chop, 1=Bounce, 2=Break)
        """
        outcomes = pd.Series(0, index=targets_df.index)  # Default: Chop
        
        # Identify breaks (strong directional moves)
        if 'max_abs_move_50' in targets_df.columns:
            is_break = targets_df['max_abs_move_50'].abs() > 0.02  # 2% threshold
            outcomes[is_break] = 2  # Break
        elif 'max_abs_move_20' in targets_df.columns:
            is_break = targets_df['max_abs_move_20'].abs() > 0.02
            outcomes[is_break] = 2
        
        # Identify bounces (reversals that didn't break)
        if 'reversal_strength_50' in targets_df.columns:
            is_bounce = (
                (targets_df['reversal_strength_50'] > 0.01) &  # 1% reversal
                (outcomes == 0)  # Not already classified as break
            )
            outcomes[is_bounce] = 1  # Bounce
        elif 'reversal_strength_20' in targets_df.columns:
            is_bounce = (
                (targets_df['reversal_strength_20'] > 0.01) &
                (outcomes == 0)
            )
            outcomes[is_bounce] = 1
        
        # Log distribution
        counts = outcomes.value_counts().sort_index()
        self.logger.info(f"   Outcome distribution:")
        for label_idx, count in counts.items():
            label_name = self.outcome_labels[label_idx]
            pct = count / len(outcomes) * 100
            self.logger.info(f"      {label_name}: {count} ({pct:.1f}%)")
        
        return outcomes
    
    def _get_target_for_outcome(
        self,
        outcome_type: str,
        targets_df: pd.DataFrame
    ) -> str:
        """
        Get the best target column for a specific outcome type.
        
        Args:
            outcome_type: 'Bounce', 'Break', or 'Chop'
            targets_df: DataFrame with all target columns
        
        Returns:
            Name of best target column for this outcome
        """
        if outcome_type == 'Bounce':
            # For bounces, predict reversal strength
            candidates = [c for c in targets_df.columns if 'reversal_strength' in c]
            if candidates:
                return candidates[0]
        elif outcome_type == 'Break':
            # For breaks, predict breakout magnitude
            candidates = [c for c in targets_df.columns if 'max_abs_move' in c]
            if candidates:
                return candidates[0]
        else:  # Chop
            # For chop, predict average distance (consolidation)
            candidates = [c for c in targets_df.columns if 'avg_dist' in c]
            if candidates:
                return candidates[0]
        
        # Fallback: use first available target
        return targets_df.columns[0]
    
    def train(
        self,
        X_train: pd.DataFrame,
        targets_df_train: pd.DataFrame,
        X_val: pd.DataFrame,
        targets_df_val: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Train the two-stage stacked model.
        
        Args:
            X_train: Training features
            targets_df_train: All training targets
            X_val: Validation features
            targets_df_val: All validation targets
        
        Returns:
            Dictionary with training results
        """
        self.logger.info("🎯 Training Two-Stage Stacked Outcome Predictor")
        self.logger.info("=" * 60)
        
        # Stage 1: Train outcome classifier
        self.logger.info("\n📊 STAGE 1: Training Outcome Type Classifier")
        self.logger.info("-" * 60)
        
        y_class_train = self._classify_outcomes(targets_df_train)
        y_class_val = self._classify_outcomes(targets_df_val)
        
        self.classifier = lgb.LGBMClassifier(**self.classifier_params)
        self.classifier.fit(X_train, y_class_train)
        
        # Evaluate classifier
        train_acc = self.classifier.score(X_train, y_class_train)
        val_acc = self.classifier.score(X_val, y_class_val)
        
        self.logger.info(f"✅ Classifier trained!")
        self.logger.info(f"   Train Accuracy: {train_acc:.4f}")
        self.logger.info(f"   Val Accuracy: {val_acc:.4f}")
        
        # Stage 2: Train specialized regressors
        self.logger.info("\n🎯 STAGE 2: Training Specialized Regressors")
        self.logger.info("-" * 60)
        
        results = {
            'stage1_classifier': {
                'model': self.classifier,
                'train_acc': train_acc,
                'val_acc': val_acc
            },
            'stage2_regressors': {}
        }
        
        # Train regressor for each outcome type
        for outcome_idx, outcome_name in self.outcome_labels.items():
            self.logger.info(f"\n   Training {outcome_name} Regressor...")
            
            # Filter to samples of this outcome type
            train_mask = (y_class_train == outcome_idx)
            val_mask = (y_class_val == outcome_idx)
            
            if train_mask.sum() < 10:
                self.logger.warning(f"      ⚠️ Too few {outcome_name} samples ({train_mask.sum()}), skipping")
                continue
            
            X_train_subset = X_train[train_mask]
            X_val_subset = X_val[val_mask] if val_mask.sum() > 0 else X_val[:5]  # Dummy if none
            
            # Get target for this outcome type
            target_col = self._get_target_for_outcome(outcome_name, targets_df_train)
            y_train_subset = targets_df_train.loc[train_mask, target_col].fillna(0)
            y_val_subset = targets_df_val.loc[val_mask, target_col].fillna(0) if val_mask.sum() > 0 else targets_df_val[target_col][:5]
            
            # Train regressor
            regressor = lgb.LGBMRegressor(**self.regressor_params)
            regressor.fit(X_train_subset, y_train_subset)
            
            # Evaluate
            train_r2 = regressor.score(X_train_subset, y_train_subset)
            val_r2 = regressor.score(X_val_subset, y_val_subset) if val_mask.sum() > 5 else 0.0
            
            self.logger.info(f"      ✅ {outcome_name} Regressor trained!")
            self.logger.info(f"         Samples: {train_mask.sum()} train, {val_mask.sum()} val")
            self.logger.info(f"         Target: {target_col}")
            self.logger.info(f"         Train R²: {train_r2:.4f}")
            self.logger.info(f"         Val R²: {val_r2:.4f}")
            
            # Store regressor
            if outcome_name == 'Bounce':
                self.bounce_regressor = regressor
            elif outcome_name == 'Break':
                self.break_regressor = regressor
            else:  # Chop
                self.chop_regressor = regressor
            
            results['stage2_regressors'][outcome_name] = {
                'model': regressor,
                'target_col': target_col,
                'n_train_samples': train_mask.sum(),
                'n_val_samples': val_mask.sum(),
                'train_r2': train_r2,
                'val_r2': val_r2
            }
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ Two-Stage Stacked Model Training Complete!")
        self.logger.info("=" * 60)
        
        return results
    
    def predict(
        self,
        X: pd.DataFrame,
        return_outcome_probs: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions using the stacked model.
        
        Args:
            X: Features to predict on
            return_outcome_probs: If True, also returns outcome probabilities
        
        Returns:
            Dictionary with predictions for each outcome type
        """
        if self.classifier is None:
            raise ValueError("Model not trained! Call train() first.")
        
        # Stage 1: Predict outcome type
        outcome_probs = self.classifier.predict_proba(X)
        outcome_preds = self.classifier.predict(X)
        
        # Stage 2: Predict with specialized regressors
        predictions = {
            'outcome_type': outcome_preds,
            'bounce_strength': np.zeros(len(X)),
            'break_magnitude': np.zeros(len(X)),
            'chop_consolidation': np.zeros(len(X))
        }
        
        if return_outcome_probs:
            predictions['outcome_probs'] = outcome_probs
        
        # Predict for each outcome type
        for idx, outcome_name in self.outcome_labels.items():
            mask = (outcome_preds == idx)
            
            if not mask.any():
                continue
            
            X_subset = X[mask]
            
            # Use appropriate regressor
            if outcome_name == 'Bounce' and self.bounce_regressor is not None:
                pred_values = self.bounce_regressor.predict(X_subset)
                predictions['bounce_strength'][mask] = pred_values
            elif outcome_name == 'Break' and self.break_regressor is not None:
                pred_values = self.break_regressor.predict(X_subset)
                predictions['break_magnitude'][mask] = pred_values
            elif outcome_name == 'Chop' and self.chop_regressor is not None:
                pred_values = self.chop_regressor.predict(X_subset)
                predictions['chop_consolidation'][mask] = pred_values
        
        return predictions
    
    def get_models(self) -> Dict[str, Any]:
        """
        Get all trained models.
        
        Returns:
            Dictionary with classifier and regressors
        """
        return {
            'classifier': self.classifier,
            'bounce_regressor': self.bounce_regressor,
            'break_regressor': self.break_regressor,
            'chop_regressor': self.chop_regressor
        }

