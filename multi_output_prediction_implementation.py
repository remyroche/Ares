#!/usr/bin/env python3
"""
Multi-Output Prediction Implementation for Existing Project.

This module implements multi-output prediction (direction + profit magnitude)
that integrates with the existing project structure. It includes direct profit
prediction and fallback to profit-weighted training when direct prediction
is not possible.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
from src.utils.logger import get_logger

@dataclass
class MultiOutputConfig:
    """Configuration for multi-output prediction."""
    # Model types
    direction_model_type: str = "RandomForestClassifier"  # "RandomForestClassifier", "LogisticRegression"
    profit_model_type: str = "RandomForestRegressor"      # "RandomForestRegressor", "LinearRegression"
    
    # Model parameters
    direction_model_params: Dict = None
    profit_model_params: Dict = None
    
    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    use_time_series_split: bool = True
    n_splits: int = 5
    
    # Profit prediction parameters
    enable_direct_profit_prediction: bool = True
    min_profit_samples: int = 100  # Minimum samples needed for profit prediction
    profit_prediction_threshold: float = 0.001  # Minimum profit to consider for prediction
    
    # Fallback parameters
    enable_profit_weighting_fallback: bool = True
    profit_weight_power: float = 1.0  # Linear weighting
    min_profit_weight: float = 0.001
    
    # Ensemble parameters
    enable_ensemble: bool = True
    ensemble_method: str = "weighted_average"  # "weighted_average", "voting"
    
    # Feature selection
    enable_feature_selection: bool = True
    feature_selection_method: str = "importance"  # "importance", "correlation"
    max_features: int = 50

class MultiOutputPredictor:
    """Multi-output prediction system for direction and profit magnitude."""
    
    def __init__(self, config: MultiOutputConfig):
        self.config = config
        self.logger = get_logger("MultiOutputPredictor")
        
        # Set default model parameters
        if self.config.direction_model_params is None:
            self.config.direction_model_params = {
                'n_estimators': 100,
                'random_state': config.random_state,
                'class_weight': 'balanced'
            }
        
        if self.config.profit_model_params is None:
            self.config.profit_model_params = {
                'n_estimators': 100,
                'random_state': config.random_state
            }
        
        # Initialize models
        self.direction_model = None
        self.profit_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.training_summary = {}
        
    def _create_model(self, model_type: str, params: Dict):
        """Create model instance based on type and parameters."""
        if model_type == "RandomForestClassifier":
            return RandomForestClassifier(**params)
        elif model_type == "RandomForestRegressor":
            return RandomForestRegressor(**params)
        elif model_type == "LogisticRegression":
            return LogisticRegression(**params)
        elif model_type == "LinearRegression":
            return LinearRegression(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for multi-output training.
        
        Args:
            data: DataFrame with features, 'label', and 'potential_profit_pct'
            
        Returns:
            Tuple of (features, direction_labels, profit_labels)
        """
        # Filter out HOLD samples (label == 0)
        signal_data = data[data['label'] != 0].copy()
        
        if len(signal_data) == 0:
            raise ValueError("No signal data found for training")
        
        # Prepare features (exclude label and profit columns)
        exclude_cols = ['label', 'potential_profit_pct', 'timestamp', 'index']
        feature_cols = [col for col in signal_data.columns if col not in exclude_cols]
        
        X = signal_data[feature_cols]
        y_direction = (signal_data['label'] == 1).astype(int)  # Binary direction
        y_profit = signal_data['potential_profit_pct']  # Profit magnitude
        
        return X, y_direction, y_profit
    
    def _select_features(self, X: pd.DataFrame, y_direction: pd.Series, y_profit: pd.Series) -> pd.DataFrame:
        """
        Select most important features for training.
        
        Args:
            X: Feature matrix
            y_direction: Direction labels
            y_profit: Profit labels
            
        Returns:
            DataFrame with selected features
        """
        if not self.config.enable_feature_selection:
            return X
        
        if self.config.feature_selection_method == "importance":
            # Use Random Forest importance for feature selection
            temp_model = RandomForestClassifier(n_estimators=50, random_state=self.config.random_state)
            temp_model.fit(X, y_direction)
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': temp_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Select top features
            selected_features = importance_df.head(self.config.max_features)['feature'].tolist()
            
        elif self.config.feature_selection_method == "correlation":
            # Use correlation with target for feature selection
            correlations = []
            for col in X.columns:
                corr_direction = abs(X[col].corr(y_direction))
                corr_profit = abs(X[col].corr(y_profit))
                correlations.append(max(corr_direction, corr_profit))
            
            # Select features with highest correlation
            feature_corr_df = pd.DataFrame({
                'feature': X.columns,
                'correlation': correlations
            }).sort_values('correlation', ascending=False)
            
            selected_features = feature_corr_df.head(self.config.max_features)['feature'].tolist()
        
        else:
            selected_features = X.columns.tolist()
        
        self.logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")
        return X[selected_features]
    
    def _can_predict_profit(self, y_profit: pd.Series) -> bool:
        """
        Check if profit prediction is feasible.
        
        Args:
            y_profit: Profit labels
            
        Returns:
            True if profit prediction is feasible
        """
        if not self.config.enable_direct_profit_prediction:
            return False
        
        # Check minimum samples
        if len(y_profit) < self.config.min_profit_samples:
            self.logger.warning(f"Insufficient samples for profit prediction: {len(y_profit)} < {self.config.min_profit_samples}")
            return False
        
        # Check profit variance
        profit_variance = y_profit.var()
        if profit_variance < 1e-6:
            self.logger.warning("Insufficient profit variance for prediction")
            return False
        
        # Check profit range
        profit_range = y_profit.max() - y_profit.min()
        if profit_range < self.config.profit_prediction_threshold:
            self.logger.warning("Insufficient profit range for prediction")
            return False
        
        return True
    
    def train_multi_output_models(self, data: pd.DataFrame) -> Dict:
        """
        Train multi-output models for direction and profit prediction.
        
        Args:
            data: DataFrame with features, 'label', and 'potential_profit_pct'
            
        Returns:
            Dictionary with training results and model information
        """
        self.logger.info("🚀 Training multi-output prediction models...")
        
        # Prepare data
        X, y_direction, y_profit = self._prepare_data(data)
        
        # Feature selection
        X_selected = self._select_features(X, y_direction, y_profit)
        
        # Check if profit prediction is feasible
        can_predict_profit = self._can_predict_profit(y_profit)
        
        if can_predict_profit:
            self.logger.info("✅ Direct profit prediction enabled")
            return self._train_direct_profit_models(X_selected, y_direction, y_profit)
        else:
            self.logger.info("⚠️ Direct profit prediction not feasible, using profit-weighted fallback")
            return self._train_profit_weighted_fallback(X_selected, y_direction, y_profit)
    
    def _train_direct_profit_models(self, X: pd.DataFrame, y_direction: pd.Series, y_profit: pd.Series) -> Dict:
        """
        Train separate models for direction and profit prediction.
        
        Args:
            X: Feature matrix
            y_direction: Direction labels
            y_profit: Profit labels
            
        Returns:
            Dictionary with training results
        """
        # Split data
        if self.config.use_time_series_split:
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            splits = list(tscv.split(X))
        else:
            X_train, X_test, y_dir_train, y_dir_test, y_prof_train, y_prof_test = train_test_split(
                X, y_direction, y_profit, test_size=self.config.test_size, random_state=self.config.random_state
            )
            splits = [(np.arange(len(X_train)), np.arange(len(X_train), len(X)))]
        
        # Initialize models
        self.direction_model = self._create_model(
            self.config.direction_model_type, 
            self.config.direction_model_params
        )
        self.profit_model = self._create_model(
            self.config.profit_model_type, 
            self.config.profit_model_params
        )
        
        # Training results
        direction_scores = []
        profit_scores = []
        combined_scores = []
        
        for train_idx, test_idx in splits:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_dir_train, y_dir_test = y_direction.iloc[train_idx], y_direction.iloc[test_idx]
            y_prof_train, y_prof_test = y_profit.iloc[train_idx], y_profit.iloc[test_idx]
            
            # Train direction model
            self.direction_model.fit(X_train, y_dir_train)
            dir_pred = self.direction_model.predict(X_test)
            dir_accuracy = accuracy_score(y_dir_test, dir_pred)
            direction_scores.append(dir_accuracy)
            
            # Train profit model
            self.profit_model.fit(X_train, y_prof_train)
            prof_pred = self.profit_model.predict(X_test)
            prof_r2 = r2_score(y_prof_test, prof_pred)
            profit_scores.append(prof_r2)
            
            # Combined evaluation
            combined_score = self._evaluate_combined_predictions(
                dir_pred, prof_pred, y_dir_test, y_prof_test
            )
            combined_scores.append(combined_score)
        
        # Store feature importance
        if hasattr(self.direction_model, 'feature_importances_'):
            self.feature_importance['direction'] = dict(zip(X.columns, self.direction_model.feature_importances_))
        if hasattr(self.profit_model, 'feature_importances_'):
            self.feature_importance['profit'] = dict(zip(X.columns, self.profit_model.feature_importances_))
        
        # Training summary
        self.training_summary = {
            'method': 'direct_profit_prediction',
            'direction_accuracy_mean': np.mean(direction_scores),
            'direction_accuracy_std': np.std(direction_scores),
            'profit_r2_mean': np.mean(profit_scores),
            'profit_r2_std': np.std(profit_scores),
            'combined_score_mean': np.mean(combined_scores),
            'combined_score_std': np.std(combined_scores),
            'n_features': len(X.columns),
            'n_samples': len(X)
        }
        
        self.logger.info(f"✅ Direct profit prediction training completed")
        self.logger.info(f"   Direction accuracy: {self.training_summary['direction_accuracy_mean']:.4f} ± {self.training_summary['direction_accuracy_std']:.4f}")
        self.logger.info(f"   Profit R² score: {self.training_summary['profit_r2_mean']:.4f} ± {self.training_summary['profit_r2_std']:.4f}")
        self.logger.info(f"   Combined score: {self.training_summary['combined_score_mean']:.4f} ± {self.training_summary['combined_score_std']:.4f}")
        
        return self.training_summary
    
    def _train_profit_weighted_fallback(self, X: pd.DataFrame, y_direction: pd.Series, y_profit: pd.Series) -> Dict:
        """
        Train profit-weighted model as fallback when direct profit prediction is not feasible.
        
        Args:
            X: Feature matrix
            y_direction: Direction labels
            y_profit: Profit labels
            
        Returns:
            Dictionary with training results
        """
        if not self.config.enable_profit_weighting_fallback:
            raise ValueError("Direct profit prediction not feasible and profit weighting fallback is disabled")
        
        # Create sample weights based on profit magnitude
        sample_weights = np.abs(y_profit) ** self.config.profit_weight_power + self.config.min_profit_weight
        
        # Split data
        if self.config.use_time_series_split:
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            splits = list(tscv.split(X))
        else:
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y_direction, sample_weights, test_size=self.config.test_size, random_state=self.config.random_state
            )
            splits = [(np.arange(len(X_train)), np.arange(len(X_train), len(X)))]
        
        # Initialize model
        self.direction_model = self._create_model(
            self.config.direction_model_type, 
            self.config.direction_model_params
        )
        
        # Training results
        accuracy_scores = []
        weighted_accuracy_scores = []
        
        for train_idx, test_idx in splits:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_direction.iloc[train_idx], y_direction.iloc[test_idx]
            w_train, w_test = sample_weights.iloc[train_idx], sample_weights.iloc[test_idx]
            
            # Train with profit weighting
            self.direction_model.fit(X_train, y_train, sample_weight=w_train)
            y_pred = self.direction_model.predict(X_test)
            
            # Standard accuracy
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)
            
            # Weighted accuracy (higher weight for high-profit trades)
            weighted_accuracy = accuracy_score(y_test, y_pred, sample_weight=w_test)
            weighted_accuracy_scores.append(weighted_accuracy)
        
        # Store feature importance
        if hasattr(self.direction_model, 'feature_importances_'):
            self.feature_importance['direction'] = dict(zip(X.columns, self.direction_model.feature_importances_))
        
        # Training summary
        self.training_summary = {
            'method': 'profit_weighted_fallback',
            'accuracy_mean': np.mean(accuracy_scores),
            'accuracy_std': np.std(accuracy_scores),
            'weighted_accuracy_mean': np.mean(weighted_accuracy_scores),
            'weighted_accuracy_std': np.std(weighted_accuracy_scores),
            'n_features': len(X.columns),
            'n_samples': len(X),
            'avg_profit_weight': sample_weights.mean(),
            'profit_weight_range': (sample_weights.min(), sample_weights.max())
        }
        
        self.logger.info(f"✅ Profit-weighted fallback training completed")
        self.logger.info(f"   Standard accuracy: {self.training_summary['accuracy_mean']:.4f} ± {self.training_summary['accuracy_std']:.4f}")
        self.logger.info(f"   Weighted accuracy: {self.training_summary['weighted_accuracy_mean']:.4f} ± {self.training_summary['weighted_accuracy_std']:.4f}")
        self.logger.info(f"   Average profit weight: {self.training_summary['avg_profit_weight']:.4f}")
        
        return self.training_summary
    
    def _evaluate_combined_predictions(self, dir_pred: np.ndarray, prof_pred: np.ndarray, 
                                     y_dir_true: pd.Series, y_prof_true: pd.Series) -> float:
        """
        Evaluate combined direction and profit predictions.
        
        Args:
            dir_pred: Direction predictions
            prof_pred: Profit predictions
            y_dir_true: True direction labels
            y_prof_true: True profit labels
            
        Returns:
            Combined evaluation score
        """
        # Calculate profit-weighted accuracy
        profit_weights = np.abs(y_prof_true) + 0.001
        weighted_accuracy = accuracy_score(y_dir_true, dir_pred, sample_weight=profit_weights)
        
        # Calculate profit prediction accuracy for high-profit trades
        high_profit_mask = y_prof_true > 0.01  # >1% profit
        if high_profit_mask.sum() > 0:
            high_profit_accuracy = accuracy_score(
                y_dir_true[high_profit_mask], 
                dir_pred[high_profit_mask]
            )
        else:
            high_profit_accuracy = 0.0
        
        # Combined score (weighted average)
        combined_score = 0.6 * weighted_accuracy + 0.4 * high_profit_accuracy
        
        return combined_score
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Make predictions using trained models.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Dictionary with predictions
        """
        if self.direction_model is None:
            raise ValueError("Models not trained. Call train_multi_output_models first.")
        
        # Ensure feature consistency
        if hasattr(self.direction_model, 'feature_names_in_'):
            X = X[self.direction_model.feature_names_in_]
        
        # Make predictions
        direction_pred = self.direction_model.predict(X)
        direction_proba = self.direction_model.predict_proba(X) if hasattr(self.direction_model, 'predict_proba') else None
        
        predictions = {
            'direction': direction_pred,
            'direction_proba': direction_proba,
            'method': self.training_summary.get('method', 'unknown')
        }
        
        # Add profit predictions if available
        if self.profit_model is not None:
            profit_pred = self.profit_model.predict(X)
            predictions['profit'] = profit_pred
            
            # Combined confidence score
            if direction_proba is not None:
                confidence = np.max(direction_proba, axis=1)
                predictions['confidence'] = confidence
                
                # High-value trade indicator
                high_value_trades = (
                    (direction_pred == 1) & (profit_pred > 0.02)  # BUY with >2% expected profit
                ) | (
                    (direction_pred == 0) & (profit_pred < -0.01)  # SELL with >1% expected profit
                )
                predictions['high_value_trades'] = high_value_trades
        
        return predictions
    
    def save_models(self, save_path: str):
        """Save trained models to disk."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save direction model
        if self.direction_model is not None:
            joblib.dump(self.direction_model, os.path.join(save_path, 'direction_model.pkl'))
        
        # Save profit model
        if self.profit_model is not None:
            joblib.dump(self.profit_model, os.path.join(save_path, 'profit_model.pkl'))
        
        # Save metadata
        metadata = {
            'training_summary': self.training_summary,
            'feature_importance': self.feature_importance,
            'config': self.config
        }
        joblib.dump(metadata, os.path.join(save_path, 'metadata.pkl'))
        
        self.logger.info(f"✅ Models saved to {save_path}")
    
    def load_models(self, load_path: str):
        """Load trained models from disk."""
        # Load direction model
        direction_model_path = os.path.join(load_path, 'direction_model.pkl')
        if os.path.exists(direction_model_path):
            self.direction_model = joblib.load(direction_model_path)
        
        # Load profit model
        profit_model_path = os.path.join(load_path, 'profit_model.pkl')
        if os.path.exists(profit_model_path):
            self.profit_model = joblib.load(profit_model_path)
        
        # Load metadata
        metadata_path = os.path.join(load_path, 'metadata.pkl')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.training_summary = metadata.get('training_summary', {})
            self.feature_importance = metadata.get('feature_importance', {})
        
        self.logger.info(f"✅ Models loaded from {load_path}")

# Integration with existing pipeline
class MultiOutputIntegration:
    """Integration class for multi-output prediction with existing pipeline."""
    
    def __init__(self, config: MultiOutputConfig):
        self.config = config
        self.predictor = MultiOutputPredictor(config)
        self.logger = get_logger("MultiOutputIntegration")
    
    def integrate_with_existing_pipeline(self, labeled_data: pd.DataFrame, 
                                       model_save_path: str = None) -> Dict:
        """
        Integrate multi-output prediction with existing triple barrier pipeline.
        
        Args:
            labeled_data: DataFrame with triple barrier labels and profit tracking
            model_save_path: Path to save trained models
            
        Returns:
            Dictionary with integration results
        """
        self.logger.info("🔗 Integrating multi-output prediction with existing pipeline...")
        
        # Verify profit tracking data exists
        if 'potential_profit_pct' not in labeled_data.columns:
            raise ValueError("Profit tracking data not found. Run triple barrier with include_profit_tracking=True")
        
        # Train multi-output models
        training_results = self.predictor.train_multi_output_models(labeled_data)
        
        # Save models if path provided
        if model_save_path:
            self.predictor.save_models(model_save_path)
        
        # Generate integration report
        integration_report = {
            'training_results': training_results,
            'model_method': training_results.get('method', 'unknown'),
            'feature_count': training_results.get('n_features', 0),
            'sample_count': training_results.get('n_samples', 0),
            'can_predict_profit': training_results.get('method') == 'direct_profit_prediction'
        }
        
        self.logger.info(f"✅ Multi-output integration completed")
        self.logger.info(f"   Method: {integration_report['model_method']}")
        self.logger.info(f"   Features: {integration_report['feature_count']}")
        self.logger.info(f"   Samples: {integration_report['sample_count']}")
        self.logger.info(f"   Profit prediction: {integration_report['can_predict_profit']}")
        
        return integration_report
    
    def predict_on_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            new_data: New data for prediction
            
        Returns:
            DataFrame with predictions added
        """
        # Make predictions
        predictions = self.predictor.predict(new_data)
        
        # Add predictions to data
        result_data = new_data.copy()
        result_data['predicted_direction'] = predictions['direction']
        
        if 'profit' in predictions:
            result_data['predicted_profit'] = predictions['profit']
        
        if 'confidence' in predictions:
            result_data['prediction_confidence'] = predictions['confidence']
        
        if 'high_value_trades' in predictions:
            result_data['high_value_trade'] = predictions['high_value_trades']
        
        return result_data

# Example usage and integration
def demonstrate_multi_output_integration():
    """Demonstrate multi-output prediction integration."""
    
    print("🎯 Multi-Output Prediction Integration Demonstration")
    print("=" * 60)
    
    # Configuration
    config = MultiOutputConfig(
        direction_model_type="RandomForestClassifier",
        profit_model_type="RandomForestRegressor",
        enable_direct_profit_prediction=True,
        enable_profit_weighting_fallback=True,
        use_time_series_split=True,
        enable_feature_selection=True
    )
    
    # Create integration instance
    integration = MultiOutputIntegration(config)
    
    print("✅ Multi-output prediction integration ready for use")
    print("\n📋 Integration Features:")
    print("1. Direct profit prediction when feasible")
    print("2. Profit-weighted fallback when direct prediction not possible")
    print("3. Time-series cross-validation")
    print("4. Feature selection and importance analysis")
    print("5. Combined direction + profit predictions")
    print("6. High-value trade identification")
    print("7. Model persistence and loading")
    
    return integration

if __name__ == "__main__":
    demonstrate_multi_output_integration()