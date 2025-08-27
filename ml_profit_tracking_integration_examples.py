#!/usr/bin/env python3
"""
Examples of how ML models can learn from the new profit tracking information.

This script demonstrates various ways to integrate profit tracking data into
machine learning model training for enhanced performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class ProfitTrackingMLIntegration:
    """Demonstrates various ways to integrate profit tracking into ML training."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def create_sample_data_with_profits(self, n_samples: int = 10000) -> pd.DataFrame:
        """Create sample data with profit tracking information."""
        np.random.seed(42)
        
        # Generate synthetic features
        data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),  # Technical indicator
            'feature_2': np.random.normal(0, 1, n_samples),  # Market sentiment
            'feature_3': np.random.normal(0, 1, n_samples),  # Volatility
            'feature_4': np.random.normal(0, 1, n_samples),  # Volume
            'feature_5': np.random.normal(0, 1, n_samples),  # Price momentum
        })
        
        # Generate labels (traditional triple barrier)
        # Simulate that some features predict direction
        signal_strength = (
            0.3 * data['feature_1'] + 
            0.2 * data['feature_2'] + 
            0.1 * data['feature_3'] +
            np.random.normal(0, 0.5, n_samples)
        )
        
        labels = np.where(signal_strength > 0.1, 1, 
                         np.where(signal_strength < -0.1, -1, 0))
        
        # Generate profit tracking data
        # Simulate that profit magnitude is related to feature combinations
        profit_potential = (
            0.4 * data['feature_1'] * data['feature_2'] +  # Interaction effect
            0.3 * data['feature_3'] +                      # Volatility impact
            0.2 * data['feature_4'] +                      # Volume impact
            np.random.normal(0, 0.3, n_samples)            # Noise
        )
        
        # Add some realistic constraints
        profit_pct = np.clip(profit_potential, -0.05, 0.10)  # -5% to +10%
        
        # Only assign profits to actual signals (non-zero labels)
        profit_pct = np.where(labels != 0, profit_pct, 0.0)
        
        data['label'] = labels
        data['potential_profit_pct'] = profit_pct
        
        return data
    
    def method_1_binary_classification_with_profit_weighting(self, data: pd.DataFrame) -> Dict:
        """
        Method 1: Weight samples by profit magnitude in binary classification.
        
        This approach gives higher importance to samples with higher profit potential
        during model training.
        """
        print("🔍 Method 1: Binary Classification with Profit Weighting")
        print("=" * 60)
        
        # Filter out HOLD samples (label == 0) for binary classification
        signal_data = data[data['label'] != 0].copy()
        
        if len(signal_data) == 0:
            print("❌ No signals found for binary classification")
            return {}
        
        # Create sample weights based on profit magnitude
        # Higher profit = higher weight
        sample_weights = np.abs(signal_data['potential_profit_pct']) + 0.001  # Add small constant
        
        # Prepare features and labels
        X = signal_data[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']]
        y = (signal_data['label'] == 1).astype(int)  # Convert to binary (1=BUY, 0=SELL)
        
        # Split data
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42
        )
        
        # Train model with profit-weighted samples
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train, sample_weight=w_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        
        print(f"✅ Model trained with profit-weighted samples")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Sample weights range: {w_train.min():.4f} - {w_train.max():.4f}")
        print(f"   Average weight: {w_train.mean():.4f}")
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Feature Importance (Profit-Weighted):")
        for _, row in feature_importance.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return {
            'method': 'profit_weighted_classification',
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'model': model
        }
    
    def method_2_profit_regression(self, data: pd.DataFrame) -> Dict:
        """
        Method 2: Direct profit prediction using regression.
        
        Train a model to predict the actual profit/loss percentage directly.
        """
        print("\n💰 Method 2: Direct Profit Prediction (Regression)")
        print("=" * 60)
        
        # Use only signal samples (non-zero labels)
        signal_data = data[data['label'] != 0].copy()
        
        if len(signal_data) == 0:
            print("❌ No signals found for profit regression")
            return {}
        
        # Prepare features and target
        X = signal_data[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']]
        y = signal_data['potential_profit_pct']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train regression model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Profit prediction model trained")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MSE: {mse:.6f}")
        print(f"   Average predicted profit: {y_pred.mean():.4f}")
        print(f"   Actual average profit: {y_test.mean():.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Feature Importance (Profit Prediction):")
        for _, row in feature_importance.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return {
            'method': 'profit_regression',
            'r2_score': r2,
            'mse': mse,
            'feature_importance': feature_importance,
            'model': model
        }
    
    def method_3_multi_output_prediction(self, data: pd.DataFrame) -> Dict:
        """
        Method 3: Multi-output prediction (direction + profit magnitude).
        
        Train a model to predict both signal direction and profit magnitude simultaneously.
        """
        print("\n🎯 Method 3: Multi-Output Prediction (Direction + Profit)")
        print("=" * 60)
        
        # Use only signal samples
        signal_data = data[data['label'] != 0].copy()
        
        if len(signal_data) == 0:
            print("❌ No signals found for multi-output prediction")
            return {}
        
        # Prepare features and targets
        X = signal_data[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']]
        y_direction = (signal_data['label'] == 1).astype(int)  # Binary direction
        y_profit = signal_data['potential_profit_pct']  # Profit magnitude
        
        # Split data
        X_train, X_test, y_dir_train, y_dir_test, y_prof_train, y_prof_test = train_test_split(
            X, y_direction, y_profit, test_size=0.2, random_state=42
        )
        
        # Train separate models for each output
        direction_model = RandomForestClassifier(n_estimators=100, random_state=42)
        profit_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        direction_model.fit(X_train, y_dir_train)
        profit_model.fit(X_train, y_prof_train)
        
        # Evaluate
        dir_pred = direction_model.predict(X_test)
        prof_pred = profit_model.predict(X_test)
        
        dir_accuracy = direction_model.score(X_test, y_dir_test)
        prof_r2 = r2_score(y_prof_test, prof_pred)
        
        print(f"✅ Multi-output model trained")
        print(f"   Direction accuracy: {dir_accuracy:.4f}")
        print(f"   Profit R² score: {prof_r2:.4f}")
        
        # Combined evaluation: how well do we predict profitable trades?
        profitable_trades = (y_prof_test > 0.01)  # >1% profit
        correctly_predicted_profitable = (
            (dir_pred == y_dir_test) & profitable_trades
        ).sum()
        
        print(f"   Profitable trades correctly predicted: {correctly_predicted_profitable}/{profitable_trades.sum()}")
        print(f"   Profitable trade prediction rate: {correctly_predicted_profitable/profitable_trades.sum():.4f}")
        
        return {
            'method': 'multi_output_prediction',
            'direction_accuracy': dir_accuracy,
            'profit_r2': prof_r2,
            'profitable_trade_rate': correctly_predicted_profitable/profitable_trades.sum(),
            'direction_model': direction_model,
            'profit_model': profit_model
        }
    
    def method_4_profit_based_feature_engineering(self, data: pd.DataFrame) -> Dict:
        """
        Method 4: Use profit information to create new features.
        
        Create derived features based on profit patterns and relationships.
        """
        print("\n🔧 Method 4: Profit-Based Feature Engineering")
        print("=" * 60)
        
        # Create profit-based features
        enhanced_data = data.copy()
        
        # 1. Profit magnitude features
        enhanced_data['profit_abs'] = np.abs(data['potential_profit_pct'])
        enhanced_data['profit_squared'] = data['potential_profit_pct'] ** 2
        enhanced_data['profit_sign'] = np.sign(data['potential_profit_pct'])
        
        # 2. Interaction features with profit
        enhanced_data['feature_1_profit_interaction'] = data['feature_1'] * data['potential_profit_pct']
        enhanced_data['feature_2_profit_interaction'] = data['feature_2'] * data['potential_profit_pct']
        enhanced_data['feature_3_profit_interaction'] = data['feature_3'] * data['potential_profit_pct']
        
        # 3. Profit-based categorical features
        enhanced_data['profit_category'] = pd.cut(
            data['potential_profit_pct'], 
            bins=[-np.inf, -0.02, -0.01, 0, 0.01, 0.02, np.inf],
            labels=['high_loss', 'medium_loss', 'small_loss', 'small_profit', 'medium_profit', 'high_profit']
        )
        
        # 4. Risk-reward ratio features
        enhanced_data['risk_reward_ratio'] = np.abs(data['potential_profit_pct']) / (1 + np.abs(data['feature_3']))  # Volatility-adjusted
        
        # Use only signal samples
        signal_data = enhanced_data[enhanced_data['label'] != 0].copy()
        
        if len(signal_data) == 0:
            print("❌ No signals found for feature engineering")
            return {}
        
        # Prepare features (include both original and profit-based features)
        feature_cols = [
            'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
            'profit_abs', 'profit_squared', 'profit_sign',
            'feature_1_profit_interaction', 'feature_2_profit_interaction', 'feature_3_profit_interaction',
            'risk_reward_ratio'
        ]
        
        X = signal_data[feature_cols]
        y = (signal_data['label'] == 1).astype(int)
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        accuracy = model.score(X_test, y_test)
        
        print(f"✅ Model trained with profit-based features")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Features used: {len(feature_cols)}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Top Feature Importance (Profit-Enhanced):")
        for _, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return {
            'method': 'profit_feature_engineering',
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'model': model,
            'feature_count': len(feature_cols)
        }
    
    def method_5_profit_threshold_optimization(self, data: pd.DataFrame) -> Dict:
        """
        Method 5: Optimize profit thresholds based on model predictions.
        
        Use profit information to find optimal thresholds for different market conditions.
        """
        print("\n⚙️ Method 5: Profit Threshold Optimization")
        print("=" * 60)
        
        # Use only signal samples
        signal_data = data[data['label'] != 0].copy()
        
        if len(signal_data) == 0:
            print("❌ No signals found for threshold optimization")
            return {}
        
        # Train a profit prediction model
        X = signal_data[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']]
        y_profit = signal_data['potential_profit_pct']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_profit, test_size=0.2, random_state=42
        )
        
        profit_model = RandomForestRegressor(n_estimators=100, random_state=42)
        profit_model.fit(X_train, y_train)
        
        # Predict profits
        profit_predictions = profit_model.predict(X_test)
        
        # Test different profit thresholds
        thresholds = np.arange(-0.03, 0.06, 0.005)  # -3% to +6%
        results = []
        
        for threshold in thresholds:
            # Only take trades above threshold
            above_threshold = profit_predictions > threshold
            if above_threshold.sum() > 0:
                avg_profit = y_test[above_threshold].mean()
                trade_count = above_threshold.sum()
                win_rate = (y_test[above_threshold] > 0).mean()
            else:
                avg_profit = 0
                trade_count = 0
                win_rate = 0
            
            results.append({
                'threshold': threshold,
                'avg_profit': avg_profit,
                'trade_count': trade_count,
                'win_rate': win_rate,
                'total_profit': avg_profit * trade_count
            })
        
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold
        optimal_by_profit = results_df.loc[results_df['total_profit'].idxmax()]
        optimal_by_rate = results_df.loc[results_df['win_rate'].idxmax()]
        
        print(f"✅ Threshold optimization completed")
        print(f"   Optimal threshold by total profit: {optimal_by_profit['threshold']:.3f}")
        print(f"     Average profit: {optimal_by_profit['avg_profit']:.4f}")
        print(f"     Trade count: {optimal_by_profit['trade_count']}")
        print(f"     Win rate: {optimal_by_profit['win_rate']:.4f}")
        
        print(f"   Optimal threshold by win rate: {optimal_by_rate['threshold']:.3f}")
        print(f"     Win rate: {optimal_by_rate['win_rate']:.4f}")
        print(f"     Average profit: {optimal_by_rate['avg_profit']:.4f}")
        
        return {
            'method': 'profit_threshold_optimization',
            'optimal_threshold_profit': optimal_by_profit['threshold'],
            'optimal_threshold_rate': optimal_by_rate['threshold'],
            'results': results_df,
            'profit_model': profit_model
        }
    
    def run_all_methods(self, data: pd.DataFrame) -> Dict:
        """Run all profit tracking integration methods."""
        print("🚀 Running All Profit Tracking ML Integration Methods")
        print("=" * 80)
        
        results = {}
        
        # Method 1: Profit-weighted classification
        results['method_1'] = self.method_1_binary_classification_with_profit_weighting(data)
        
        # Method 2: Direct profit regression
        results['method_2'] = self.method_2_profit_regression(data)
        
        # Method 3: Multi-output prediction
        results['method_3'] = self.method_3_multi_output_prediction(data)
        
        # Method 4: Profit-based feature engineering
        results['method_4'] = self.method_4_profit_based_feature_engineering(data)
        
        # Method 5: Profit threshold optimization
        results['method_5'] = self.method_5_profit_threshold_optimization(data)
        
        # Summary
        print("\n📊 Summary of All Methods")
        print("=" * 60)
        
        method_summaries = []
        for method_name, result in results.items():
            if result:  # Check if method returned results
                if 'accuracy' in result:
                    method_summaries.append(f"{method_name}: Accuracy = {result['accuracy']:.4f}")
                elif 'r2_score' in result:
                    method_summaries.append(f"{method_name}: R² = {result['r2_score']:.4f}")
                elif 'direction_accuracy' in result:
                    method_summaries.append(f"{method_name}: Dir Acc = {result['direction_accuracy']:.4f}, Profit R² = {result['profit_r2']:.4f}")
        
        for summary in method_summaries:
            print(f"   {summary}")
        
        return results

def demonstrate_integration():
    """Demonstrate the profit tracking ML integration."""
    
    print("💰 ML Model Training with Profit Tracking Integration")
    print("=" * 80)
    
    # Create integration instance
    integrator = ProfitTrackingMLIntegration()
    
    # Generate sample data
    print("📊 Generating sample data with profit tracking...")
    data = integrator.create_sample_data_with_profits(10000)
    
    print(f"   Total samples: {len(data)}")
    print(f"   BUY signals: {(data['label'] == 1).sum()}")
    print(f"   SELL signals: {(data['label'] == -1).sum()}")
    print(f"   HOLD signals: {(data['label'] == 0).sum()}")
    print(f"   Average profit: {data['potential_profit_pct'].mean():.4f}")
    print(f"   Profit std: {data['potential_profit_pct'].std():.4f}")
    
    # Run all integration methods
    results = integrator.run_all_methods(data)
    
    print("\n✅ All integration methods completed successfully!")
    print("\n🎯 Key Takeaways:")
    print("1. Profit weighting improves model focus on high-value trades")
    print("2. Direct profit prediction enables profit magnitude forecasting")
    print("3. Multi-output models can predict both direction and profit")
    print("4. Profit-based features enhance model performance")
    print("5. Threshold optimization maximizes profit potential")
    
    return results

if __name__ == "__main__":
    demonstrate_integration()