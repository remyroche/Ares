"""
NAS Integration with Market Analysis Pipeline

This script demonstrates how to integrate the NAS system with the existing
market analysis pipeline for regime detection and HMM modeling.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from .core.nas_search import NASArchitectureSearch, NASSearchConfig
from .core.nas_model import NASModel, HMM_NAS_Model
from .core.nas_trainer import NASTrainer, TrainingConfig
from .core.nas_evaluator import NASEvaluator, EvaluationConfig
from .applications.hmm_nas import HMM_NAS_Optimizer
from .applications.regime_nas import Regime_NAS_Detector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataPreprocessor:
    """Preprocesses market data for NAS training."""

    def __init__(self, sequence_length: int = 20):
        """Initialize preprocessor.

        Args:
            sequence_length: Length of input sequences
        """
        self.sequence_length = sequence_length

    def preprocess_market_data(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess market data for NAS training.

        Args:
            market_data: Raw market data DataFrame

        Returns:
            Tuple of (features, labels)
        """
        # Extract features (simplified for demo)
        features = market_data[['close', 'volume', 'high', 'low']].values

        # Normalize features
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)

        # Create sequence data
        if len(features) > self.sequence_length:
            X = []
            y = []

            for i in range(len(features) - self.sequence_length):
                # Input sequence
                seq = features[i:i+self.sequence_length]
                X.append(seq)

                # Target (next price movement)
                current_price = features[i+self.sequence_length-1, 0]
                next_price = features[i+self.sequence_length, 0]

                # Simple regime classification based on price movement
                if next_price > current_price * 1.01:  # +1% up
                    regime = 0  # Bullish
                elif next_price < current_price * 0.99:  # -1% down
                    regime = 1  # Bearish
                elif next_price > current_price * 1.05:  # +5% up
                    regime = 2  # Strong Bullish
                elif next_price < current_price * 0.95:  # -5% down
                    regime = 3  # Strong Bearish
                else:
                    regime = 4  # Sideways

                y.append(regime)

            X = np.array(X)
            y = np.array(y)
        else:
            # Pad if too short
            pad_length = self.sequence_length - len(features) + 1
            X = np.pad(features, ((0, pad_length), (0, 0)), mode='edge')
            X = X[:self.sequence_length].reshape(1, self.sequence_length, -1)
            y = np.array([4])  # Default to sideways

        return X.astype(np.float32), y.astype(np.int64)

class NASPipelineIntegrator:
    """Integrates NAS with the market analysis pipeline."""

    def __init__(self):
        """Initialize NAS pipeline integrator."""
        self.preprocessor = MarketDataPreprocessor()
        self.nas_search = None
        self.trainer = None
        self.evaluator = None

    def setup_nas_system(self, search_strategy: str = "random", use_gpu: bool = True):
        """Setup the NAS system.

        Args:
            search_strategy: Search strategy to use
            use_gpu: Whether to use GPU
        """
        # Configure NAS search
        search_config = NASSearchConfig(
            max_iterations=50,
            search_strategy=search_strategy,
            primary_metric="accuracy",
            minimize_metric=False,
            use_gpu=use_gpu,
            batch_size=32,
            max_time_seconds=1800  # 30 minutes
        )

        # Configure training
        train_config = TrainingConfig(
            epochs=50,
            batch_size=64,
            learning_rate=0.001,
            optimizer="adam",
            loss_function="cross_entropy",
            early_stopping_patience=10,
            use_gpu=use_gpu
        )

        # Configure evaluation
        eval_config = EvaluationConfig(
            batch_size=64,
            use_gpu=use_gpu,
            compute_confusion_matrix=True,
            compute_per_class_metrics=True,
            compute_complexity_metrics=True
        )

        self.nas_search = NASArchitectureSearch(search_config)
        self.trainer = NASTrainer(train_config)
        self.evaluator = NASEvaluator(eval_config)

        logger.info("✅ NAS system configured")

    def prepare_data_for_nas(self, market_data: pd.DataFrame) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        Prepare market data for NAS training.

        Args:
            market_data: Raw market data

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("📊 Preparing market data for NAS...")

        # Preprocess data
        X, y = self.preprocessor.preprocess_market_data(market_data)

        # Split data
        n_samples = len(X)
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        n_test = n_samples - n_train - n_val

        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        logger.info(f"📈 Data prepared: Train={n_train}, Val={n_val}, Test={n_test}")
        return train_dataset, val_dataset, test_dataset

    def run_regime_detection_nas(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run NAS for market regime detection.

        Args:
            market_data: Market data for training

        Returns:
            Dictionary with results
        """
        logger.info("🎯 Running Regime Detection NAS...")

        # Prepare data
        train_dataset, val_dataset, test_dataset = self.prepare_data_for_nas(market_data)

        # Perform NAS search
        search_result = self.nas_search.search(
            train_data=(market_data.values, self.preprocessor.preprocess_market_data(market_data)[1]),
            validation_data=(market_data.values, self.preprocessor.preprocess_market_data(market_data)[1]),
            problem_type="regime_detection",
            input_shape=(32, self.preprocessor.sequence_length, market_data.shape[1])
        )

        # Train best architecture
        best_model = NASModel.create_from_config(search_result.best_architecture, "regime_detection")
        training_result = self.trainer.train(best_model, train_dataset, val_dataset, "regime_detection")

        # Evaluate
        evaluation_result = self.evaluator.evaluate_architecture(
            training_result.model, train_dataset, val_dataset, test_dataset,
            search_result.best_architecture.name, "regime_detection"
        )

        results = {
            'search_result': search_result,
            'training_result': training_result,
            'evaluation_result': evaluation_result,
            'best_architecture': search_result.best_architecture,
            'best_score': search_result.best_score,
            'regime_detection_accuracy': evaluation_result.accuracy
        }

        logger.info(f"✅ Regime detection NAS completed with accuracy: {evaluation_result.accuracy:.4f}")
        return results

    def run_hmm_nas(self, market_data: pd.DataFrame, n_states: int = 5) -> Dict[str, Any]:
        """
        Run NAS for HMM state modeling.

        Args:
            market_data: Market data for training
            n_states: Number of HMM states

        Returns:
            Dictionary with results
        """
        logger.info(f"🔍 Running HMM NAS with {n_states} states...")

        # Prepare data
        train_dataset, val_dataset, test_dataset = self.prepare_data_for_nas(market_data)

        # Perform HMM NAS search
        search_result = self.nas_search.search(
            train_data=(market_data.values, self.preprocessor.preprocess_market_data(market_data)[1]),
            validation_data=(market_data.values, self.preprocessor.preprocess_market_data(market_data)[1]),
            problem_type="hmm",
            input_shape=(32, self.preprocessor.sequence_length, market_data.shape[1])
        )

        # Train best HMM architecture
        best_model = HMM_NAS_Model(search_result.best_architecture, n_states)
        training_result = self.trainer.train(best_model, train_dataset, val_dataset, "hmm")

        # Evaluate
        evaluation_result = self.evaluator.evaluate_architecture(
            training_result.model, train_dataset, val_dataset, test_dataset,
            search_result.best_architecture.name, "hmm"
        )

        results = {
            'search_result': search_result,
            'training_result': training_result,
            'evaluation_result': evaluation_result,
            'best_architecture': search_result.best_architecture,
            'best_score': search_result.best_score,
            'hmm_state_accuracy': evaluation_result.accuracy,
            'n_states': n_states
        }

        logger.info(f"✅ HMM NAS completed with accuracy: {evaluation_result.accuracy:.4f}")
        return results

    def compare_search_strategies(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare different NAS search strategies.

        Args:
            market_data: Market data for comparison

        Returns:
            Dictionary with comparison results
        """
        logger.info("🔬 Comparing NAS search strategies...")

        strategies = ['random', 'bayesian']
        results = {}

        for strategy in strategies:
            logger.info(f"🧪 Testing {strategy} search strategy...")

            # Configure for this strategy
            self.setup_nas_system(search_strategy=strategy)

            # Run regime detection
            strategy_result = self.run_regime_detection_nas(market_data)

            results[strategy] = {
                'best_score': strategy_result['best_score'],
                'accuracy': strategy_result['regime_detection_accuracy'],
                'search_time': strategy_result['search_result'].execution_time,
                'n_evaluations': strategy_result['search_result'].n_evaluations
            }

        # Determine best strategy
        best_strategy = max(results.keys(), key=lambda k: results[k]['best_score'])
        best_score = results[best_strategy]['best_score']

        comparison = {
            'strategy_results': results,
            'best_strategy': best_strategy,
            'best_score': best_score,
            'recommendation': f"Use {best_strategy} search for best performance"
        }

        logger.info(f"🏆 Best strategy: {best_strategy} (score: {best_score:.4f})")
        return comparison

    def save_nas_model(self, model: nn.Module, architecture: Any, save_path: str):
        """Save trained NAS model.

        Args:
            model: Trained model
            architecture: Architecture configuration
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        model_path = save_path / "nas_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': architecture.to_dict(),
            'model_type': model.__class__.__name__
        }, model_path)

        # Save architecture info
        arch_path = save_path / "architecture.json"
        with open(arch_path, 'w') as f:
            import json
            json.dump(architecture.to_dict(), f, indent=2)

        logger.info(f"💾 NAS model saved to {model_path}")

    def load_nas_model(self, model_path: str) -> Tuple[nn.Module, Any]:
        """Load trained NAS model.

        Args:
            model_path: Path to saved model

        Returns:
            Tuple of (model, architecture)
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path / "nas_model.pt", map_location='cpu')

        # Recreate architecture
        architecture = ArchitectureConfig.from_dict(checkpoint['architecture'])

        # Recreate model
        if checkpoint['model_type'] == 'HMM_NAS_Model':
            model = HMM_NAS_Model(architecture)
        else:
            model = NASModel.create_from_config(architecture, architecture.problem_type)

        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"📁 NAS model loaded from {model_path}")
        return model, architecture

def main():
    """Main function demonstrating NAS integration."""
    logger.info("🚀 NAS Integration with Market Analysis Pipeline")
    logger.info("=" * 60)

    # Create sample market data (in real usage, this would come from your data pipeline)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)

    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(1000) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(1000) * 0.1) + np.abs(np.random.randn(1000) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(1000) * 0.1) - np.abs(np.random.randn(1000) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.1),
        'volume': np.random.exponential(1000, 1000)
    })

    market_data.set_index('timestamp', inplace=True)

    try:
        # Initialize NAS integrator
        integrator = NASPipelineIntegrator()
        integrator.setup_nas_system()

        # Run regime detection NAS
        logger.info("\\n1. Running Regime Detection NAS...")
        regime_results = integrator.run_regime_detection_nas(market_data)

        # Run HMM NAS
        logger.info("\\n2. Running HMM NAS...")
        hmm_results = integrator.run_hmm_nas(market_data, n_states=5)

        # Compare search strategies
        logger.info("\\n3. Comparing Search Strategies...")
        comparison_results = integrator.compare_search_strategies(market_data)

        # Save best model
        logger.info("\\n4. Saving Best Model...")
        best_model = regime_results['training_result'].model
        best_architecture = regime_results['best_architecture']
        integrator.save_nas_model(best_model, best_architecture, "nas_models/regime_detector")

        # Display results
        logger.info("\\n✅ NAS Integration Results:")
        logger.info("=" * 40)
        logger.info(f"🏆 Regime Detection Accuracy: {regime_results['regime_detection_accuracy']:.4f}")
        logger.info(f"🔍 HMM State Accuracy: {hmm_results['hmm_state_accuracy']:.4f}")
        logger.info(f"🎯 Best Search Strategy: {comparison_results['best_strategy']}")
        logger.info(f"📊 Best Strategy Score: {comparison_results['best_score']:.4f}")

        # Integration summary
        integration_summary = {
            'regime_detection': {
                'accuracy': regime_results['regime_detection_accuracy'],
                'architecture': regime_results['best_architecture'].name,
                'parameters': regime_results['evaluation_result'].num_parameters
            },
            'hmm_modeling': {
                'accuracy': hmm_results['hmm_state_accuracy'],
                'n_states': hmm_results['n_states'],
                'architecture': hmm_results['best_architecture'].name,
                'parameters': hmm_results['evaluation_result'].num_parameters
            },
            'search_comparison': comparison_results,
            'recommendations': {
                'best_strategy': comparison_results['best_strategy'],
                'model_saved': True,
                'ready_for_production': regime_results['regime_detection_accuracy'] > 0.7
            }
        }

        logger.info("\\n📋 Integration Summary:")
        logger.info(f"   Model Quality: {'✅ Production Ready' if integration_summary['recommendations']['ready_for_production'] else '⚠️ Needs Improvement'}")
        logger.info(f"   Best Strategy: {integration_summary['recommendations']['best_strategy']}")
        logger.info(f"   Models Saved: {integration_summary['recommendations']['model_saved']}")

        return integration_summary

    except Exception as e:
        logger.error(f"❌ NAS integration failed: {e}")
        raise

if __name__ == "__main__":
    main()