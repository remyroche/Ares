from ..standardized_parquet_handler import standardized_parquet_handler
"""Step 10 Configuration Management.

This module provides comprehensive configuration management for the unified
regime intelligence system, including validation and defaults.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from src.utils.logger import system_logger
import logging

logger = system_logger.getChild('Step10Config')

# Default configuration
DEFAULT_CONFIG = {
    # Model configuration
    "timeframes": ["5m", "15m", "30m"],
    "hmm_states_per_tf": 5,
    "sequence_length": 20,
    "d_model": 256,
    "nhead": 8,
    "num_layers": 4,
    "dropout": 0.1,

    # Data configuration
    "data_dir": "data",
    "symbol": "ETHUSDT",
    "exchange": "BINANCE",

    # Training configuration
    "learning_rate": 0.0001,
    "batch_size": 32,
    "epochs": 100,
    "validation_split": 0.2,

    # Enhancement configuration
    "enhancement": {
        "hpo_enabled": False,
        "architecture_optimization_enabled": False,
        "hpo": {
            "n_trials": 50,
            "timeout": 3600,
            "pruning": True,
        }
    },

    # SR integration
    "sr_breakout_predictor": {
        "use_optimized_params": True,
    },

    # Artifacts
    "artifacts_dir": "checkpoints/unified_regime_intelligence",
}


@dataclass
class Step10Config:
    """Configuration class for Step 10 Unified Regime Intelligence.

    This class manages all configuration parameters for the unified regime
    intelligence system with validation and type safety.
    """

    # Model configuration
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "30m", "1h"])
    hmm_states_per_tf: int = 5
    sequence_length: int = 20
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dropout: float = 0.1

    # Data configuration
    data_dir: str = "data"
    symbol: str = "ETHUSDT"
    exchange: str = "BINANCE"

    # Training configuration
    learning_rate: float = 0.0001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2

    # Enhancement configuration
    enhancement: Dict[str, Any] = field(default_factory=dict)
    hpo_enabled: bool = False
    architecture_optimization_enabled: bool = False
    hpo_config: Dict[str, Any] = field(default_factory=dict)

    # SR integration
    sr_config: Dict[str, Any] = field(default_factory=dict)

    # Artifacts
    artifacts_dir: str = "checkpoints/unified_regime_intelligence"

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'Step10Config':
        """Create configuration from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Step10Config instance
        """
        # Merge with defaults
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)

        # Extract nested configurations
        enhancement = merged_config.get("enhancement", {})
        sr_config = merged_config.get("sr_breakout_predictor", {})

        return cls(
            timeframes=merged_config.get("timeframes", ["1m", "5m", "15m", "30m", "1h"]),
            hmm_states_per_tf=merged_config.get("hmm_states_per_tf", 5),
            sequence_length=merged_config.get("sequence_length", 20),
            d_model=merged_config.get("d_model", 256),
            nhead=merged_config.get("nhead", 8),
            num_layers=merged_config.get("num_layers", 4),
            dropout=merged_config.get("dropout", 0.1),
            data_dir=merged_config.get("data_dir", "data"),
            symbol=merged_config.get("symbol", "ETHUSDT"),
            exchange=merged_config.get("exchange", "BINANCE"),
            learning_rate=merged_config.get("learning_rate", 0.0001),
            batch_size=merged_config.get("batch_size", 32),
            epochs=merged_config.get("epochs", 100),
            validation_split=merged_config.get("validation_split", 0.2),
            enhancement=enhancement,
            hpo_enabled=enhancement.get("hpo_enabled", False),
            architecture_optimization_enabled=enhancement.get("architecture_optimization_enabled", False),
            hpo_config=enhancement.get("hpo", {}),
            sr_config=sr_config,
            artifacts_dir=merged_config.get("artifacts_dir", "checkpoints/unified_regime_intelligence"),
        )

    def validate(self) -> List[str]:
        """Validate configuration parameters.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate model parameters
        if self.d_model <= 0:
            errors.append("d_model must be positive")
        if self.nhead <= 0:
            errors.append("nhead must be positive")
        if self.num_layers <= 0:
            errors.append("num_layers must be positive")
        if not (0 <= self.dropout <= 1):
            errors.append("dropout must be between 0 and 1")
        if self.d_model % self.nhead != 0:
            errors.append("d_model must be divisible by nhead")

        # Validate training parameters
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.epochs <= 0:
            errors.append("epochs must be positive")
        if not (0 < self.validation_split < 1):
            errors.append("validation_split must be between 0 and 1")

        # Validate sequence parameters
        if self.sequence_length <= 0:
            errors.append("sequence_length must be positive")
        if self.hmm_states_per_tf <= 0:
            errors.append("hmm_states_per_tf must be positive")

        # Validate timeframes
        if not self.timeframes:
            errors.append("timeframes cannot be empty")

        return errors

    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration.

        Returns:
            Dictionary with model configuration
        """
        return {
            "timeframes": self.timeframes,
            "hmm_states_per_tf": self.hmm_states_per_tf,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration.

        Returns:
            Dictionary with training configuration
        """
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validation_split": self.validation_split,
        }

    def get_data_config(self) -> Dict[str, Any]:
        """Get data-specific configuration.

        Returns:
            Dictionary with data configuration
        """
        return {
            "data_dir": self.data_dir,
            "symbol": self.symbol,
            "exchange": self.exchange,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "timeframes": self.timeframes,
            "hmm_states_per_tf": self.hmm_states_per_tf,
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "data_dir": self.data_dir,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validation_split": self.validation_split,
            "enhancement": self.enhancement,
            "sr_breakout_predictor": self.sr_config,
            "artifacts_dir": self.artifacts_dir,
        }


def create_step10_config(config: Optional[Dict[str, Any]] = None) -> Step10Config:
    """Create Step 10 configuration from dictionary.

    Args:
        config: Configuration dictionary (optional)

    Returns:
        Step10Config instance
    """
    if config is None:
        config = {}

    step10_config = Step10Config.from_dict(config)

    # Validate configuration
    errors = step10_config.validate()
    if errors:
        logger.warning("⚠️ Configuration validation warnings:")
        for error in errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("✅ Configuration validation passed")

    return step10_config
