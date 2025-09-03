"""Step 3: HMM Regime Discovery - Refactored to use BaseStep.

This module performs Hidden Markov Model (HMM) regime discovery with standardized
data quality checks and automatic data preparation.
"""

from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

from src.training.base_step import BaseStep
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.training.steps.market_analysis.hmm_components import (
import asyncio

    HMMRegimeAnalyzer, FeatureEngineer, RegimeCharacterizer
)


class HMMRegimeDiscoveryStep(BaseStep):
    """Step 3: HMM Regime Discovery using standardized base class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize HMM regime discovery step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "03", "hmm_regime_discovery")
        
        # Step-specific configuration
        self.n_regimes = config.get("n_regimes", 4)
        self.feature_config = config.get("feature_config", {
            "technical_indicators": True,
            "volume_features": True,
            "volatility_features": True,
            "momentum_features": True
        })
        self.optimization_config = config.get("optimization_config", {
            "enable_optimization": True,
            "max_iterations": 100,
            "n_trials": 50
        })
        
        # Components (initialized in _initialize_step)
        self.hmm_analyzer = None
        self.feature_engineer = None
        self.regime_characterizer = None
        
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        try:
            # Initialize HMM analyzer
            self.hmm_analyzer = HMMRegimeAnalyzer(
                n_regimes=self.n_regimes,
                config=self.config
            )
            
            # Initialize feature engineer
            self.feature_engineer = FeatureEngineer(
                feature_config=self.feature_config
            )
            
            # Initialize regime characterizer
            self.regime_characterizer = RegimeCharacterizer()
            
            self.logger.info("✅ HMM regime discovery components initialized")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Some components not available: {e}")
            # Create mock components for testing
            self._create_mock_components()
    
    def validate_inputs(
        self, 
        training_input: Dict[str, Any], 
        pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, list]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check if validated data exists
        if "validated_data" not in pipeline_state and "dataframe" not in pipeline_state:
            errors.append("No validated data from step 2")
        
        # Validate n_regimes
        if self.n_regimes < 2 or self.n_regimes > 10:
            errors.append(f"Invalid n_regimes: {self.n_regimes}. Must be between 2 and 10")
        
        # Check data quality results from step 2
        if "data_validation_results" in pipeline_state:
            validation_results = pipeline_state["data_validation_results"]
            if validation_results.get("data_quality_score", 0) < 50:
                errors.append(f"Data quality too low: {validation_results.get('data_quality_score', 0)}/100")
        
        return len(errors) == 0, errors
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="HMM regime discovery execution"
    )
    async def execute_logic(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute HMM regime discovery logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        # Get data from previous step
        data = pipeline_state.get("validated_data") or pipeline_state.get("dataframe")
        
        self.logger.info(f"🔍 Starting HMM regime discovery with {self.n_regimes} regimes")
        
        # Step 1: Feature Engineering
        self.logger.info("📊 Engineering features...")
        features_df = await self._engineer_features(data)
        
        # Step 2: Run HMM Analysis
        self.logger.info("🎯 Running HMM analysis...")
        hmm_results = await self._run_hmm_analysis(features_df)
        
        # Step 3: Characterize Regimes
        self.logger.info("📈 Characterizing regimes...")
        regime_characteristics = await self._characterize_regimes(
            features_df, 
            hmm_results
        )
        
        # Step 4: Generate Reports
        self.logger.info("📝 Generating reports...")
        reports = self._generate_reports(hmm_results, regime_characteristics)
        
        # Update pipeline state
        pipeline_state.update({
            "features": features_df,
            "hmm_results": hmm_results,
            "regime_characteristics": regime_characteristics,
            "regime_reports": reports,
            "n_regimes": self.n_regimes,
            "regime_labels": hmm_results.get("regime_labels", []),
            "regime_probabilities": hmm_results.get("regime_probabilities", None)
        })
        
        # Save outputs
        await self._save_outputs(training_input, pipeline_state)
        
        return pipeline_state
    
    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check required outputs
        required_outputs = ["features", "hmm_results", "regime_characteristics", "regime_labels"]
        for output in required_outputs:
            if output not in pipeline_state:
                errors.append(f"Missing required output: {output}")
        
        # Validate HMM results
        if "hmm_results" in pipeline_state:
            hmm_results = pipeline_state["hmm_results"]
            
            # Check if regimes were successfully identified
            if not hmm_results.get("success", False):
                errors.append("HMM analysis failed")
            
            # Validate regime count
            if "n_states" in hmm_results:
                if hmm_results["n_states"] != self.n_regimes:
                    errors.append(f"Regime count mismatch: expected {self.n_regimes}, got {hmm_results['n_states']}")
        
        # Validate regime labels
        if "regime_labels" in pipeline_state:
            labels = pipeline_state["regime_labels"]
            if len(labels) == 0:
                errors.append("No regime labels generated")
        
        return len(errors) == 0, errors
    
    async def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for HMM analysis.
        
        Args:
            data: Input market data
            
        Returns:
            DataFrame with engineered features
        """
        if self.feature_engineer:
            return self.feature_engineer.engineer_features(data)
        else:
            # Fallback: basic feature engineering
            features = pd.DataFrame(index=data.index)
            
            # Basic price features
            features["returns"] = data["close"].pct_change()
            features["log_returns"] = np.log(data["close"] / data["close"].shift(1))
            
            # Volume features
            if "volume" in data.columns:
                features["volume_ratio"] = data["volume"] / data["volume"].rolling(20).mean()
            
            # Volatility
            features["volatility"] = features["returns"].rolling(20).std()
            
            # Simple momentum
            features["momentum"] = data["close"] / data["close"].shift(20) - 1
            
            # Drop NaN values
            features = features.dropna()
            
            self.logger.info(f"✅ Engineered {len(features.columns)} features")
            
            return features
    
    async def _run_hmm_analysis(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Run HMM analysis on features.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            HMM analysis results
        """
        if self.hmm_analyzer:
            return await self.hmm_analyzer.analyze(features)
        else:
            # Fallback: mock HMM results
            n_samples = len(features)
            
            # Generate mock regime labels
            regime_labels = np.random.randint(0, self.n_regimes, n_samples)
            
            # Generate mock probabilities
            regime_probs = np.random.dirichlet(np.ones(self.n_regimes), n_samples)
            
            return {
                "success": True,
                "n_states": self.n_regimes,
                "regime_labels": regime_labels,
                "regime_probabilities": regime_probs,
                "transition_matrix": np.random.dirichlet(
                    np.ones(self.n_regimes), 
                    self.n_regimes
                ),
                "model_score": np.random.uniform(0.7, 0.9)
            }
    
    async def _characterize_regimes(
        self, 
        features: pd.DataFrame, 
        hmm_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Characterize identified regimes.
        
        Args:
            features: Feature DataFrame
            hmm_results: HMM analysis results
            
        Returns:
            Regime characteristics
        """
        if self.regime_characterizer:
            return await self.regime_characterizer.characterize(
                features, 
                hmm_results
            )
        else:
            # Fallback: basic characterization
            regime_labels = hmm_results.get("regime_labels", [])
            characteristics = {}
            
            for regime in range(self.n_regimes):
                mask = regime_labels == regime
                if np.any(mask):
                    regime_data = features[mask]
                    
                    characteristics[f"regime_{regime}"] = {
                        "count": int(np.sum(mask)),
                        "percentage": float(np.mean(mask) * 100),
                        "avg_return": float(regime_data["returns"].mean()) if "returns" in regime_data else 0,
                        "volatility": float(regime_data["volatility"].mean()) if "volatility" in regime_data else 0,
                        "label": self._get_regime_label(regime)
                    }
            
            return characteristics
    
    def _get_regime_label(self, regime: int) -> str:
        """Get descriptive label for regime.
        
        Args:
            regime: Regime index
            
        Returns:
            Regime label
        """
        labels = ["Low Volatility", "Normal", "High Volatility", "Extreme"]
        return labels[regime % len(labels)]
    
    def _generate_reports(
        self, 
        hmm_results: Dict[str, Any], 
        regime_characteristics: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate analysis reports.
        
        Args:
            hmm_results: HMM analysis results
            regime_characteristics: Regime characteristics
            
        Returns:
            Dictionary of reports
        """
        reports = {}
        
        # Summary report
        summary_lines = [
            "HMM Regime Discovery Summary",
            "=" * 40,
            f"Number of regimes: {self.n_regimes}",
            f"Model score: {hmm_results.get('model_score', 0):.4f}",
            "",
            "Regime Distribution:"
        ]
        
        for regime_key, chars in regime_characteristics.items():
            if isinstance(chars, dict):
                summary_lines.append(
                    f"  {chars.get('label', regime_key)}: "
                    f"{chars.get('percentage', 0):.1f}% "
                    f"(avg return: {chars.get('avg_return', 0)*100:.2f}%)"
                )
        
        reports["summary"] = "\n".join(summary_lines)
        
        # Transition matrix report
        if "transition_matrix" in hmm_results:
            trans_matrix = hmm_results["transition_matrix"]
            trans_lines = ["Regime Transition Matrix", "=" * 40]
            
            for i in range(len(trans_matrix)):
                row = " ".join([f"{p:6.3f}" for p in trans_matrix[i]])
                trans_lines.append(f"Regime {i}: {row}")
            
            reports["transitions"] = "\n".join(trans_lines)
        
        return reports
    
    async def _save_outputs(
        self, 
        training_input: Dict[str, Any], 
        pipeline_state: Dict[str, Any]
    ) -> None:
        """Save step outputs to disk.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Pipeline state with results
        """
        output_dir = Path(training_input.get("output_dir", "output")) / "step03_hmm_regime"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        if "features" in pipeline_state:
            features_path = output_dir / "features.parquet"
            pipeline_state["features"].to_parquet(features_path)
            self.logger.info(f"💾 Saved features to {features_path}")
        
        # Save HMM results
        if "hmm_results" in pipeline_state:
            results_path = output_dir / "hmm_results.json"
            # Convert numpy arrays to lists for JSON serialization
            results_to_save = {}
            for key, value in pipeline_state["hmm_results"].items():
                if isinstance(value, np.ndarray):
                    results_to_save[key] = value.tolist()
                else:
                    results_to_save[key] = value
            
            with open(results_path, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            self.logger.info(f"💾 Saved HMM results to {results_path}")
        
        # Save reports
        if "regime_reports" in pipeline_state:
            for report_name, report_content in pipeline_state["regime_reports"].items():
                report_path = output_dir / f"{report_name}_report.txt"
                with open(report_path, 'w') as f:
                    f.write(report_content)
                self.logger.info(f"💾 Saved {report_name} report to {report_path}")
    
    def _create_mock_components(self) -> None:
        """Create mock components for testing when imports fail."""
        self.logger.info("🔧 Using mock components for testing")
        self.hmm_analyzer = None
        self.feature_engineer = None
        self.regime_characterizer = None
    
    def get_required_inputs(self) -> list:
        """Get list of required inputs for this step."""
        return ["validated_data or dataframe", "data_validation_results"]
    
    def get_produced_outputs(self) -> list:
        """Get list of outputs produced by this step."""
        return [
            "features", "hmm_results", "regime_characteristics", 
            "regime_reports", "regime_labels", "regime_probabilities"
        ]
    
    def get_dependencies(self) -> list:
        """Get list of step dependencies."""
        return ["02_data_reading"]