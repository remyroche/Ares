"""
Layer 2.5 Chaser Integration for Layer 3

Automatically feeds top 2-3 Layer 2.5 Chaser model predictions into Layer 3.
Stores predictions and models using the artifact manager for persistence.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle
import json
from datetime import datetime

# Import artifact manager
try:
    from src.utils.artifact_manager import ArtifactManager
    ARTIFACT_MANAGER_AVAILABLE = True
except ImportError:
    ARTIFACT_MANAGER_AVAILABLE = False
    print("⚠️ Artifact Manager not available, using local storage")

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class Layer25Integration:
    """Manages integration of Layer 2.5 Chaser models into Layer 3."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str, 
                 outcomes_dir: Optional[Path] = None):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.outcomes_dir = outcomes_dir or Path('outcomes')
        self.outcomes_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize artifact manager
        if ARTIFACT_MANAGER_AVAILABLE:
            self.artifact_manager = ArtifactManager()
        else:
            self.artifact_manager = None
            
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def extract_top_chaser_models(self, chaser_results: Dict[str, Any], 
                                 top_n: int = 3) -> Dict[str, Any]:
        """
        Extract top N chaser models based on performance.
        
        Args:
            chaser_results: Results from Layer 2.5 Chaser training
            top_n: Number of top models to extract
            
        Returns:
            Dictionary with top models and their metadata
        """
        tprint_info(f"🏆 Extracting top {top_n} Layer 2.5 Chaser models...")
        
        top_models = {}
        model_scores = []
        
        # Collect all model performances
        for geometry_uuid, geometry_data in chaser_results.items():
            if 'meta_features' not in geometry_data:
                continue
                
            meta_data = geometry_data['meta_features']
            
            # Extract model performances
            for model_name, model_data in meta_data.items():
                if model_name.startswith('chaser_') and 'performance' in model_data:
                    performance = model_data['performance']
                    
                    score = performance.get('auc', 0.5) if 'auc' in performance else performance.get('ic', 0.0)
                    
                    model_scores.append({
                        'geometry_uuid': geometry_uuid,
                        'model_name': model_name,
                        'score': score,
                        'performance': performance,
                        'model_data': model_data,
                        'geometry_data': geometry_data
                    })
        
        # Sort by score (descending) and take top N
        model_scores.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = model_scores[:top_n]
        
        tprint_info(f"📊 Found {len(model_scores)} total models, selecting top {len(top_candidates)}")
        
        # Build top models dictionary
        for i, candidate in enumerate(top_candidates):
            model_key = f"top_{i+1}_{candidate['model_name']}"
            
            top_models[model_key] = {
                'geometry_uuid': candidate['geometry_uuid'],
                'model_name': candidate['model_name'],
                'score': candidate['score'],
                'performance': candidate['performance'],
                'model_artifact': candidate['model_data'].get('model'),
                'predictions_oof': candidate['model_data'].get('predictions_oof'),
                'features': candidate['model_data'].get('features'),
                'rank': i + 1
            }
            
            tprint_info(f"   {i+1}. {candidate['model_name']} (Score: {candidate['score']:.4f})")
        
        return top_models
    
    def save_chaser_artifacts(self, top_models: Dict[str, Any]) -> Dict[str, str]:
        """
        Save chaser models and predictions using artifact manager.
        
        Args:
            top_models: Dictionary of top chaser models
            
        Returns:
            Dictionary with artifact paths
        """
        tprint_info("💾 Saving Layer 2.5 Chaser artifacts...")
        
        artifact_paths = {}
        
        for model_key, model_info in top_models.items():
            try:
                # Save model artifact
                if model_info['model_artifact'] is not None:
                    model_path = self._save_model_artifact(model_key, model_info)
                    artifact_paths[f"{model_key}_model"] = model_path
                
                # Save OOF predictions
                if model_info['predictions_oof'] is not None:
                    pred_path = self._save_predictions_artifact(model_key, model_info)
                    artifact_paths[f"{model_key}_predictions"] = pred_path
                
                # Save metadata
                metadata_path = self._save_metadata_artifact(model_key, model_info)
                artifact_paths[f"{model_key}_metadata"] = metadata_path
                
            except Exception as e:
                tprint_warning(f"⚠️ Failed to save artifacts for {model_key}: {e}")
        
        tprint_success(f"✅ Saved {len(artifact_paths)} chaser artifacts")
        return artifact_paths
    
    def _save_model_artifact(self, model_key: str, model_info: Dict[str, Any]) -> str:
        """Save model artifact using artifact manager or local storage."""
        model = model_info['model_artifact']
        
        if self.artifact_manager:
            # Use artifact manager
            artifact_id = f"layer25_chaser_{self.symbol}_{self.exchange}_{self.timeframe}_{model_key}_{self.ts}"
            path = self.artifact_manager.save_artifact(model, artifact_id, "model")
        else:
            # Local storage fallback
            filename = f"layer25_chaser_{model_key}_{self.ts}.pkl"
            path = self.outcomes_dir / "layer25_artifacts" / filename
            path.parent.mkdir(exist_ok=True)
            
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        
        return str(path)
    
    def _save_predictions_artifact(self, model_key: str, model_info: Dict[str, Any]) -> str:
        """Save OOF predictions artifact."""
        predictions = model_info['predictions_oof']
        
        if self.artifact_manager:
            # Use artifact manager
            artifact_id = f"layer25_chaser_{self.symbol}_{self.exchange}_{self.timeframe}_{model_key}_predictions_{self.ts}"
            path = self.artifact_manager.save_artifact(predictions, artifact_id, "predictions")
        else:
            # Local storage fallback
            filename = f"layer25_chaser_{model_key}_predictions_{self.ts}.parquet"
            path = self.outcomes_dir / "layer25_artifacts" / filename
            path.parent.mkdir(exist_ok=True)
            
            if isinstance(predictions, pd.Series):
                predictions.to_frame().to_parquet(path)
            elif isinstance(predictions, np.ndarray):
                pd.Series(predictions).to_frame().to_parquet(path)
            else:
                pd.DataFrame(predictions).to_parquet(path)
        
        return str(path)
    
    def _save_metadata_artifact(self, model_key: str, model_info: Dict[str, Any]) -> str:
        """Save metadata artifact."""
        metadata = {
            'model_key': model_key,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframe': self.timeframe,
            'geometry_uuid': model_info['geometry_uuid'],
            'model_name': model_info['model_name'],
            'score': model_info['score'],
            'performance': model_info['performance'],
            'rank': model_info['rank'],
            'timestamp': self.ts,
            'features': model_info.get('features', [])
        }
        
        if self.artifact_manager:
            # Use artifact manager
            artifact_id = f"layer25_chaser_{self.symbol}_{self.exchange}_{self.timeframe}_{model_key}_metadata_{self.ts}"
            path = self.artifact_manager.save_artifact(metadata, artifact_id, "metadata")
        else:
            # Local storage fallback
            filename = f"layer25_chaser_{model_key}_metadata_{self.ts}.json"
            path = self.outcomes_dir / "layer25_artifacts" / filename
            path.parent.mkdir(exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return str(path)
    
    def load_chaser_predictions(self, artifact_paths: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Load chaser OOF predictions from artifacts.
        
        Args:
            artifact_paths: Dictionary of artifact paths
            
        Returns:
            Dictionary with loaded predictions
        """
        tprint_info("📂 Loading Layer 2.5 Chaser predictions...")
        
        predictions = {}
        
        for key, path in artifact_paths.items():
            if not key.endswith('_predictions'):
                continue
                
            model_key = key.replace('_predictions', '')
            
            try:
                if self.artifact_manager:
                    # Load from artifact manager
                    pred_data = self.artifact_manager.load_artifact(path)
                else:
                    # Load from local storage
                    if path.endswith('.parquet'):
                        pred_data = pd.read_parquet(path)
                        if pred_data.shape[1] == 1:
                            pred_data = pred_data.iloc[:, 0]
                        else:
                            pred_data = pred_data.values.flatten()
                    else:
                        with open(path, 'rb') as f:
                            pred_data = pickle.load(f)
                
                predictions[model_key] = np.asarray(pred_data, dtype=np.float32)
                tprint_info(f"   ✅ Loaded {model_key}: {len(predictions[model_key])} predictions")
                
            except Exception as e:
                tprint_warning(f"⚠️ Failed to load predictions for {model_key}: {e}")
        
        return predictions
    
    def integrate_chaser_features_into_layer3(self, df: pd.DataFrame, 
                                            chaser_predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Integrate chaser predictions as features into Layer 3 DataFrame.
        
        Args:
            df: Layer 3 DataFrame
            chaser_predictions: Dictionary of chaser OOF predictions
            
        Returns:
            Enhanced DataFrame with chaser features
        """
        tprint_info("🔗 Integrating Layer 2.5 Chaser features into Layer 3...")
        
        df_enhanced = df.copy()
        added_features = []
        
        for model_key, predictions in chaser_predictions.items():
            try:
                # Align predictions with DataFrame index
                if len(predictions) == len(df):
                    # Direct alignment
                    feature_name = f"chaser_{model_key}"
                    df_enhanced[feature_name] = predictions
                    added_features.append(feature_name)
                else:
                    # Need to align by index
                    if hasattr(predictions, 'index'):
                        # Predictions have index (pandas)
                        aligned_pred = predictions.reindex(df.index, method='ffill').fillna(0)
                    else:
                        # Predictions are numpy array - use positional alignment
                        min_len = min(len(predictions), len(df))
                        aligned_pred = np.full(len(df), 0.0, dtype=np.float32)
                        aligned_pred[:min_len] = predictions[:min_len]
                    
                    feature_name = f"chaser_{model_key}"
                    df_enhanced[feature_name] = aligned_pred
                    added_features.append(feature_name)
                
                tprint_info(f"   ✅ Added {feature_name}")
                
            except Exception as e:
                tprint_warning(f"⚠️ Failed to integrate {model_key}: {e}")
        
        tprint_success(f"🎉 Integrated {len(added_features)} chaser features into Layer 3")
        return df_enhanced
    
    def generate_integration_report(self, top_models: Dict[str, Any], 
                                  artifact_paths: Dict[str, str]) -> str:
        """
        Generate integration report.
        
        Args:
            top_models: Dictionary of top models
            artifact_paths: Dictionary of artifact paths
            
        Returns:
            Path to generated report
        """
        tprint_info("📊 Generating Layer 2.5 integration report...")
        
        lines = ["# Layer 2.5 Chaser Integration Report\n\n"]
        lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"Symbol: {self.symbol}\n")
        lines.append(f"Exchange: {self.exchange}\n")
        lines.append(f"Timeframe: {self.timeframe}\n")
        lines.append(f"Integration ID: {self.ts}\n\n")
        
        # Summary
        lines.append("## Integration Summary\n")
        lines.append(f"- **Top Models Selected**: {len(top_models)}\n")
        lines.append(f"- **Artifacts Saved**: {len(artifact_paths)}\n")
        lines.append("\n")
        
        # Model details
        lines.append("## Top Chaser Models\n")
        lines.append("| Rank | Model Name | Score | Geometry UUID | Artifacts |\n")
        lines.append("|------|------------|-------|---------------|-----------|\n")
        
        for model_key, model_info in top_models.items():
            rank = model_info['rank']
            name = model_info['model_name']
            score = model_info['score']
            geometry = model_info['geometry_uuid'][:8] + "..."  # Shortened UUID
            
            # Count artifacts for this model
            model_artifacts = [k for k in artifact_paths.keys() if model_key in k]
            artifacts_count = len(model_artifacts)
            
            lines.append(f"| {rank} | {name} | {score:.4f} | {geometry} | {artifacts_count} |\n")
        
        lines.append("\n")
        
        # Artifact paths
        lines.append("## Artifact Paths\n")
        for artifact_type, path in artifact_paths.items():
            lines.append(f"**{artifact_type}**: `{path}`\n")
        
        lines.append("\n")
        
        # Save report
        report_path = self.outcomes_dir / f"layer25_integration_report_{self.ts}.md"
        report_path.write_text("".join(lines))
        
        tprint_success(f"✅ Integration report saved: {report_path}")
        return str(report_path)

def integrate_layer25_into_layer3(
    df: pd.DataFrame,
    chaser_results: Dict[str, Any],
    symbol: str = "ETHUSDT",
    exchange: str = "binance", 
    timeframe: str = "15m",
    top_n_models: int = 3,
    outcomes_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main integration function for Layer 2.5 → Layer 3.
    
    Args:
        df: Layer 3 DataFrame
        chaser_results: Results from Layer 2.5 Chaser training
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        top_n_models: Number of top models to integrate
        outcomes_dir: Outcomes directory
        
    Returns:
        Tuple of (enhanced DataFrame, integration metadata)
    """
    integrator = Layer25Integration(symbol, exchange, timeframe, outcomes_dir)
    
    # Extract top models
    top_models = integrator.extract_top_chaser_models(chaser_results, top_n_models)
    
    if not top_models:
        tprint_warning("⚠️ No valid Layer 2.5 models found for integration")
        return df, {'status': 'no_models_found'}
    
    # Save artifacts
    artifact_paths = integrator.save_chaser_artifacts(top_models)
    
    # Load predictions
    chaser_predictions = integrator.load_chaser_predictions(artifact_paths)
    
    # Integrate into Layer 3
    df_enhanced = integrator.integrate_chaser_features_into_layer3(df, chaser_predictions)
    
    # Generate report
    report_path = integrator.generate_integration_report(top_models, artifact_paths)
    
    integration_metadata = {
        'status': 'success',
        'top_models_count': len(top_models),
        'features_added': len(chaser_predictions),
        'artifact_paths': artifact_paths,
        'report_path': report_path,
        'timestamp': integrator.ts
    }
    
    tprint_success(f"🎉 Layer 2.5 → Layer 3 integration complete!")
    return df_enhanced, integration_metadata
