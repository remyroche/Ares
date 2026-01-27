"""
Layer 2 Checkpoint Manager

Manages checkpoints for Layer 2 sub-step execution, enabling:
- Resuming pipeline execution from any sub-step
- Cleaning up artifacts from a specific step onwards
- Automatic replacement of old checkpoints
"""

import json
import os
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Sub-step order (0-indexed) - must match pipeline execution order
LAYER2_SUBSTEPS = [
    'data_loading',           # 0: Load market data, dollar bars
    'regime_generation',      # 1: Generate regimes via AdaptiveHunterRouter
    'causal_initialization',  # 2: Initialize components, precompute features
    'causal_discovery',       # 3: Build causal DAG
    'causal_graph_saved',     # 3.5: Save completed causal graph
    'specialist_training',    # 4: Train AEDL/traditional specialists
    'event_generation',       # 5: Generate causal surprise events
    'feature_engineering',    # 6: Apply causal denoising
    'causal_targets',         # 6.5: Compute causal targets
    'dml_effects_computed',   # 6.6: DML causal effects computed
    'cate_computed',          # 6.7: CATE estimates computed
    'causal_model_training',  # 6.8: Train causal models
    'raw_candidates_selected', # 6.9: Raw geometry candidates selected
    'regime_optimization_progress', # 6.95: Partial regime progress
    'geometry_optimization',  # 7: De Prado protocol
    'model_race_complete',    # 7.5: Model race + deduplication complete
    'simulation_complete',    # 7.6: Simulations + Layer-12 complete
    'final_processing',       # 8: OOF analytics, reports
]


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    step_name: str
    step_index: int
    timestamp: str
    symbol: str
    config_hash: str
    data_keys: List[str]


class Layer2CheckpointManager:
    """
    Manages checkpoints for Layer 2 sub-step execution.
    
    Checkpoints are stored in versioned_artifacts/layer2_checkpoints/<symbol>/
    using HDF5 format for DataFrames and JSON for metadata.
    """
    
    SUBSTEPS = LAYER2_SUBSTEPS
    CHECKPOINT_DIR = Path("versioned_artifacts/layer2_checkpoints")
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Optional custom checkpoint directory path
        """
        self.checkpoint_dir = checkpoint_dir or self.CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_cache: Dict[str, Dict[str, CheckpointMetadata]] = {}
    
    def _get_symbol_dir(self, symbol: str) -> Path:
        """Get or create the checkpoint directory for a symbol."""
        symbol_dir = self.checkpoint_dir / symbol.upper()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir
    
    def _get_step_index(self, step: str) -> int:
        """Get the index of a step in the pipeline."""
        if step not in self.SUBSTEPS:
            raise ValueError(f"Unknown step: {step}. Valid steps: {self.SUBSTEPS}")
        return self.SUBSTEPS.index(step)
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute a hash of the config for versioning."""
        # Filter to relevant config keys only
        relevant_keys = [
            'symbol',
            'timeframe',
            'execution_mode',
            'direction',
            'exchange',
            'assets',
            'multi_asset_mode',
            'layer2_checkpoint_symbol',
        ]
        filtered = {k: v for k, v in config.items() if k in relevant_keys}
        config_str = json.dumps(filtered, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def validate_checkpoint_data(self, step: str, data: Dict[str, Any]) -> None:
        """
        Validate checkpoint data content before saving.
        
        Args:
            step: Name of the sub-step
            data: Dictionary of data to checkpoint
            
        Raises:
            ValueError: If data is invalid (empty, all zeros, etc.)
        """
        if step == 'causal_targets':
            if 'causal_targets_df' in data:
                df = data['causal_targets_df']
                if df.empty:
                    raise ValueError(f"❌ Invalid checkpoint for {step}: causal_targets_df is empty")
                
                # Check for zero values indicating failure
                if 'cate_estimates' in df.columns:
                    if (df['cate_estimates'] == 0).all():
                        raise ValueError(f"❌ Invalid checkpoint for {step}: all cate_estimates are zero")
                        
                if 'causal_residuals' in df.columns:
                    if (df['causal_residuals'] == 0).all():
                        raise ValueError(f"❌ Invalid checkpoint for {step}: all causal_residuals are zero")
                        
        elif step == 'event_generation':
            if 'causal_events_df' in data:
                df = data['causal_events_df']
                if df.empty:
                    raise ValueError(f"❌ Invalid checkpoint for {step}: causal_events_df is empty")
                    
        elif step == 'specialist_training':
             if 'specialist_predictions' in data:
                preds = data['specialist_predictions']
                if not preds and not isinstance(preds, (pd.DataFrame, pd.Series)): # handle dict
                     raise ValueError(f"❌ Invalid checkpoint for {step}: specialist_predictions is empty")


    def save_checkpoint(
        self, 
        step: str, 
        data: Dict[str, Any], 
        symbol: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a checkpoint for a given sub-step.
        
        Automatically replaces any existing checkpoint for the same step.
        
        Args:
            step: Name of the sub-step (e.g., 'causal_discovery')
            data: Dictionary of data to checkpoint (DataFrames, dicts, lists, etc.)
            symbol: Trading symbol (e.g., 'ETHUSDT')
            config: Optional pipeline configuration for versioning
            
        Returns:
            Path to the saved checkpoint
        """
        # Validate data before proceeding
        self.validate_checkpoint_data(step, data)

        step_idx = self._get_step_index(step)
        symbol_dir = self._get_symbol_dir(symbol)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = self._compute_config_hash(config or {})
        
        # File paths
        checkpoint_base = symbol_dir / f"checkpoint_{step}"
        h5_path = checkpoint_base.with_suffix('.h5')
        meta_path = checkpoint_base.with_suffix('.json')
        
        # Remove old checkpoint if exists (automatic replacement)
        if h5_path.exists():
            h5_path.unlink()
            logger.info(f"Replaced existing checkpoint: {h5_path.name}")
        if meta_path.exists():
            meta_path.unlink()
        
        # Save DataFrames to HDF5 and other data to JSON
        data_keys = list(data.keys())
        json_data = {}
        
        try:
            import tables  # Check if PyTables is available
            with pd.HDFStore(h5_path, mode='w') as store:
                for key, value in data.items():
                    if isinstance(value, pd.DataFrame):
                        try:
                            store.put(key, value, format='table', data_columns=True)
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to save {key} as HDF5 table (likely too many columns): {e}. Falling back to fixed format.")
                            store.put(key, value, format='fixed')
                    elif isinstance(value, pd.Series):
                        try:
                            store.put(key, value.to_frame(), format='table')
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to save {key} as HDF5 table: {e}. Falling back to fixed format.")
                            store.put(key, value.to_frame(), format='fixed')
                    elif isinstance(value, np.ndarray):
                        try:
                            store.put(key, pd.DataFrame(value), format='table')
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to save {key} as HDF5 table: {e}. Falling back to fixed format.")
                            store.put(key, pd.DataFrame(value), format='fixed')
                    else:
                        # For everything else, use robust recursive serialization for JSON metadata
                        json_data[key] = self._serialize_for_json(value)
        except Exception as e:
            # Fallback: Use pickle if tables not available or generic failure
            logger.warning(f"⚠️ HDF5 storage failed: {e}. Falling back to pickle.")
            import pickle
            # Clean up the partial/failed HDF5 file
            if h5_path.exists():
                h5_path.unlink()

            with open(h5_path, 'wb') as f:
                pickle.dump(data, f)
            logger.warning("Using pickle for checkpoint due to HDF5 failure")
        
        # Save metadata
        metadata = CheckpointMetadata(
            step_name=step,
            step_index=step_idx,
            timestamp=timestamp,
            symbol=symbol.upper(),
            config_hash=config_hash,
            data_keys=data_keys
        )
        
        meta_content = {
            **asdict(metadata),
            'json_data': json_data
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta_content, f, indent=2, default=str)
        
        logger.info(f"✅ Saved checkpoint '{step}' for {symbol} ({len(data_keys)} keys)")
        
        # Invalidate cache
        if symbol.upper() in self._metadata_cache:
            del self._metadata_cache[symbol.upper()]
        
        return h5_path
    
    def _serialize_for_json(self, obj: Any) -> Any:
        """
        Recursively serialize an object for JSON storage.
        
        Handles:
        - dict: Recursively serializes values and converts keys to strings
        - list/tuple: Recursively serializes elements
        - pd.DataFrame/pd.Series: Converts to dict and recursively serializes
        - pd.Timestamp/datetime: Converts to ISO string
        - np.ndarray: Converts to list and recursively serializes
        - Other types: Uses str(obj) as fallback if not primitive
        """
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        
        if isinstance(obj, np.ndarray):
            return self._serialize_for_json(obj.tolist())
        
        if isinstance(obj, pd.Index):
            return self._serialize_for_json(obj.tolist())

        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return self._serialize_for_json(obj.to_dict())
        
        if isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item) for item in obj]
        
        if isinstance(obj, dict):
            serialized_dict = {}
            for k, v in obj.items():
                # Ensure key is a string
                if isinstance(k, (pd.Timestamp, datetime)):
                    key = k.isoformat()
                elif not isinstance(k, (str, int, float, bool, type(None))):
                    key = str(k)
                else:
                    key = k
                serialized_dict[str(key)] = self._serialize_for_json(v)
            return serialized_dict
        
        # Fallback for others
        return str(obj)

    def _serialize_dict(self, d: Dict) -> Dict:
        """Deprecated: Use _serialize_for_json instead."""
        return self._serialize_for_json(d)
    
    def load_checkpoint(
        self, 
        step: str, 
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint for a given sub-step.
        
        Args:
            step: Name of the sub-step
            symbol: Trading symbol
            
        Returns:
            Dictionary of checkpoint data, or None if not found
        """
        symbol_dir = self._get_symbol_dir(symbol)
        checkpoint_base = symbol_dir / f"checkpoint_{step}"
        h5_path = checkpoint_base.with_suffix('.h5')
        meta_path = checkpoint_base.with_suffix('.json')
        
        if not h5_path.exists() or not meta_path.exists():
            logger.warning(f"Checkpoint not found for step '{step}' ({symbol})")
            return None
        
        # Load metadata
        with open(meta_path, 'r') as f:
            meta_content = json.load(f)
        
        json_data = meta_content.get('json_data', {})
        data_keys = meta_content.get('data_keys', [])
        
        # Load data
        data = {}
        
        try:
            import tables
            with pd.HDFStore(h5_path, mode='r') as store:
                for key in store.keys():
                    clean_key = key.lstrip('/')
                    data[clean_key] = store.get(key)
        except ImportError:
            # Fallback: Use pickle
            import pickle
            try:
                with open(h5_path, 'rb') as f:
                    data = pickle.load(f)
            except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
                logger.warning(f"⚠️ Corrupted pickle checkpoint file {h5_path}: {e}")
                logger.info(f"🗑️ Deleting corrupted checkpoint file...")
                try:
                    os.remove(h5_path)
                    logger.info(f"✅ Deleted corrupted checkpoint file")
                except OSError as remove_error:
                    logger.error(f"❌ Failed to delete corrupted checkpoint file: {remove_error}")
                return None
        
        # Merge JSON data
        for key, value in json_data.items():
            if key not in data:
                data[key] = value
        
        logger.info(f"✅ Loaded checkpoint '{step}' for {symbol} ({len(data)} keys)")
        return data
    
    def delete_checkpoints_from(self, step: str, symbol: str) -> int:
        """
        Delete checkpoints from a specific step onwards.
        
        Args:
            step: Starting step to delete from (inclusive)
            symbol: Trading symbol
            
        Returns:
            Number of checkpoints deleted
        """
        start_idx = self._get_step_index(step)
        symbol_dir = self._get_symbol_dir(symbol)
        deleted = 0
        
        for i in range(start_idx, len(self.SUBSTEPS)):
            step_name = self.SUBSTEPS[i]
            checkpoint_base = symbol_dir / f"checkpoint_{step_name}"
            h5_path = checkpoint_base.with_suffix('.h5')
            meta_path = checkpoint_base.with_suffix('.json')
            
            if h5_path.exists():
                h5_path.unlink()
                deleted += 1
            if meta_path.exists():
                meta_path.unlink()
        
        logger.info(f"🗑️ Deleted {deleted} checkpoints from '{step}' onwards for {symbol}")
        
        # Invalidate cache
        if symbol.upper() in self._metadata_cache:
            del self._metadata_cache[symbol.upper()]
        
        return deleted
    
    def get_latest_checkpoint(self, symbol: str) -> Optional[Tuple[str, CheckpointMetadata]]:
        """
        Get the latest (highest step index) checkpoint for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (step_name, metadata) or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints(symbol)
        if not checkpoints:
            return None
        
        # Return highest index checkpoint
        latest = max(checkpoints, key=lambda x: x[1].step_index)
        return latest
    
    def list_checkpoints(self, symbol: str) -> List[Tuple[str, CheckpointMetadata]]:
        """
        List all available checkpoints for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of (step_name, metadata) tuples, ordered by step index
        """
        symbol_dir = self._get_symbol_dir(symbol)
        checkpoints = []
        
        for step in self.SUBSTEPS:
            meta_path = symbol_dir / f"checkpoint_{step}.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta_content = json.load(f)
                
                metadata = CheckpointMetadata(
                    step_name=meta_content['step_name'],
                    step_index=meta_content['step_index'],
                    timestamp=meta_content['timestamp'],
                    symbol=meta_content['symbol'],
                    config_hash=meta_content['config_hash'],
                    data_keys=meta_content['data_keys']
                )
                checkpoints.append((step, metadata))
        
        return sorted(checkpoints, key=lambda x: x[1].step_index)
    
    def get_auto_resume_step(self, symbol: str) -> str:
        """
        Automatically determine the best step to resume execution from.
        Finds the latest valid checkpoint and returns the NEXT step in the sequence.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Name of the step to start/resume execution from. 
            Returns 'data_loading' (first step) if no checkpoints exist.
        """
        latest = self.get_latest_checkpoint(symbol)
        
        if not latest:
            return self.SUBSTEPS[0]  # data_loading
            
        step_name, metadata = latest
        current_idx = metadata.step_index
        
        # If we have the last step, we might want to re-run it or just return it?
        # Typically we want to run the NEXT step.
        if current_idx < len(self.SUBSTEPS) - 1:
            next_step = self.SUBSTEPS[current_idx + 1]
            logger.info(f"🔄 Auto-resume: Found checkpoint '{step_name}', resuming from '{next_step}'")
            return next_step
        else:
            logger.info(f"✅ Pipeline completed (found final checkpoint '{step_name}'). Re-running final step.")
            return step_name

    def get_resume_point(self, symbol: str, requested_step: str) -> Optional[str]:
        """
        Determine the actual step to resume from based on available checkpoints.
        
        If the requested step has a valid predecessor checkpoint, returns the requested step.
        Otherwise, returns None (must start from beginning).
        
        Args:
            symbol: Trading symbol
            requested_step: The step user wants to resume from
            
        Returns:
            Step name to resume from, or None if not possible
        """
        requested_idx = self._get_step_index(requested_step)
        
        if requested_idx == 0:
            # Can always start from beginning
            return 'data_loading'
        
        # Need checkpoint from previous step
        prev_step = self.SUBSTEPS[requested_idx - 1]
        checkpoint = self.load_checkpoint(prev_step, symbol)
        
        if checkpoint is None:
            logger.warning(
                f"Cannot resume from '{requested_step}': "
                f"missing checkpoint for '{prev_step}'"
            )
            return None
        
        return requested_step
    
    def print_checkpoint_status(self, symbol: str) -> str:
        """
        Generate a formatted status string for checkpoints.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Formatted status string
        """
        checkpoints = self.list_checkpoints(symbol)
        
        if not checkpoints:
            return f"No checkpoints found for {symbol}"
        
        lines = [f"\n📋 Layer 2 Checkpoints for {symbol}:", ""]
        lines.append("  Step                      | Timestamp        | Config Hash")
        lines.append("  " + "-" * 60)
        
        for step, meta in checkpoints:
            status = "✓"
            lines.append(
                f"  {status} {step:<22} | {meta.timestamp} | {meta.config_hash}"
            )
        
        # Show missing steps
        existing = {s[0] for s in checkpoints}
        missing = [s for s in self.SUBSTEPS if s not in existing]
        if missing:
            lines.append("")
            lines.append(f"  Missing: {', '.join(missing)}")
        
        return "\n".join(lines)


# Convenience function for external use
def get_checkpoint_manager() -> Layer2CheckpointManager:
    """Get the singleton checkpoint manager instance."""
    if not hasattr(get_checkpoint_manager, '_instance'):
        get_checkpoint_manager._instance = Layer2CheckpointManager()
    return get_checkpoint_manager._instance
