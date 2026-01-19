"""
Checkpoint-Aware Script Runner

Automatically detects available checkpoints and resumes execution from the appropriate step.
Symbol-specific checkpoint management for Layer 2.5, Layer 3, and Layer 4.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import time

from .unified_checkpoint_manager import get_checkpoint_manager, get_all_checkpoint_managers

logger = logging.getLogger(__name__)

@dataclass
class ExecutionPlan:
    """Plan for resuming execution from checkpoints."""
    layer: str
    symbol: str
    resume_step: str
    start_from_beginning: bool
    available_checkpoints: List[Tuple[str, Any]]
    execution_order: List[str]

class CheckpointAwareRunner:
    """
    Universal checkpoint-aware runner for all layers.
    
    Automatically:
    1. Detects available checkpoints for a symbol
    2. Determines optimal resume point
    3. Executes pipeline with checkpoint integration
    4. Saves progress at each sub-step
    """
    
    def __init__(self, layer: str, symbol: str, checkpoint_dir: Optional[Path] = None):
        """
        Initialize checkpoint-aware runner.
        
        Args:
            layer: Layer name ('layer2', 'layer25', 'layer3', 'layer4')
            symbol: Trading symbol
            checkpoint_dir: Optional custom checkpoint directory
        """
        self.layer = layer.lower()
        self.symbol = symbol.upper()
        self.checkpoint_dir = checkpoint_dir
        
        # Get checkpoint manager
        self.manager = get_checkpoint_manager(self.layer, self.symbol, checkpoint_dir)
        
        # Analyze available checkpoints
        self.execution_plan = self._analyze_checkpoints()
        
        logger.info(f"🔧 Initialized checkpoint-aware runner for {self.layer} ({self.symbol})")
        logger.info(f"📍 Resume step: {self.execution_plan.resume_step}")
        logger.info(f"📋 Available checkpoints: {len(self.execution_plan.available_checkpoints)}")
    
    def _analyze_checkpoints(self) -> ExecutionPlan:
        """Analyze available checkpoints and create execution plan."""
        # Get all checkpoints for this symbol/layer
        checkpoints = self.manager.list_checkpoints()
        available_steps = [step for step, _ in checkpoints]
        
        # Get all possible steps for this layer
        all_steps = self.manager.get_available_steps()
        
        # Determine resume step
        if checkpoints:
            # Find latest checkpoint
            latest_step, latest_metadata = max(checkpoints, key=lambda x: x[1].step_index)
            current_idx = latest_metadata.step_index
            
            # Resume from next step (or re-run current if it's the last step)
            if current_idx < len(all_steps) - 1:
                resume_step = all_steps[current_idx + 1]
                start_from_beginning = False
            else:
                resume_step = latest_step  # Re-run final step
                start_from_beginning = False
                
            logger.info(f"🔄 Found checkpoint '{latest_step}' (index {current_idx}), resuming from '{resume_step}'")
        else:
            # No checkpoints available
            resume_step = all_steps[0]  # Start from beginning
            start_from_beginning = True
            logger.info(f"🆕 No checkpoints found, starting from '{resume_step}'")
        
        # Determine execution order (from resume step onwards)
        resume_idx = self.manager.get_step_index(resume_step)
        execution_order = all_steps[resume_idx:]
        
        return ExecutionPlan(
            layer=self.layer,
            symbol=self.symbol,
            resume_step=resume_step,
            start_from_beginning=start_from_beginning,
            available_checkpoints=checkpoints,
            execution_order=execution_order
        )
    
    def run_with_checkpoints(
        self,
        step_functions: Dict[str, Callable],
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run pipeline with automatic checkpoint integration.
        
        Args:
            step_functions: Dictionary mapping step names to functions
            config: Pipeline configuration
            **kwargs: Additional arguments passed to step functions
            
        Returns:
            Dictionary with final results and execution metadata
        """
        logger.info(f"🚀 Starting {self.layer} execution for {self.symbol}")
        logger.info(f"📋 Execution plan: {self.execution_plan.execution_order}")
        
        results = {}
        execution_metadata = {
            'layer': self.layer,
            'symbol': self.symbol,
            'start_time': time.time(),
            'resume_step': self.execution_plan.resume_step,
            'steps_executed': [],
            'checkpoints_saved': [],
            'start_from_beginning': self.execution_plan.start_from_beginning
        }
        
        try:
            # Execute each step in order
            for step_name in self.execution_plan.execution_order:
                logger.info(f"⏭️ Executing step: {step_name}")
                
                # Check if we should load from checkpoint or execute
                checkpoint_data = self.manager.load_checkpoint(step_name)
                
                if checkpoint_data and step_name != self.execution_plan.resume_step:
                    # Load from checkpoint (skip execution)
                    logger.info(f"📂 Loading {step_name} from checkpoint")
                    results[step_name] = checkpoint_data
                    execution_metadata['steps_executed'].append(f"{step_name} (loaded)")
                else:
                    # Execute step function
                    if step_name not in step_functions:
                        logger.error(f"❌ No function defined for step: {step_name}")
                        continue
                    
                    step_func = step_functions[step_name]
                    step_start_time = time.time()
                    
                    try:
                        # Execute step with previous results and config
                        step_result = step_func(
                            results=results,
                            config=config,
                            symbol=self.symbol,
                            **kwargs
                        )
                        
                        # Save checkpoint
                        checkpoint_path = self.manager.save_checkpoint(step_name, step_result, config)
                        execution_metadata['checkpoints_saved'].append(str(checkpoint_path))
                        
                        results[step_name] = step_result
                        execution_metadata['steps_executed'].append(f"{step_name} (executed)")
                        
                        step_duration = time.time() - step_start_time
                        logger.info(f"✅ Completed {step_name} in {step_duration:.2f}s")
                        
                    except Exception as e:
                        logger.error(f"❌ Step {step_name} failed: {e}")
                        execution_metadata['steps_executed'].append(f"{step_name} (failed)")
                        raise
            
            execution_metadata['end_time'] = time.time()
            execution_metadata['total_duration'] = execution_metadata['end_time'] - execution_metadata['start_time']
            execution_metadata['status'] = 'completed'
            
            logger.info(f"🎉 {self.layer} execution completed in {execution_metadata['total_duration']:.2f}s")
            
            return {
                'results': results,
                'metadata': execution_metadata
            }
            
        except Exception as e:
            execution_metadata['end_time'] = time.time()
            execution_metadata['total_duration'] = execution_metadata['end_time'] - execution_metadata['start_time']
            execution_metadata['status'] = 'failed'
            execution_metadata['error'] = str(e)
            
            logger.error(f"❌ {self.layer} execution failed: {e}")
            raise
    
    def get_available_steps(self) -> List[str]:
        """
        Get the list of available sub-steps for the current layer.
        
        Returns:
            List of step names in execution order
        """
        return self.manager.get_available_steps()
    
    def get_checkpoint_status(self) -> Dict[str, Any]:
        """Get detailed checkpoint status for this symbol/layer."""
        checkpoints = self.manager.list_checkpoints()
        
        status = {
            'layer': self.layer,
            'symbol': self.symbol,
            'total_checkpoints': len(checkpoints),
            'available_checkpoints': [],
            'latest_checkpoint': None,
            'resume_step': self.execution_plan.resume_step,
            'start_from_beginning': self.execution_plan.start_from_beginning,
            'completion_percentage': 0.0
        }
        
        if checkpoints:
            # Latest checkpoint
            latest_step, latest_metadata = max(checkpoints, key=lambda x: x[1].step_index)
            status['latest_checkpoint'] = {
                'step': latest_step,
                'timestamp': latest_metadata.timestamp,
                'step_index': latest_metadata.step_index,
                'data_keys': latest_metadata.data_keys
            }
            
            # Completion percentage
            total_steps = len(self.manager.get_available_steps())
            completed_steps = latest_metadata.step_index + 1
            status['completion_percentage'] = (completed_steps / total_steps) * 100
            
            # Available checkpoints list
            for step, metadata in checkpoints:
                status['available_checkpoints'].append({
                    'step': step,
                    'timestamp': metadata.timestamp,
                    'step_index': metadata.step_index,
                    'data_keys': len(metadata.data_keys)
                })
        
        return status
    
    def clean_checkpoints_from(self, step: str) -> int:
        """Clean checkpoints from a specific step onwards."""
        deleted_count = self.manager.delete_checkpoints_from(step)
        logger.info(f"🗑️ Cleaned {deleted_count} checkpoints from '{step}' onwards")
        return deleted_count
    
    def reset_all_checkpoints(self) -> int:
        """Reset all checkpoints for this symbol/layer."""
        return self.clean_checkpoints_from(self.manager.get_available_steps()[0])

def create_checkpoint_aware_runner(layer: str, symbol: str, checkpoint_dir: Optional[Path] = None) -> CheckpointAwareRunner:
    """
    Create a checkpoint-aware runner for any layer.
    
    Args:
        layer: Layer name ('layer2', 'layer25', 'layer3', 'layer4')
        symbol: Trading symbol
        checkpoint_dir: Optional custom checkpoint directory
        
    Returns:
        CheckpointAwareRunner instance
    """
    return CheckpointAwareRunner(layer, symbol, checkpoint_dir)

def run_layer_with_checkpoints(
    layer: str,
    symbol: str,
    step_functions: Dict[str, Callable],
    config: Dict[str, Any],
    checkpoint_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run any layer with automatic checkpoint management.
    
    Args:
        layer: Layer name
        symbol: Trading symbol
        step_functions: Dictionary mapping step names to functions
        config: Pipeline configuration
        checkpoint_dir: Optional checkpoint directory
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with results and metadata
    """
    runner = create_checkpoint_aware_runner(layer, symbol, checkpoint_dir)
    return runner.run_with_checkpoints(step_functions, config, **kwargs)

def get_symbol_checkpoint_status(symbol: str, checkpoint_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get checkpoint status for all layers of a symbol.
    
    Args:
        symbol: Trading symbol
        checkpoint_dir: Optional checkpoint directory
        
    Returns:
        Dictionary with layer-wise checkpoint status
    """
    from .unified_checkpoint_manager import get_all_checkpoint_managers
    
    all_managers = get_all_checkpoint_managers(symbol, checkpoint_dir)
    status = {'symbol': symbol.upper(), 'layers': {}}
    
    for layer_name, manager in all_managers.items():
        runner = CheckpointAwareRunner(layer_name, symbol, checkpoint_dir)
        status['layers'][layer_name] = runner.get_checkpoint_status()
    
    return status

# Decorator for automatic checkpoint integration
def checkpoint_aware_step(step_name: str):
    """
    Decorator to automatically integrate checkpoint saving for a step function.
    
    Usage:
    @checkpoint_aware_step('dual_head_training')
    def train_models(results, config, symbol, **kwargs):
        # Training logic here
        return {'models': trained_models, 'predictions': predictions}
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract checkpoint manager if available
            manager = kwargs.get('checkpoint_manager')
            
            # Check if we should load from checkpoint
            if manager and step_name in kwargs.get('skip_steps', []):
                checkpoint_data = manager.load_checkpoint(step_name)
                if checkpoint_data:
                    logger.info(f"📂 Loading {step_name} from checkpoint")
                    return checkpoint_data
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Save checkpoint if manager available
            if manager and 'config' in kwargs:
                try:
                    manager.save_checkpoint(step_name, result, kwargs['config'])
                    logger.info(f"💾 Saved checkpoint for {step_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save checkpoint for {step_name}: {e}")
            
            return result
        return wrapper
    return decorator
