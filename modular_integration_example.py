#!/usr/bin/env python3
"""
ModularComponent Integration Example

This script demonstrates the complete ModularComponent integration
with monitoring, dashboard, and pipeline orchestration.
"""

import sys
import os
import logging
import time
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging():
    """Setup logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    return data

def main():
    """Main example function."""
    print("🚀 ModularComponent Integration Example")
    print("=" * 50)
    
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Import ModularComponent classes
        from src.training.steps.pre_training.unified_data_driven_pipeline.core import (
            ModularComponent, create_monitor, create_dashboard,
            ModularPipelineOrchestrator, create_modular_pipeline_orchestrator
        )
        
        print("✅ Successfully imported ModularComponent classes")
        
        # Create a simple test component
        class TestDataProcessor(ModularComponent):
            def _initialize_resources(self) -> bool:
                self.set_state('processed_count', 0)
                self.set_state('initialized_at', time.time())
                return True
            
            def _cleanup_resources(self) -> None:
                self.set_state('cleaned_up_at', time.time())
            
            def _process_data(self, data, **kwargs):
                # Simulate some processing
                processed_count = self.get_state('processed_count', 0)
                self.set_state('processed_count', processed_count + 1)
                
                # Simulate processing time
                time.sleep(0.1)
                
                # Return processed data
                if isinstance(data, pd.DataFrame):
                    return data.copy()
                return data
            
            def _get_validation_rules(self):
                return {
                    'min_size': 10,
                    'max_size': 10000,
                    'required_attributes': ['open', 'high', 'low', 'close'],
                    'data_types': ['pandas.DataFrame']
                }
            
            def _validate_component_specific(self, data):
                errors = []
                warnings = []
                metadata = {}
                
                if isinstance(data, pd.DataFrame):
                    if len(data) < 10:
                        warnings.append("Data size is small")
                    metadata['shape'] = data.shape
                
                return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
        
        print("✅ Created test component class")
        
        # Create monitoring system
        monitor = create_monitor(
            log_file="modular_monitoring.log",
            alert_thresholds={
                'error_rate': 0.1,
                'avg_execution_time': 1.0,
                'memory_usage_mb': 100.0,
                'health_score': 0.8
            }
        )
        print("✅ Created monitoring system")
        
        # Create pipeline orchestrator
        orchestrator = create_modular_pipeline_orchestrator({
            'monitoring_enabled': True,
            'performance_tracking': True
        })
        print("✅ Created pipeline orchestrator")
        
        # Create test components
        processor1 = TestDataProcessor("data_processor_1", {})
        processor2 = TestDataProcessor("data_processor_2", {})
        
        # Initialize components
        if processor1.initialize():
            print("✅ Initialized processor 1")
        if processor2.initialize():
            print("✅ Initialized processor 2")
        
        # Register components with monitoring
        monitor.register_component(processor1)
        monitor.register_component(processor2)
        print("✅ Registered components with monitoring")
        
        # Register components with orchestrator
        orchestrator.register_component("processor1", processor1)
        orchestrator.register_component("processor2", processor2)
        print("✅ Registered components with orchestrator")
        
        # Create sample data
        sample_data = create_sample_data()
        print(f"✅ Created sample data: {sample_data.shape}")
        
        # Simulate some processing with monitoring
        print("\n🔄 Simulating component processing...")
        for i in range(5):
            try:
                # Process with component 1
                start_time = time.time()
                result1 = processor1._safe_process(sample_data)
                execution_time = time.time() - start_time
                
                monitor.record_execution(
                    "data_processor_1", 
                    execution_time, 
                    True,
                    memory_usage_mb=50.0
                )
                
                # Process with component 2
                start_time = time.time()
                result2 = processor2._safe_process(sample_data)
                execution_time = time.time() - start_time
                
                monitor.record_execution(
                    "data_processor_2", 
                    execution_time, 
                    True,
                    memory_usage_mb=45.0
                )
                
                print(f"  Processing iteration {i+1}/5 completed")
                time.sleep(0.5)  # Small delay between iterations
                
            except Exception as e:
                print(f"  Error in iteration {i+1}: {e}")
                monitor.record_execution(
                    "data_processor_1" if i % 2 == 0 else "data_processor_2",
                    0.0,
                    False,
                    error_message=str(e)
                )
        
        print("✅ Completed processing simulation")
        
        # Display monitoring results
        print("\n📊 MONITORING RESULTS")
        print("-" * 40)
        
        # Get pipeline metrics
        pipeline_metrics = monitor.get_pipeline_metrics()
        print(f"Overall Health: {pipeline_metrics.overall_health_score:.1%}")
        print(f"Total Components: {pipeline_metrics.total_components}")
        print(f"Healthy Components: {pipeline_metrics.healthy_components}")
        print(f"Total Executions: {pipeline_metrics.total_executions}")
        print(f"Success Rate: {(pipeline_metrics.total_successes/max(1, pipeline_metrics.total_executions)):.1%}")
        
        # Get component details
        print(f"\nComponent Details:")
        for name in ["data_processor_1", "data_processor_2"]:
            metrics = monitor.get_component_metrics(name)
            if metrics:
                print(f"  {name}: {metrics.status} ({metrics.health_score:.1%}) - {metrics.execution_count} executions")
        
        # Get recommendations
        recommendations = monitor.get_performance_recommendations()
        if recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec}")
        
        # Test orchestrator functionality
        print(f"\n🔧 ORCHESTRATOR TEST")
        print("-" * 40)
        
        # Get pipeline health
        health = orchestrator.get_pipeline_health()
        print(f"Pipeline Health: {health['overall_health']}")
        print(f"Component Count: {health['total_components']}")
        
        # Test component execution through orchestrator
        try:
            result = orchestrator.execute_component("processor1", sample_data)
            print("✅ Orchestrator component execution successful")
        except Exception as e:
            print(f"❌ Orchestrator execution failed: {e}")
        
        # Test state management
        print(f"\n💾 STATE MANAGEMENT TEST")
        print("-" * 40)
        
        # Check component states
        state1 = processor1.get_all_state()
        state2 = processor2.get_all_state()
        
        print(f"Processor 1 state keys: {list(state1.keys())}")
        print(f"Processor 2 state keys: {list(state2.keys())}")
        
        # Test serialization
        try:
            serialized = processor1.serialize()
            print(f"✅ Component serialization successful (keys: {len(serialized)})")
        except Exception as e:
            print(f"❌ Serialization failed: {e}")
        
        # Test performance stats
        stats1 = processor1.get_performance_stats()
        stats2 = processor2.get_performance_stats()
        
        print(f"Processor 1 performance: {stats1['success_rate']:.1%} success rate")
        print(f"Processor 2 performance: {stats2['success_rate']:.1%} success rate")
        
        # Cleanup
        processor1.cleanup()
        processor2.cleanup()
        monitor.stop_monitoring()
        
        print(f"\n✅ Integration example completed successfully!")
        print("=" * 50)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed and paths are correct")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)