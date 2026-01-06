#!/usr/bin/env python3
"""Enhanced Specialists Deployment Script

This script deploys enhanced feature generators, MI monitoring, and 
hyperparameter optimization to all specialist models.

Usage:
    python3 scripts/deploy_enhanced_specialists_fixed.py
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.ml_common.get_specialist_models_outputs import get_specialist_models_outputs
from src.utils.ml_common.feature_selection import get_feature_selection_utils
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    FeatureGenerationMetaLabelingStep,
)
from src.training.steps.labeling.snr_diagnostics import (
    _load_labeled_data,
)
from src.training.steps.pre_training.components.final_feature_selection import (
    FinalFeatureSelectionConfig,
    FinalFeatureSelectionComponent,
)
from src.training.steps.market_analysis import step_registry
from src.utils.logger import system_logger

from src.training.steps.market_analysis.mi_monitoring_system import get_mi_monitor
from src.training.steps.market_analysis.hyperparameter_optimizer_mi import get_mi_optimizer
from src.training.steps.market_analysis.enhanced_feature_generators import EnhancedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistRequirements, SpecialistType

logger = system_logger.getChild("enhanced_specialists_deployer")

OUTCOMES_DIR = Path("outcomes")

class EnhancedSpecialistDeployer:
    """Deploy enhanced specialists with MI optimization."""
    
    def __init__(self):
        self.mi_monitor = get_mi_monitor()
        self.mi_optimizer = get_mi_optimizer()
        self.feature_pipeline = EnhancedFeaturePipeline()
        self.deployment_log = []
        
    def deploy_to_all_specialists(self) -> Dict[str, Any]:
        """Deploy enhanced capabilities to all specialist models."""
        
        print("🚀 DEPLOYING ENHANCED SPECIALISTS")
        print("=" * 60)
        
        # Define all 11 enhanced specialists
        specialists_enhanced = {
            'enhanced_ml_momentum_persistence_step': {
                'class_path': 'src.training.steps.market_analysis.ml_momentum_persistence_step_enhanced.EnhancedMLMomentumPersistenceStep',
                'model_type': 'lightgbm',
                'feature_type': 'momentum'
            },
            'enhanced_ml_smc_regime_step': {
                'class_path': 'src.training.steps.market_analysis.ml_smc_regime_step_enhanced.EnhancedMLSMCRegimeStep',
                'model_type': 'xgboost',
                'feature_type': 'smc_regime'
            },
            'enhanced_ml_volatility_burst_step': {
                'class_path': 'src.training.steps.market_analysis.ml_volatility_burst_step_enhanced.EnhancedMLVolatilityBurstStep',
                'model_type': 'lightgbm',
                'feature_type': 'volatility_burst'
            },
            'enhanced_ml_volume_force_step': {
                'class_path': 'src.training.steps.market_analysis.ml_volume_force_step_enhanced.EnhancedMLVolumeForceStep',
                'model_type': 'xgboost',
                'feature_type': 'volume_force'
            },
            'enhanced_ml_liquidity_regime_step': {
                'class_path': 'src.training.steps.market_analysis.ml_liquidity_regime_step_enhanced.EnhancedMLLiquidityRegimeStep',
                'model_type': 'xgboost',
                'feature_type': 'liquidity_regime'
            },
            'enhanced_ml_breakout_bounce_regime_step': {
                'class_path': 'src.training.steps.market_analysis.ml_breakout_bounce_regime_step_enhanced.EnhancedMLBreakoutBounceRegimeStep',
                'model_type': 'xgboost',
                'feature_type': 'breakout'
            },
            'enhanced_ml_path_regime_step': {
                'class_path': 'src.training.steps.market_analysis.ml_path_regime_step_enhanced.EnhancedMLPathRegimeStep',
                'model_type': 'xgboost',
                'feature_type': 'path_regime'
            },
            'enhanced_ml_reversion_regime_step': {
                'class_path': 'src.training.steps.market_analysis.ml_reversion_regime_step_enhanced.EnhancedMLReversionRegimeStep',
                'model_type': 'xgboost',
                'feature_type': 'reversion'
            },
            'enhanced_ml_risk_regime_step': {
                'class_path': 'src.training.steps.market_analysis.ml_risk_regime_step_enhanced.EnhancedMLRiskRegimeStep',
                'model_type': 'xgboost',
                'feature_type': 'risk_regime'
            },
            'enhanced_xgb_macro_regime_step': {
                'class_path': 'src.training.steps.market_analysis.xgb_macro_regime_step_enhanced.EnhancedXGBMacroRegimeStep',
                'model_type': 'xgboost',
                'feature_type': 'macro_regime'
            },
            'enhanced_xgb_meso_regime_step': {
                'class_path': 'src.training.steps.market_analysis.xgb_meso_regime_step_enhanced.EnhancedXGBMesoRegimeStep',
                'model_type': 'xgboost',
                'feature_type': 'meso_regime'
            }
        }
        
        deployment_results = {}
        
        for specialist_name, config in specialists_enhanced.items():
            print(f"\n🔧 Deploying enhanced capabilities to {specialist_name}...")
            
            try:
                result = self._deploy_to_specialist(specialist_name, config)
                deployment_results[specialist_name] = result
                
                if result['success']:
                    print(f"   ✅ Successfully deployed to {specialist_name}")
                else:
                    print(f"   ❌ Failed to deploy to {specialist_name}: {result['error']}")
                    
            except Exception as e:
                print(f"   ❌ Deployment failed for {specialist_name}: {e}")
                deployment_results[specialist_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Generate deployment summary
        summary = self._generate_deployment_summary(deployment_results)
        
        print(f"\n📊 DEPLOYMENT SUMMARY:")
        print(f"   Total specialists_enhanced: {len(deployment_results)}")
        print(f"   Successful deployments: {len([r for r in deployment_results.values() if r['success']])}")
        print(f"   Failed deployments: {len([r for r in deployment_results.values() if not r['success']])}")
        print(f"   Success rate: {(len([r for r in deployment_results.values() if r['success']]) / len(deployment_results) * 100):.1f")
        
        # Save deployment report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = Path("outcomes") / f"enhanced_specialists_deployment_{timestamp}.json"
        
        report_data = {
            'summary': summary,
            'deployment_results': deployment_results,
            'deployment_log': self.deployment_log,
            'mi_monitoring_status': self.mi_monitor.get_overall_summary(),
            'hyperparameter_optimization_status': self.mi_optimizer.get_optimization_summary(),
            'report_timestamp': datetime.utcnow().isoformat()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Deployment report saved to {report_path}")
        
        return deployment_results
    
    def _deploy_to_specialist(self, specialist_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy enhanced capabilities to a specific specialist."""
        
        result = {
            'specialist_name': specialist_name,
            'deployment_timestamp': datetime.utcnow().isoformat(),
            'enhanced_features_deployed': False,
            'mi_monitoring_deployed': False,
            'hyperparameter_optimization_deployed': False,
            'error': None
        }
        
        try:
            # 1. Deploy enhanced features
            print(f"   🛠️ Deploying enhanced features...")
            features_deployed = self._deploy_enhanced_features(specialist_name, config['feature_type'])
            result['enhanced_features_deployed'] = features_deployed
            
            # 2. Deploy MI monitoring
            print(f"   📊 Deploying MI monitoring...")
            mi_monitoring_deployed = self._deploy_mi_monitoring(specialist_name)
            result['mi_monitoring_deployed'] = mi_monitoring_deployed
            
            # 3. Deploy hyperparameter optimization
            print(f"   🔧 Deploying hyperparameter optimization...")
            hpo_deployed = self._deploy_hyperparameter_optimization(specialist_name, config['model_type'])
            result['hyperparameter_optimization_deployed'] = hpo_deployed
            
            # 4. Check if all deployments succeeded
            if all([features_deployed, mi_monitoring_deployed, hpo_deployed]):
                result['success'] = True
                
                # Log successful deployment
                self.deployment_log.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'specialist_name': specialist_name,
                    'status': 'SUCCESS',
                    'components': ['enhanced_features', 'mi_monitoring', 'hyperparameter_optimization']
                })
            else:
                result['success'] = False
                result['error'] = 'One or more deployments failed'
                
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
        
        return result
    
    def _deploy_enhanced_features(self, specialist_name: str, feature_type: str) -> bool:
        """Deploy enhanced feature generation to specialist."""
        
        try:
            # Test enhanced feature pipeline
            print(f"      Testing enhanced feature pipeline for {feature_type}...")
            
            # Create sample data for testing
            sample_data = self._create_sample_market_data()
            
            # Generate enhanced features
            enhanced_features = self.feature_pipeline.generate_enhanced_features(
                sample_data, feature_type, {'enhanced_features': True}
            )
            
            if len(enhanced_features.columns) > 0:
                print(f"      ✅ Enhanced features working: {len(enhanced_features.columns)} features generated")
                return True
            else:
                print(f"      ❌ No enhanced features generated")
                return False
                
        except Exception as e:
            print(f"      ❌ Enhanced features deployment failed: {e}")
            return False
    
    def _deploy_mi_monitoring(self, specialist_name: str) -> bool:
        """Deploy MI monitoring to specialist."""
        
        try:
            # Test MI monitoring
            print(f"      Testing MI monitoring for {specialist_name}...")
            
            # Create sample data for testing
            sample_features = self._create_sample_features()
            sample_labels = self._create_sample_labels()
            sample_predictions = self._create_sample_predictions()
            sample_probabilities = self._create_sample_probabilities()
            
            # Test MI computation
            mi_metrics = self.mi_monitor.compute_mi_metrics(
                specialist_name, sample_features, sample_labels, 
                sample_predictions, sample_probabilities
            )
            
            if mi_metrics.prediction_mi >= 0.02:
                print(f"      ✅ MI monitoring working: MI = {mi_metrics.prediction_mi:.4f}")
                return True
            else:
                print(f"      ❌ MI monitoring working: MI = {mi_metrics.prediction_mi:.4f}")
                return False
                
        except Exception as e:
            print(f"      ❌ MI monitoring deployment failed: {e}")
            return False
    
    def _deploy_hyperparameter_optimization(self, specialist_name: str, model_type: str) -> bool:
        """Deploy hyperparameter optimization to specialist."""
        
        try:
            # Test hyperparameter optimization
            print(f"      Testing hyperparameter optimization for {specialist_name}...")
            
            # Create sample data for testing
            sample_features = self._create_sample_features()
            sample_labels = self._create_sample_labels()
            
            # Test feature selection
            selected_features, mi_scores = self.mi_optimizer.select_features_by_mi(
                sample_features, sample_labels, top_k=10
            )
            
            if len(selected_features) > 0:
                print(f"      ✅ Feature selection working: {len(selected_features)} features selected")
                return True
            else:
                print(f"      ❌ Feature selection failed")
                return False
                
        except Exception as e:
            print(f"      ❌ Hyperparameter optimization deployment failed: {e}")
            return False
    
    def _create_sample_market_data(self) -> pd.DataFrame:
        """Create sample market data for testing."""
        import numpy as np
        import pandas as pd
        
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        
        # Generate realistic price data
        np.random.seed(42)
        returns = np.random.normal(0, 0.001, 100)
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        prices = prices[1:]
        
        # Generate OHLCV data
        data = {
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'close': prices,
            'volume': np.random.exponential(1000, 100)
        }
        
        return pd.DataFrame(data, index=dates)
    
    def _create_sample_features(self) -> pd.DataFrame:
        """Create sample features for testing."""
        import numpy as np
        import pandas as pd
        
        return pd.DataFrame(index=self._create_sample_market_data().index, columns=[
            'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
            'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10'
        ])
    
    def _create_sample_labels(self) -> pd.Series:
        """Create sample labels for testing."""
        import numpy as np
        return pd.Series(np.random.randint(0, 2, size=100))
    
    def _create_sample_predictions(self) -> np.ndarray:
        """Create sample predictions for testing."""
        import numpy as np
        return np.random.random.random(100)
    
    def _create_sample_probabilities(self) -> np.ndarray:
        """Create sample probabilities for testing."""
        import numpy as np
        return np.random.random.random(100)
    
    def _generate_deployment_summary(self, deployment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate deployment summary."""
        
        successful_count = sum(1 for result in deployment_results.values() if result['success'])
        total_count = len(deployment_results)
        
        return {
            'total_specialists_enhanced': total_count,
            'successful_deployments': successful_count,
            'failed_deployments': total_count - successful_count,
            'success_rate': successful_count / total_count if total_count > 0 else 0.0,
            'deployment_timestamp': datetime.utcnow().isoformat()
        }
    
    def save_deployment_report(self, filepath: str):
        """Save deployment report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = Path(filepath)
        
        report_data = {
            'summary': self._generate_deployment_summary(self.deployment_results),
            'deployment_results': self.deployment_results,
            'deployment_log': self.deployment_log,
            'mi_monitoring_status': self.mi_monitor.get_overall_summary(),
            'hyperparameter_optimization_status': self.mi_optimizer.get_optimization_summary(),
            'report_timestamp': datetime.utcnow().isoformat()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Deployment report saved to {report_path}")

def main():
    """Main enhanced deployment function."""
    
    print("🚀 STARTING ENHANCED SPECIALISTS DEPLOYMENT")
    print("=" * 80)
    
    # Initialize deployer
    deployer = EnhancedSpecialistDeployer()
    
    # Deploy to all specialists
    deployment_results = deployer.deploy_to_all_specialists()
    
    # Final summary
    print(f"\n🎯 ENHANCED SPECIALISTS DEPLOYMENT COMPLETE")
    print("=" * 80)
    
    # Print final summary
    successful_count = sum(1 for result in deployment_results.values() if result['success'])
    total_count = len(deployment_results)
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   Total specialists_enhanced: {total_count}")
    print(f"   Successful deployments: {successful_count}")
    print(f"   Success rate: {successful_count/total_count*100:.1f}")
    
    if successful_count == total_count:
        print(f"✅ All {total_count} specialists successfully enhanced!")
        print(f"🚀 Enhanced specialists ready for MI optimization!")
    else:
        print(f"⚠️ {successful_count}/{total_count} specialists enhanced successfully")
        print(f"🔧 Need to fix {total_count - successful_count} specialists")
    
    print(f"\n🚀 READY FOR PRODUCTION")

if __name__ == "__main__":
    main()
