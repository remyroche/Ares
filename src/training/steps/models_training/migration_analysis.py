"""
Migration Analysis Script for Models Training Components

This script analyzes existing models_training components to determine
migration requirements and compatibility with the ModularComponent architecture.
"""

import os
import sys
import inspect
import logging
from typing import Dict, List, Any, Type
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.training.steps.models_training.unified_data_driven_pipeline.core.migration_utils import (
    ModelsTrainingMigrationUtils, analyze_component, generate_migration_report
)

# Import existing components for analysis (only tactician components remain)
try:
    from src.training.steps.models_training.tactician_models_training import (
        TacticianModelsTrainingStep, TacticianModelsTrainingConfig, TacticianModelsTrainingResult
    )
    TACTICIAN_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import tactician_models_training: {e}")
    TACTICIAN_MODELS_AVAILABLE = False

try:
    from src.training.steps.models_training.tactician_ensemble_training import (
        TacticianEnsembleTrainingStep, TacticianEnsembleTrainingConfig, TacticianEnsembleTrainingResult
    )
    TACTICIAN_ENSEMBLE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import tactician_ensemble_training: {e}")
    TACTICIAN_ENSEMBLE_AVAILABLE = False

# Analyst and ML components have been successfully migrated to ModularComponent architecture
ANALYST_MODELS_AVAILABLE = False
ANALYST_ENSEMBLE_AVAILABLE = False
ML_LABELER_AVAILABLE = False


def analyze_models_training_components():
    """Analyze all models_training components for migration."""
    print("🔍 Analyzing Models Training Components for Migration")
    print("=" * 60)
    
    # Initialize migration utils
    migration_utils = ModelsTrainingMigrationUtils()
    
    # Collect components to analyze (only tactician components remain)
    components_to_analyze = []
    
    if TACTICIAN_MODELS_AVAILABLE:
        components_to_analyze.extend([
            TacticianModelsTrainingStep,
            TacticianModelsTrainingConfig,
            TacticianModelsTrainingResult
        ])
    
    if TACTICIAN_ENSEMBLE_AVAILABLE:
        components_to_analyze.extend([
            TacticianEnsembleTrainingStep,
            TacticianEnsembleTrainingConfig,
            TacticianEnsembleTrainingResult
        ])
    
    # Note: Analyst and ML components have been successfully migrated
    print("✅ Analyst models training: MIGRATED to ModularComponent")
    print("✅ Analyst ensemble training: MIGRATED to ModularComponent") 
    print("✅ ML entry timing labeler: MIGRATED to ModularComponent")
    
    # Filter to only classes (not dataclasses/enums)
    class_components = [comp for comp in components_to_analyze if inspect.isclass(comp)]
    
    print(f"Found {len(class_components)} components to analyze:")
    for comp in class_components:
        print(f"  - {comp.__name__}")
    
    print("\n" + "=" * 60)
    
    # Generate migration report
    report = generate_migration_report(class_components)
    
    # Print summary
    print("📊 MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Total Components: {report['total_components']}")
    print(f"Compatible: {report['summary']['compatible']}")
    print(f"Incompatible: {report['summary']['incompatible']}")
    print(f"Easy Migration: {report['summary']['easy_migration']}")
    print(f"Medium Migration: {report['summary']['medium_migration']}")
    print(f"Hard Migration: {report['summary']['hard_migration']}")
    print(f"Very Hard Migration: {report['summary']['very_hard_migration']}")
    
    print("\n📋 RECOMMENDATIONS")
    print("=" * 60)
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    print("\n🔍 DETAILED ANALYSIS")
    print("=" * 60)
    
    for component_info in report['components']:
        print(f"\n📦 {component_info['name']}")
        print(f"   Compatible: {'✅' if component_info['compatible'] else '❌'}")
        print(f"   Compatibility Score: {component_info['compatibility_score']:.2f}")
        print(f"   Migration Difficulty: {component_info['migration_difficulty']}")
        print(f"   Methods: {len(component_info['methods'])}")
        print(f"   Dependencies: {', '.join(component_info['dependencies'])}")
        
        if component_info['recommendations']:
            print("   Recommendations:")
            for rec in component_info['recommendations']:
                print(f"     • {rec}")
    
    return report


def create_migration_plan(report: Dict[str, Any]) -> Dict[str, Any]:
    """Create a detailed migration plan based on analysis."""
    print("\n📋 CREATING MIGRATION PLAN")
    print("=" * 60)
    
    migration_plan = {
        'phases': [],
        'priority_order': [],
        'estimated_effort': {},
        'dependencies': {}
    }
    
    # Phase 1: Easy migrations
    easy_components = [
        comp for comp in report['components'] 
        if comp['migration_difficulty'] == 'easy' and comp['compatible']
    ]
    
    if easy_components:
        migration_plan['phases'].append({
            'phase': 1,
            'name': 'Easy Migrations',
            'components': [comp['name'] for comp in easy_components],
            'estimated_days': len(easy_components) * 0.5,
            'description': 'Quick wins with high compatibility scores'
        })
    
    # Phase 2: Medium migrations
    medium_components = [
        comp for comp in report['components'] 
        if comp['migration_difficulty'] == 'medium' and comp['compatible']
    ]
    
    if medium_components:
        migration_plan['phases'].append({
            'phase': 2,
            'name': 'Medium Migrations',
            'components': [comp['name'] for comp in medium_components],
            'estimated_days': len(medium_components) * 1.0,
            'description': 'Moderate effort with good compatibility'
        })
    
    # Phase 3: Hard migrations
    hard_components = [
        comp for comp in report['components'] 
        if comp['migration_difficulty'] == 'hard' and comp['compatible']
    ]
    
    if hard_components:
        migration_plan['phases'].append({
            'phase': 3,
            'name': 'Hard Migrations',
            'components': [comp['name'] for comp in hard_components],
            'estimated_days': len(hard_components) * 2.0,
            'description': 'Significant effort with moderate compatibility'
        })
    
    # Phase 4: Very hard migrations
    very_hard_components = [
        comp for comp in report['components'] 
        if comp['migration_difficulty'] == 'very_hard' and comp['compatible']
    ]
    
    if very_hard_components:
        migration_plan['phases'].append({
            'phase': 4,
            'name': 'Very Hard Migrations',
            'components': [comp['name'] for comp in very_hard_components],
            'estimated_days': len(very_hard_components) * 3.0,
            'description': 'Major refactoring required'
        })
    
    # Calculate total effort
    total_days = sum(phase['estimated_days'] for phase in migration_plan['phases'])
    
    print(f"📅 MIGRATION PLAN")
    print("=" * 60)
    print(f"Total Estimated Effort: {total_days:.1f} days")
    print(f"Number of Phases: {len(migration_plan['phases'])}")
    
    for phase in migration_plan['phases']:
        print(f"\nPhase {phase['phase']}: {phase['name']}")
        print(f"  Components: {len(phase['components'])}")
        print(f"  Estimated Days: {phase['estimated_days']}")
        print(f"  Description: {phase['description']}")
        for comp in phase['components']:
            print(f"    • {comp}")
    
    return migration_plan


def main():
    """Main analysis function."""
    print("🚀 Models Training Migration Analysis")
    print("=" * 60)
    
    try:
        # Analyze components
        report = analyze_models_training_components()
        
        # Create migration plan
        migration_plan = create_migration_plan(report)
        
        # Save results
        import json
        with open('migration_analysis_report.json', 'w') as f:
            json.dump({
                'report': report,
                'migration_plan': migration_plan
            }, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: migration_analysis_report.json")
        
        return report, migration_plan
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    main()