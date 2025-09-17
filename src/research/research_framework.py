"""
Research Framework for Systematic Trading Strategy Investigation

This framework provides a structured approach to conducting research across
various aspects of algorithmic trading, with proper methodology, validation,
and documentation standards.
"""

import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import json
import joblib
from enum import Enum

from ..utils.logger import system_logger
from ..utils.tprint import tprint


class ResearchPhase(Enum):
    """Research phases for systematic investigation"""
    HYPOTHESIS = "hypothesis"
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    DOCUMENTATION = "documentation"
    IMPLEMENTATION = "implementation"


@dataclass
class ResearchHypothesis:
    """Structure for research hypotheses"""
    id: str
    title: str
    description: str
    expected_outcome: str
    success_criteria: List[str]
    risk_factors: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"
    
    
@dataclass
class ResearchResult:
    """Structure for research results"""
    hypothesis_id: str
    phase: ResearchPhase
    results: Dict[str, Any]
    metrics: Dict[str, float]
    validation_results: Dict[str, Any]
    conclusions: List[str]
    next_steps: List[str]
    artifacts: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class BaseResearcher(ABC):
    """Base class for all research components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild(self.__class__.__name__)
        self.research_dir = Path(config.get('research_dir', '/workspace/research_outputs'))
        self.research_dir.mkdir(parents=True, exist_ok=True)
        
        # Research tracking
        self.hypotheses: Dict[str, ResearchHypothesis] = {}
        self.results: Dict[str, List[ResearchResult]] = {}
        
    @abstractmethod
    def generate_hypotheses(self, context: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate research hypotheses based on context"""
        pass
    
    @abstractmethod
    def collect_data(self, hypothesis: ResearchHypothesis) -> Dict[str, Any]:
        """Collect data needed for hypothesis testing"""
        pass
    
    @abstractmethod
    def analyze_data(self, hypothesis: ResearchHypothesis, data: Dict[str, Any]) -> ResearchResult:
        """Analyze data to test hypothesis"""
        pass
    
    @abstractmethod
    def validate_results(self, result: ResearchResult) -> Dict[str, Any]:
        """Validate research results"""
        pass
    
    def document_research(self, result: ResearchResult) -> str:
        """Document research findings"""
        doc_path = self.research_dir / f"research_report_{result.hypothesis_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        hypothesis = self.hypotheses[result.hypothesis_id]
        
        report = f"""# Research Report: {hypothesis.title}

## Hypothesis
**ID**: {hypothesis.id}
**Created**: {hypothesis.created_at}
**Description**: {hypothesis.description}
**Expected Outcome**: {hypothesis.expected_outcome}

### Success Criteria
{chr(10).join(f'- {criteria}' for criteria in hypothesis.success_criteria)}

### Risk Factors
{chr(10).join(f'- {risk}' for risk in hypothesis.risk_factors)}

## Results
**Phase**: {result.phase.value}
**Analysis Date**: {result.created_at}

### Key Metrics
{chr(10).join(f'- **{metric}**: {value}' for metric, value in result.metrics.items())}

### Validation Results
```json
{json.dumps(result.validation_results, indent=2)}
```

### Conclusions
{chr(10).join(f'- {conclusion}' for conclusion in result.conclusions)}

### Next Steps
{chr(10).join(f'- {step}' for step in result.next_steps)}

### Artifacts
{chr(10).join(f'- {artifact}' for artifact in result.artifacts)}

## Raw Results
```json
{json.dumps(result.results, indent=2, default=str)}
```
"""
        
        with open(doc_path, 'w') as f:
            f.write(report)
            
        tprint(f"📋 Research report saved: {doc_path}")
        return str(doc_path)
    
    def save_artifacts(self, data: Dict[str, Any], prefix: str) -> List[str]:
        """Save research artifacts"""
        artifacts = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, artifact in data.items():
            if isinstance(artifact, pd.DataFrame):
                path = self.research_dir / f"{prefix}_{name}_{timestamp}.parquet"
                artifact.to_parquet(path)
                artifacts.append(str(path))
            elif isinstance(artifact, (dict, list)):
                path = self.research_dir / f"{prefix}_{name}_{timestamp}.json"
                with open(path, 'w') as f:
                    json.dump(artifact, f, indent=2, default=str)
                artifacts.append(str(path))
            elif hasattr(artifact, 'save') or hasattr(artifact, 'dump'):
                path = self.research_dir / f"{prefix}_{name}_{timestamp}.joblib"
                joblib.dump(artifact, path)
                artifacts.append(str(path))
        
        return artifacts


class ResearchFramework:
    """Main framework for coordinating research activities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild('ResearchFramework')
        self.researchers: Dict[str, BaseResearcher] = {}
        self.active_research: Dict[str, ResearchHypothesis] = {}
        
    def register_researcher(self, name: str, researcher: BaseResearcher):
        """Register a research component"""
        self.researchers[name] = researcher
        tprint(f"🔬 Registered researcher: {name}")
    
    def conduct_research(self, 
                        researcher_name: str, 
                        context: Dict[str, Any],
                        phases: List[ResearchPhase] = None) -> List[ResearchResult]:
        """Conduct systematic research using specified researcher"""
        if researcher_name not in self.researchers:
            raise ValueError(f"Researcher {researcher_name} not registered")
        
        researcher = self.researchers[researcher_name]
        phases = phases or [ResearchPhase.HYPOTHESIS, ResearchPhase.DATA_COLLECTION, 
                          ResearchPhase.ANALYSIS, ResearchPhase.VALIDATION]
        
        results = []
        
        # Generate hypotheses
        if ResearchPhase.HYPOTHESIS in phases:
            hypotheses = researcher.generate_hypotheses(context)
            tprint(f"🧪 Generated {len(hypotheses)} hypotheses for {researcher_name}")
            
            for hypothesis in hypotheses:
                researcher.hypotheses[hypothesis.id] = hypothesis
                self.active_research[hypothesis.id] = hypothesis
        
        # Process each hypothesis through remaining phases
        for hypothesis_id, hypothesis in researcher.hypotheses.items():
            try:
                # Data collection
                if ResearchPhase.DATA_COLLECTION in phases:
                    tprint(f"📊 Collecting data for hypothesis: {hypothesis.title}")
                    data = researcher.collect_data(hypothesis)
                
                # Analysis
                if ResearchPhase.ANALYSIS in phases:
                    tprint(f"🔍 Analyzing hypothesis: {hypothesis.title}")
                    result = researcher.analyze_data(hypothesis, data)
                    results.append(result)
                
                # Validation
                if ResearchPhase.VALIDATION in phases:
                    tprint(f"✅ Validating results for: {hypothesis.title}")
                    validation = researcher.validate_results(result)
                    result.validation_results = validation
                
                # Documentation
                if ResearchPhase.DOCUMENTATION in phases:
                    tprint(f"📝 Documenting research: {hypothesis.title}")
                    researcher.document_research(result)
                
                # Store results
                if hypothesis_id not in researcher.results:
                    researcher.results[hypothesis_id] = []
                researcher.results[hypothesis_id].append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing hypothesis {hypothesis_id}: {e}")
                continue
        
        return results
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get summary of all research activities"""
        summary = {
            'total_researchers': len(self.researchers),
            'active_research': len(self.active_research),
            'researchers': {}
        }
        
        for name, researcher in self.researchers.items():
            summary['researchers'][name] = {
                'total_hypotheses': len(researcher.hypotheses),
                'total_results': sum(len(results) for results in researcher.results.values()),
                'recent_activity': max(
                    (result.created_at for results in researcher.results.values() 
                     for result in results), 
                    default=None
                )
            }
        
        return summary
    
    def export_research_data(self, output_dir: str = None) -> str:
        """Export all research data for analysis"""
        output_dir = Path(output_dir or '/workspace/research_exports')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_data = {
            'framework_config': self.config,
            'researchers': {},
            'summary': self.get_research_summary(),
            'exported_at': timestamp
        }
        
        for name, researcher in self.researchers.items():
            export_data['researchers'][name] = {
                'hypotheses': {hid: {
                    'id': h.id,
                    'title': h.title,
                    'description': h.description,
                    'expected_outcome': h.expected_outcome,
                    'success_criteria': h.success_criteria,
                    'risk_factors': h.risk_factors,
                    'created_at': h.created_at.isoformat(),
                    'status': h.status
                } for hid, h in researcher.hypotheses.items()},
                'results': {hid: [{
                    'hypothesis_id': r.hypothesis_id,
                    'phase': r.phase.value,
                    'metrics': r.metrics,
                    'conclusions': r.conclusions,
                    'next_steps': r.next_steps,
                    'artifacts': r.artifacts,
                    'created_at': r.created_at.isoformat()
                } for r in results] for hid, results in researcher.results.items()}
            }
        
        export_path = output_dir / f'research_export_{timestamp}.json'
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        tprint(f"📤 Research data exported to: {export_path}")
        return str(export_path)