import re

# Read the file
with open('label_based_layer_2.py', 'r') as f:
    content = f.read()

# Add import for Bayesian causal discovery
import_pattern = r'(from \.causal_discovery import CausalDiscovery, quick_causal_discovery)'
import_replacement = r'''from .causal_discovery import CausalDiscovery, quick_causal_discovery
from .causal_uncertainty_quantification import BayesianCausalDiscovery, quick_bayesian_causal_discovery'''

# Apply the replacement
content = re.sub(import_pattern, import_replacement, content)

# Find the _run_causal_discovery method and enhance it
discovery_pattern = r'(    def _run_causal_discovery\(self, df: pd\.DataFrame\) -> Dict\[str, List\[str\]]:.*?return causal_graph)'

enhanced_discovery_method = '''    def _run_causal_discovery(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Run enhanced causal discovery with Bayesian uncertainty quantification.
        """
        try:
            if self.verbose:
                tprint_info("🔍 Enhanced Causal Discovery: Starting Bayesian discovery with uncertainty...")
            
            # Check if Bayesian discovery is enabled
            use_bayesian = getattr(self, 'use_bayesian_discovery', True)
            
            if use_bayesian and self.CAUSAL_MODULES_AVAILABLE:
                if self.verbose:
                    tprint_info("   📊 Using Bayesian Causal Discovery...")
                
                # Use Bayesian discovery with uncertainty
                discovery_results = quick_bayesian_causal_discovery(
                    df, n_bootstrap=50, verbose=self.verbose  # Reduced for speed
                )
                
                if 'error' in discovery_results:
                    if self.verbose:
                        tprint_warning("   ⚠️ Bayesian discovery failed, falling back to deterministic...")
                    return self._fallback_causal_discovery(df)
                
                # Extract causal graph and uncertainty metrics
                causal_graph = discovery_results.get('consensus_graph', {})
                uncertainty_metrics = discovery_results.get('uncertainty_metrics', {})
                
                # Store uncertainty metrics for reporting
                self.causal_discovery_uncertainty_ = uncertainty_metrics
                
                if self.verbose:
                    tprint_success(f"   ✅ Bayesian discovery complete:")
                    tprint_info(f"      - Graph edges: {len(causal_graph)}")
                    tprint_info(f"      - Graph stability: {uncertainty_metrics.get('graph_stability', 0):.3f}")
                    tprint_info(f"      - Avg confidence: {uncertainty_metrics.get('avg_confidence', 0):.3f}")
                
                return causal_graph
            else:
                if self.verbose:
                    tprint_info("   📊 Using deterministic Causal Discovery...")
                return self._fallback_causal_discovery(df)
                
        except Exception as e:
            if self.verbose:
                tprint_error(f"   ❌ Enhanced causal discovery failed: {e}")
            return self._fallback_causal_discovery(df)
    
    def _fallback_causal_discovery(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Fallback to deterministic causal discovery."""
        try:
            if self.verbose:
                tprint_info("   🔄 Using fallback deterministic causal discovery...")
            
            causal_discovery = CausalDiscovery(verbose=self.verbose)
            discovery_results = causal_discovery.discover_causal_structure(df)
            
            if 'error' in discovery_results:
                if self.verbose:
                    tprint_error("   ❌ Fallback discovery also failed")
                return {}
            
            causal_graph = discovery_results.get('causal_graph', {})
            
            if self.verbose:
                tprint_success(f"   ✅ Fallback discovery complete: {len(causal_graph)} edges")
            
            return causal_graph
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"   ❌ Fallback discovery failed: {e}")
            return {}'''

# Apply the replacement
content = re.sub(discovery_pattern, enhanced_discovery_method, content, flags=re.DOTALL)

# Add Bayesian discovery configuration to __init__
init_pattern = r'(        # Causal Discovery Parameters\s+self\.significance_level = kwargs\.get\("significance_level", 0\.05\)\s+self\.max_conditioning_set = kwargs\.get\("max_conditioning_set", 3)\s+self\.use_lingam = kwargs\.get\("use_lingam", True))'

init_replacement = r'''        # Causal Discovery Parameters
        self.significance_level = kwargs.get("significance_level", 0.05)
        self.max_conditioning_set = kwargs.get("max_conditioning_set", 3)
        self.use_lingam = kwargs.get("use_lingam", True)
        self.use_bayesian_discovery = kwargs.get("use_bayesian_discovery", True)
        self.bayesian_n_bootstrap = kwargs.get("bayesian_n_bootstrap", 50)'''

# Apply the replacement
content = re.sub(init_pattern, init_replacement, content)

# Write back to file
with open('label_based_layer_2.py', 'w') as f:
    f.write(content)

print("Wired Bayesian causal discovery into Layer 2 pipeline")
