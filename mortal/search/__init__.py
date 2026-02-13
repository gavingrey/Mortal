# DEPRECATED: This Python search module is superseded by the Rust-side
# PolicyEvaluator + SearchIntegration pipeline (libriichi/src/search/policy.rs,
# libriichi/src/agent/mortal.rs). Use search_policy_model in MortalEngine instead.
from .config import SearchConfig
from .criticality import compute_criticality
from .engine import SearchEngine

__all__ = ['SearchConfig', 'compute_criticality', 'SearchEngine']
