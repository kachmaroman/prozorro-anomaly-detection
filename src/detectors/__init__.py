"""
Anomaly Detection Modules for Public Procurement.

Available detectors:
- RuleBasedDetector: Expert rules (red flags)
- StatisticalDetector: Statistical screens (Benford, Z-score, etc.)
- IsolationForestDetector: Tree-based anomaly isolation
- HDBSCANDetector: Clustering + outlier detection
- NetworkAnalysisDetector: Graph-based cartel/collusion detection
- EnsembleDetector: Combines multiple methods

Author: Roman Kachmar
"""

# Level 1: Rule-based
from .rule_based import RuleBasedDetector, RULE_DEFINITIONS

# Level 2: Statistical
from .statistical import StatisticalDetector, benford_test, hhi_index

# Level 3: ML-based
from .isolation_forest import IsolationForestDetector
from .hdbscan import HDBSCANDetector

# Level 4: Network
from .network import NetworkAnalysisDetector

# Ensemble
from .ensemble import EnsembleDetector

# Legacy (for backward compatibility)
from .ml_based import LOFDetector, compare_detectors, ensemble_score

__all__ = [
    # Level 1
    "RuleBasedDetector",
    "RULE_DEFINITIONS",
    # Level 2
    "StatisticalDetector",
    "benford_test",
    "hhi_index",
    # Level 3
    "IsolationForestDetector",
    "HDBSCANDetector",
    "LOFDetector",
    # Level 4
    "NetworkAnalysisDetector",
    # Ensemble
    "EnsembleDetector",
    # Utils
    "compare_detectors",
    "ensemble_score",
]
