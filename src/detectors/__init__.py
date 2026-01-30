"""
Anomaly Detection Modules for Public Procurement.

Available detectors:
- RuleBasedDetector: Expert rules (red flags)
- StatisticalDetector: Statistical screens (Benford, Z-score, etc.)
- PyODDetector: Unified ML detector (IForest, LOF, HBOS, ECOD, etc.)
- IsolationForestDetector: Tree-based anomaly isolation (legacy)
- HDBSCANDetector: Clustering + outlier detection (legacy)
- NetworkAnalysisDetector: Graph-based cartel/collusion detection
- EnsembleDetector: Combines multiple methods

Author: Roman Kachmar
"""

# Level 1: Rule-based
from .rule_based import RuleBasedDetector, RULE_DEFINITIONS

# Level 2: Statistical
from .statistical import StatisticalDetector, benford_test, hhi_index

# Level 3: ML-based (PyOD - recommended)
from .pyod_detector import PyODDetector, compare_algorithms

# Level 3: ML-based (legacy)
from .isolation_forest import IsolationForestDetector
from .hdbscan import HDBSCANDetector, AggregatedHDBSCAN

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
    # Level 3 (PyOD - recommended)
    "PyODDetector",
    "compare_algorithms",
    # Level 3 (legacy)
    "IsolationForestDetector",
    "HDBSCANDetector",
    "AggregatedHDBSCAN",
    "LOFDetector",
    # Level 4
    "NetworkAnalysisDetector",
    # Ensemble
    "EnsembleDetector",
    # Utils
    "compare_detectors",
    "ensemble_score",
]
