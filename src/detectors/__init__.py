"""
Anomaly Detection Modules for Public Procurement.

5 core methods (ensemble):
1. RuleBasedDetector: Expert rules (45 red flags)
2. StatisticalDetector: Statistical screens (Benford, Z-score, etc.)
3. PyODDetector / AggregatedPyOD (IForest): Global anomaly detection
4. AggregatedPyOD (LOF): Local anomaly detection
5. NetworkAnalysisDetector: Graph-based cartel/collusion detection

EnsembleDetector: Combines all 5 methods with consensus voting.

Author: Roman Kachmar
"""

# Level 1: Rule-based
from .rule_based import RuleBasedDetector, RULE_DEFINITIONS

# Level 2: Statistical
from .statistical import StatisticalDetector, benford_test, hhi_index

# Level 3: ML-based (PyOD — IForest + LOF)
from .pyod_detector import PyODDetector, AggregatedPyOD, compare_algorithms

# Level 4: Network
from .network import NetworkAnalysisDetector

# Ensemble
from .ensemble import EnsembleDetector

__all__ = [
    # Level 1
    "RuleBasedDetector",
    "RULE_DEFINITIONS",
    # Level 2
    "StatisticalDetector",
    "benford_test",
    "hhi_index",
    # Level 3 (PyOD — IForest + LOF)
    "PyODDetector",       # Tender-level IForest
    "AggregatedPyOD",     # Aggregated IForest + LOF (buyer/supplier/pair)
    "compare_algorithms",
    # Level 4
    "NetworkAnalysisDetector",
    # Ensemble
    "EnsembleDetector",
]
