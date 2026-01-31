"""
Anomaly Detection Modules for Public Procurement.

Available detectors:
- RuleBasedDetector: Expert rules (red flags)
- StatisticalDetector: Statistical screens (Benford, Z-score, etc.)
- PyODDetector: Tender-level ML (IForest, KNN, HBOS, ECOD, COPOD, OCSVM)
- AggregatedPyOD: Buyer/Supplier/Pair-level ML (includes LOF)
- HDBSCANDetector: Clustering + outlier detection
- AggregatedHDBSCAN: Clustering at aggregated levels
- AutoencoderDetector: Deep learning (reconstruction error)
- AggregatedAutoencoder: Deep learning at aggregated levels
- NetworkAnalysisDetector: Graph-based cartel/collusion detection
- EnsembleDetector: Combines multiple methods

Author: Roman Kachmar
"""

# Level 1: Rule-based
from .rule_based import RuleBasedDetector, RULE_DEFINITIONS

# Level 2: Statistical
from .statistical import StatisticalDetector, benford_test, hhi_index

# Level 3: ML-based (PyOD)
from .pyod_detector import PyODDetector, AggregatedPyOD, compare_algorithms

# Level 3: ML-based (HDBSCAN)
from .hdbscan import HDBSCANDetector, AggregatedHDBSCAN

# Level 3: ML-based (Autoencoder)
from .autoencoder import AutoencoderDetector, AggregatedAutoencoder

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
    # Level 3 (PyOD)
    "PyODDetector",
    "AggregatedPyOD",
    "compare_algorithms",
    # Level 3 (HDBSCAN)
    "HDBSCANDetector",
    "AggregatedHDBSCAN",
    # Level 3 (Autoencoder)
    "AutoencoderDetector",
    "AggregatedAutoencoder",
    # Level 4
    "NetworkAnalysisDetector",
    # Ensemble
    "EnsembleDetector",
]
