"""
PyOD-based Anomaly Detection for Public Procurement.

Unified interface for multiple anomaly detection algorithms using PyOD library.
Supported algorithms: IForest, LOF, KNN, HBOS, ECOD, COPOD, OCSVM.

Author: Roman Kachmar
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from typing import Optional, List, Dict, Literal

# Features to log-transform (monetary and count variables with skewed distributions)
LOG_TRANSFORM_FEATURES = [
    "total_value", "tender_value", "award_value", "avg_value", "avg_tender_value",
    "avg_award_value", "total_savings", "median_value",
    "total_awards", "total_tenders", "contracts_count", "buyer_count",
]

# PyOD imports
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.ocsvm import OCSVM


# Algorithm configurations
ALGORITHMS = {
    "iforest": {
        "class": IForest,
        "params": {"n_estimators": 100, "behaviour": "new", "n_jobs": -1},
        "description": "Isolation Forest - isolates anomalies using random trees",
    },
    "lof": {
        "class": LOF,
        "params": {"n_neighbors": 20, "n_jobs": -1},
        "description": "Local Outlier Factor - density-based local anomalies",
    },
    "knn": {
        "class": KNN,
        "params": {"n_neighbors": 5, "n_jobs": -1},
        "description": "K-Nearest Neighbors - distance-based anomalies",
    },
    "hbos": {
        "class": HBOS,
        "params": {"n_bins": 10},
        "description": "Histogram-based Outlier Score - very fast, assumption of independence",
    },
    "ecod": {
        "class": ECOD,
        "params": {"n_jobs": -1},
        "description": "Empirical Cumulative Distribution - unsupervised, parameter-free",
    },
    "copod": {
        "class": COPOD,
        "params": {"n_jobs": -1},
        "description": "Copula-based Outlier Detection - fast, parameter-free",
    },
    "ocsvm": {
        "class": OCSVM,
        "params": {"kernel": "rbf"},
        "description": "One-Class SVM - boundary-based detection",
    },
}


# Default features
DEFAULT_FEATURES = {
    "tender": [
        "tender_value",
        "price_change_pct",
        "number_of_tenderers",
        "is_single_bidder",
        "is_competitive",
        "is_weekend",
        "is_q4",
        "is_december",
    ],
    "buyer": [
        "single_bidder_rate",
        "competitive_rate",
        "avg_discount_pct",
        "supplier_diversity_index",
    ],
    "supplier": [
        "total_awards",
        "total_value",
    ],
}


class PyODDetector:
    """
    Unified anomaly detector using PyOD library.

    Supports multiple algorithms with the same interface:
    - iforest: Isolation Forest
    - lof: Local Outlier Factor
    - knn: K-Nearest Neighbors
    - hbos: Histogram-based Outlier Score (fastest)
    - ecod: Empirical Cumulative Distribution
    - copod: Copula-based Outlier Detection
    - ocsvm: One-Class SVM

    Usage:
        detector = PyODDetector(algorithm="iforest", contamination=0.05)
        results = detector.fit_detect(tenders, buyers_df=buyers)
        print(detector.summary())

        # Compare multiple algorithms
        for algo in ["iforest", "hbos", "ecod"]:
            det = PyODDetector(algorithm=algo)
            results = det.fit_detect(tenders)
            print(f"{algo}: {results['anomaly'].sum()} anomalies")
    """

    def __init__(
        self,
        algorithm: Literal["iforest", "lof", "knn", "hbos", "ecod", "copod", "ocsvm"] = "iforest",
        contamination: float = 0.05,
        features: Optional[Dict[str, List[str]]] = None,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize PyOD-based detector.

        Args:
            algorithm: Algorithm to use (iforest, lof, knn, hbos, ecod, copod, ocsvm)
            contamination: Expected proportion of anomalies (0.01-0.5)
            features: Dict with "tender", "buyer", "supplier" feature lists
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters passed to the algorithm
        """
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: {list(ALGORITHMS.keys())}")

        self.algorithm = algorithm
        self.contamination = contamination
        self.features = features or DEFAULT_FEATURES
        self.random_state = random_state
        self.extra_params = kwargs

        # Will be set during fit
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names_ = None
        self.results = None

    def _create_model(self):
        """Create PyOD model instance."""
        algo_config = ALGORITHMS[self.algorithm]
        params = {**algo_config["params"], **self.extra_params}

        # Add contamination if supported
        if "contamination" in algo_config["class"].__init__.__code__.co_varnames:
            params["contamination"] = self.contamination

        # Add random_state if supported
        if "random_state" in algo_config["class"].__init__.__code__.co_varnames:
            params["random_state"] = self.random_state

        return algo_config["class"](**params)

    def fit_detect(
        self,
        tenders: pd.DataFrame,
        buyers_df: Optional[pd.DataFrame] = None,
        suppliers_df: Optional[pd.DataFrame] = None,
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fit model and detect anomalies.

        Args:
            tenders: Tenders DataFrame
            buyers_df: Optional buyers DataFrame
            suppliers_df: Optional suppliers DataFrame
            sample_size: If set, use random sample (for large datasets)

        Returns:
            DataFrame with anomaly scores and labels
        """
        print(f"PyOD Detector: {self.algorithm.upper()}")
        print(f"  {ALGORITHMS[self.algorithm]['description']}")
        print(f"Processing {len(tenders):,} tenders...")

        # Sample if needed
        if sample_size and len(tenders) > sample_size:
            print(f"  Sampling {sample_size:,} tenders...")
            tenders_work = tenders.sample(sample_size, random_state=self.random_state)
        else:
            tenders_work = tenders

        # Step 1: Prepare features
        print("Step 1/3: Preparing features...")
        X_df, feature_names = self._prepare_features(tenders_work, buyers_df, suppliers_df)
        self.feature_names_ = feature_names
        print(f"  Features: {len(feature_names)}")

        # Step 2: Preprocess
        print("Step 2/3: Preprocessing...")
        X_processed = self._preprocess(X_df)
        print(f"  Shape: {X_processed.shape}")

        # Step 3: Fit and predict
        print(f"Step 3/3: Fitting {self.algorithm.upper()}...")
        self.model = self._create_model()
        self.model.fit(X_processed)

        # Get scores and labels
        scores = self.model.decision_scores_
        labels = self.model.labels_

        # Normalize scores to 0-1
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        # Build results
        result = tenders_work[["tender_id"]].copy()
        result["score"] = scores_norm
        result["anomaly"] = labels

        # Risk levels
        result["risk_level"] = pd.cut(
            result["score"],
            bins=[0, 0.5, 0.75, 0.9, 1.01],
            labels=["low", "medium", "high", "critical"]
        )

        self.results = result

        n_anomalies = result["anomaly"].sum()
        print(f"\n{self.algorithm.upper()} complete!")
        print(f"  Anomalies: {n_anomalies:,} ({n_anomalies/len(result)*100:.2f}%)")

        return result

    def _prepare_features(
        self,
        tenders: pd.DataFrame,
        buyers_df: Optional[pd.DataFrame],
        suppliers_df: Optional[pd.DataFrame],
    ):
        """Prepare feature matrix."""
        df = tenders.copy()
        feature_cols = []

        # Tender features
        for col in self.features.get("tender", []):
            if col in df.columns:
                feature_cols.append(col)

        # Merge buyer features
        if buyers_df is not None and "buyer_id" in df.columns:
            buyer_cols = self.features.get("buyer", [])
            cols_to_merge = ["buyer_id"] + [c for c in buyer_cols if c in buyers_df.columns]
            if len(cols_to_merge) > 1:
                df = df.merge(buyers_df[cols_to_merge], on="buyer_id", how="left")
                for col in buyer_cols:
                    if col in df.columns:
                        feature_cols.append(col)

        # Merge supplier features
        if suppliers_df is not None and "supplier_id" in df.columns:
            supplier_cols = self.features.get("supplier", [])
            cols_to_merge = ["supplier_id"] + [c for c in supplier_cols if c in suppliers_df.columns]
            if len(cols_to_merge) > 1:
                df = df.merge(suppliers_df[cols_to_merge], on="supplier_id", how="left")
                for col in supplier_cols:
                    if col in df.columns:
                        feature_cols.append(col)

        feature_cols = list(dict.fromkeys(feature_cols))
        return df[feature_cols], feature_cols

    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        """Log-transform skewed features, impute, and scale."""
        X_work = X.copy()

        # Log-transform skewed features (monetary and count variables)
        for col in X_work.columns:
            if col in LOG_TRANSFORM_FEATURES:
                X_work[col] = np.log1p(X_work[col].clip(lower=0))

        self.imputer = SimpleImputer(strategy="median")
        X_imputed = self.imputer.fit_transform(X_work)

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)

        return X_scaled

    def summary(self) -> pd.DataFrame:
        """Get summary of results."""
        if self.results is None:
            raise ValueError("Run fit_detect() first")

        total = len(self.results)
        risk_counts = self.results["risk_level"].value_counts()

        summary_data = [
            {"metric": "algorithm", "value": self.algorithm},
            {"metric": "total_tenders", "value": total},
            {"metric": "anomalies", "value": self.results["anomaly"].sum()},
            {"metric": "anomaly_rate", "value": self.results["anomaly"].mean() * 100},
        ]

        for risk in ["critical", "high", "medium", "low"]:
            count = risk_counts.get(risk, 0)
            summary_data.append({"metric": f"risk_{risk}", "value": count})

        return pd.DataFrame(summary_data)

    def get_anomalies(self, min_score: float = 0.75) -> pd.DataFrame:
        """Get tenders with score above threshold."""
        if self.results is None:
            raise ValueError("Run fit_detect() first")
        return self.results[self.results["score"] >= min_score]


def compare_algorithms(
    tenders: pd.DataFrame,
    algorithms: List[str] = ["iforest", "hbos", "ecod"],
    contamination: float = 0.05,
    sample_size: Optional[int] = None,
    buyers_df: Optional[pd.DataFrame] = None,
    suppliers_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compare multiple anomaly detection algorithms.

    Args:
        tenders: Tenders DataFrame
        algorithms: List of algorithm names to compare
        contamination: Expected anomaly rate
        sample_size: Sample size (for large datasets)
        buyers_df: Optional buyers DataFrame
        suppliers_df: Optional suppliers DataFrame

    Returns:
        DataFrame with comparison results
    """
    results = []

    for algo in algorithms:
        print(f"\n{'='*60}")
        detector = PyODDetector(algorithm=algo, contamination=contamination)
        result = detector.fit_detect(
            tenders,
            buyers_df=buyers_df,
            suppliers_df=suppliers_df,
            sample_size=sample_size
        )

        results.append({
            "algorithm": algo,
            "anomalies": result["anomaly"].sum(),
            "anomaly_rate": result["anomaly"].mean() * 100,
            "mean_score": result["score"].mean(),
            "max_score": result["score"].max(),
        })

    return pd.DataFrame(results)
