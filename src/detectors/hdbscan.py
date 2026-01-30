"""
HDBSCAN-based Anomaly Detection for Public Procurement.

HDBSCAN provides both clustering and outlier detection:
- Clusters = groups of similar tenders
- Noise points = outliers that don't fit any cluster
- Outlier scores = probability-based anomaly measure

Author: Roman Kachmar
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


@dataclass
class HDBSCANConfig:
    """Configuration for HDBSCAN detector."""
    min_cluster_size: int = 50
    min_samples: int = 10
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"  # "eom" or "leaf"
    contamination: float = 0.05  # For anomaly threshold


# Default features for HDBSCAN
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


class HDBSCANDetector:
    """
    HDBSCAN-based anomaly detector for procurement data.

    HDBSCAN (Hierarchical DBSCAN) provides:
    1. Clustering without specifying number of clusters
    2. Outlier scores based on cluster membership probability
    3. Noise detection (points that don't fit any cluster)

    Usage:
        detector = HDBSCANDetector(min_cluster_size=50)
        results = detector.fit_detect(tenders, buyers_df=buyers)
        print(detector.summary())
    """

    def __init__(
        self,
        min_cluster_size: int = 50,
        min_samples: int = 10,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        contamination: float = 0.05,
        features: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize HDBSCAN detector.

        Args:
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples for core point
            metric: Distance metric
            cluster_selection_method: "eom" (Excess of Mass) or "leaf"
            contamination: Expected proportion of anomalies (for threshold)
            features: Dict with "tender", "buyer", "supplier" feature lists
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan package not installed. Run: pip install hdbscan")

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.contamination = contamination
        self.features = features or DEFAULT_FEATURES

        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names_ = None
        self.results = None
        self.cluster_stats_ = None

    def fit_detect(
        self,
        tenders: pd.DataFrame,
        buyers_df: Optional[pd.DataFrame] = None,
        suppliers_df: Optional[pd.DataFrame] = None,
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fit HDBSCAN and detect anomalies.

        Args:
            tenders: Tenders DataFrame
            buyers_df: Optional buyers DataFrame
            suppliers_df: Optional suppliers DataFrame
            sample_size: If set, use random sample (for large datasets)

        Returns:
            DataFrame with cluster labels, outlier scores, and anomaly flags
        """
        print(f"Processing {len(tenders):,} tenders...")

        # Sample if needed
        if sample_size and len(tenders) > sample_size:
            print(f"  Sampling {sample_size:,} tenders...")
            tenders_work = tenders.sample(sample_size, random_state=42)
            self._sampled = True
        else:
            tenders_work = tenders
            self._sampled = False

        # Step 1: Prepare features
        print("Step 1/4: Preparing features...")
        X_df, feature_names = self._prepare_features(tenders_work, buyers_df, suppliers_df)
        self.feature_names_ = feature_names
        print(f"  Features: {len(feature_names)}")

        # Step 2: Preprocess
        print("Step 2/4: Preprocessing (impute + scale)...")
        X_processed = self._preprocess(X_df)
        print(f"  Shape: {X_processed.shape}")

        # Step 3: Fit HDBSCAN
        print("Step 3/4: Fitting HDBSCAN...")
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            core_dist_n_jobs=-1,
        )
        self.model.fit(X_processed)

        # Step 4: Compute results
        print("Step 4/4: Computing results...")

        labels = self.model.labels_
        probabilities = self.model.probabilities_

        # Outlier score = 1 - probability (higher = more anomalous)
        outlier_scores = 1 - probabilities

        # Build results DataFrame
        result = tenders_work[["tender_id"]].copy()
        result["hdbscan_cluster"] = labels
        result["hdbscan_probability"] = probabilities
        result["hdbscan_score"] = outlier_scores
        result["hdbscan_is_noise"] = (labels == -1).astype(int)

        # Define anomaly based on score threshold (top X%)
        threshold = np.percentile(outlier_scores, 100 * (1 - self.contamination))
        result["hdbscan_anomaly"] = (outlier_scores >= threshold).astype(int)

        # Risk levels
        result["hdbscan_risk_level"] = pd.cut(
            result["hdbscan_score"],
            bins=[0, 0.5, 0.75, 0.9, 1.01],
            labels=["low", "medium", "high", "critical"]
        )

        self.results = result

        # Compute cluster statistics
        self._compute_cluster_stats(tenders_work)

        # Summary
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        n_anomalies = result["hdbscan_anomaly"].sum()

        print(f"\nHDBSCAN complete!")
        print(f"  Clusters: {n_clusters}")
        print(f"  Noise points: {n_noise:,} ({n_noise/len(result)*100:.1f}%)")
        print(f"  Anomalies (top {self.contamination*100:.0f}%): {n_anomalies:,}")

        return result

    def _prepare_features(
        self,
        tenders: pd.DataFrame,
        buyers_df: Optional[pd.DataFrame],
        suppliers_df: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, List[str]]:
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
        """Impute and scale features."""
        self.imputer = SimpleImputer(strategy="median")
        X_imputed = self.imputer.fit_transform(X)

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)

        return X_scaled

    def _compute_cluster_stats(self, tenders: pd.DataFrame) -> None:
        """Compute statistics for each cluster."""
        if self.results is None:
            return

        analysis_df = tenders.merge(self.results[["tender_id", "hdbscan_cluster"]], on="tender_id")

        # Only for non-noise clusters
        clustered = analysis_df[analysis_df["hdbscan_cluster"] != -1]

        if len(clustered) == 0:
            self.cluster_stats_ = pd.DataFrame()
            return

        stats = clustered.groupby("hdbscan_cluster").agg({
            "tender_id": "count",
            "tender_value": ["mean", "median"],
            "is_single_bidder": "mean" if "is_single_bidder" in clustered.columns else "count",
            "is_competitive": "mean" if "is_competitive" in clustered.columns else "count",
        })

        stats.columns = ["count", "mean_value", "median_value", "single_bidder_rate", "competitive_rate"]
        stats = stats.sort_values("count", ascending=False)

        self.cluster_stats_ = stats

    def get_suspicious_clusters(
        self,
        min_size: int = 100,
        min_single_bidder_rate: float = 0.5,
    ) -> pd.DataFrame:
        """
        Find clusters with suspicious characteristics.

        Args:
            min_size: Minimum cluster size
            min_single_bidder_rate: Minimum single bidder rate

        Returns:
            DataFrame with suspicious clusters
        """
        if self.cluster_stats_ is None or len(self.cluster_stats_) == 0:
            return pd.DataFrame()

        suspicious = self.cluster_stats_[
            (self.cluster_stats_["count"] >= min_size) &
            (self.cluster_stats_["single_bidder_rate"] >= min_single_bidder_rate)
        ]

        return suspicious

    def summary(self) -> pd.DataFrame:
        """Get summary of results."""
        if self.results is None:
            raise ValueError("Run fit_detect() first")

        labels = self.results["hdbscan_cluster"]

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        n_anomalies = self.results["hdbscan_anomaly"].sum()
        total = len(self.results)

        summary_data = [
            {"metric": "total_tenders", "value": total},
            {"metric": "clusters", "value": n_clusters},
            {"metric": "noise_points", "value": n_noise},
            {"metric": "noise_pct", "value": n_noise / total * 100},
            {"metric": "anomalies", "value": n_anomalies},
            {"metric": "anomaly_pct", "value": n_anomalies / total * 100},
        ]

        return pd.DataFrame(summary_data)

    def get_anomalies(self, min_score: float = 0.75) -> pd.DataFrame:
        """Get tenders with outlier score above threshold."""
        if self.results is None:
            raise ValueError("Run fit_detect() first")

        return self.results[self.results["hdbscan_score"] >= min_score]
