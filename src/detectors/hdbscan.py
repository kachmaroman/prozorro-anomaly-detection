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
import polars as pl
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field

from ..data_loader import aggregate_by_buyer, aggregate_by_supplier, aggregate_by_pair

# Features to log-transform (monetary and count variables with skewed distributions)
LOG_TRANSFORM_FEATURES = [
    "total_value", "tender_value", "award_value", "avg_value", "avg_tender_value",
    "avg_award_value", "total_savings", "median_value",
    "total_awards", "total_tenders", "contracts_count", "buyer_count",
]

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


class AggregatedHDBSCAN:
    """
    HDBSCAN clustering at aggregated levels: Buyer, Supplier, Buyer-Supplier.

    More effective for detecting:
    - Groups of suspicious buyers with similar patterns
    - Supplier cartels (groups working together)
    - Collusive buyer-supplier relationships

    Usage:
        detector = AggregatedHDBSCAN()

        # Cluster buyers
        buyer_results = detector.cluster_buyers(tenders, buyers_df)

        # Cluster suppliers
        supplier_results = detector.cluster_suppliers(tenders, suppliers_df)

        # Cluster buyer-supplier pairs
        pair_results = detector.cluster_pairs(tenders)
    """

    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: int = 5,
        metric: str = "euclidean",
    ):
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan package not installed. Run: pip install hdbscan")

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric

        self.buyer_results_ = None
        self.supplier_results_ = None
        self.pair_results_ = None

    def _fit_hdbscan(self, X: np.ndarray) -> tuple:
        """Fit HDBSCAN and return labels, probabilities, scores."""
        model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            core_dist_n_jobs=-1,
        )
        model.fit(X)

        labels = model.labels_
        probabilities = model.probabilities_
        outlier_scores = 1 - probabilities

        return labels, probabilities, outlier_scores

    def _preprocess(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Log-transform skewed features, impute, and scale."""
        X = df[feature_cols].copy()

        # Log-transform skewed features (monetary and count variables)
        for col in X.columns:
            if col in LOG_TRANSFORM_FEATURES:
                X[col] = np.log1p(X[col].clip(lower=0))

        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        return X_scaled

    def cluster_buyers(
        self,
        tenders: Union[pd.DataFrame, pl.DataFrame],
        buyers_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Cluster buyers based on their procurement behavior patterns.

        Features used:
        - single_bidder_rate: Rate of single-bidder tenders
        - competitive_rate: Rate of competitive tenders
        - avg_discount_pct: Average price reduction
        - supplier_diversity_index: How diverse are their suppliers
        - total_tenders: Volume of activity
        - avg_tender_value: Average tender size

        Returns:
            DataFrame with buyer_id, cluster, score, is_anomaly
        """
        print("Clustering BUYERS...")

        # Aggregate if buyers_df not provided or missing features
        if buyers_df is not None and all(col in buyers_df.columns for col in
            ["single_bidder_rate", "competitive_rate", "supplier_diversity_index"]):
            buyer_agg = buyers_df.copy()
            print(f"  Using pre-computed buyer features")
        else:
            print(f"  Computing buyer features from tenders (Polars)...")
            buyer_agg = aggregate_by_buyer(tenders, return_polars=False)

        # Select features
        feature_cols = []
        for col in ["single_bidder_rate", "competitive_rate", "avg_discount_pct",
                    "supplier_diversity_index", "total_tenders", "avg_tender_value", "total_value"]:
            if col in buyer_agg.columns:
                feature_cols.append(col)

        print(f"  Features: {feature_cols}")
        print(f"  Buyers: {len(buyer_agg):,}")

        # Preprocess and cluster
        X = self._preprocess(buyer_agg, feature_cols)
        labels, probs, scores = self._fit_hdbscan(X)

        # Build results
        result = buyer_agg[["buyer_id"]].copy()
        result["cluster"] = labels
        result["probability"] = probs
        result["outlier_score"] = scores
        result["is_noise"] = (labels == -1).astype(int)
        result["is_anomaly"] = (scores >= 0.5).astype(int)  # High outlier score

        # Add features for analysis
        for col in feature_cols:
            result[col] = buyer_agg[col].values

        self.buyer_results_ = result

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        n_anomaly = result["is_anomaly"].sum()

        print(f"  Clusters: {n_clusters}")
        print(f"  Noise (outliers): {n_noise:,} ({n_noise/len(result)*100:.1f}%)")
        print(f"  Anomalies (score>=0.5): {n_anomaly:,} ({n_anomaly/len(result)*100:.1f}%)")

        return result

    def cluster_suppliers(
        self,
        tenders: Union[pd.DataFrame, pl.DataFrame],
        suppliers_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Cluster suppliers based on their winning patterns.

        Features used:
        - win_rate: How often they win when bidding
        - total_awards: Number of contracts won
        - total_value: Total value of contracts
        - buyer_count: Number of unique buyers
        - avg_competitors: Average number of competitors
        - single_bidder_rate: Rate of winning without competition

        Returns:
            DataFrame with supplier_id, cluster, score, is_anomaly
        """
        print("Clustering SUPPLIERS...")

        # Aggregate from tenders using Polars
        print(f"  Computing supplier features from tenders (Polars)...")
        supplier_agg = aggregate_by_supplier(tenders, return_polars=False)

        # Select features
        feature_cols = []
        for col in ["total_awards", "total_value", "avg_award_value",
                    "buyer_count", "single_bidder_rate", "avg_competitors"]:
            if col in supplier_agg.columns:
                feature_cols.append(col)

        print(f"  Features: {feature_cols}")
        print(f"  Suppliers: {len(supplier_agg):,}")

        # Preprocess and cluster
        X = self._preprocess(supplier_agg, feature_cols)
        labels, probs, scores = self._fit_hdbscan(X)

        # Build results
        result = supplier_agg[["supplier_id"]].copy()
        result["cluster"] = labels
        result["probability"] = probs
        result["outlier_score"] = scores
        result["is_noise"] = (labels == -1).astype(int)
        result["is_anomaly"] = (scores >= 0.5).astype(int)

        for col in feature_cols:
            result[col] = supplier_agg[col].values

        self.supplier_results_ = result

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        n_anomaly = result["is_anomaly"].sum()

        print(f"  Clusters: {n_clusters}")
        print(f"  Noise (outliers): {n_noise:,} ({n_noise/len(result)*100:.1f}%)")
        print(f"  Anomalies (score>=0.5): {n_anomaly:,} ({n_anomaly/len(result)*100:.1f}%)")

        return result

    def cluster_pairs(
        self,
        tenders: Union[pd.DataFrame, pl.DataFrame],
        min_contracts: int = 3,
    ) -> pd.DataFrame:
        """
        Cluster buyer-supplier pairs based on their relationship patterns.

        Features used:
        - contracts_count: Number of contracts between pair
        - total_value: Total value of contracts
        - avg_value: Average contract value
        - single_bidder_rate: Rate of non-competitive awards
        - exclusivity_buyer: % of buyer's contracts with this supplier
        - exclusivity_supplier: % of supplier's contracts with this buyer

        Returns:
            DataFrame with buyer_id, supplier_id, cluster, score, is_anomaly
        """
        print("Clustering BUYER-SUPPLIER PAIRS...")

        # Aggregate by pair using Polars
        print(f"  Computing pair features from tenders (Polars)...")
        pair_agg = aggregate_by_pair(tenders, min_contracts=min_contracts, return_polars=False)
        print(f"  Pairs with {min_contracts}+ contracts: {len(pair_agg):,}")

        if len(pair_agg) < self.min_cluster_size * 2:
            print(f"  Not enough pairs for clustering. Returning empty.")
            return pd.DataFrame()

        # Select features
        feature_cols = [
            "contracts_count", "total_value", "avg_value",
            "single_bidder_rate", "exclusivity_buyer", "exclusivity_supplier"
        ]

        print(f"  Features: {feature_cols}")

        # Preprocess and cluster
        X = self._preprocess(pair_agg, feature_cols)
        labels, probs, scores = self._fit_hdbscan(X)

        # Build results
        result = pair_agg[["buyer_id", "supplier_id"]].copy()
        result["cluster"] = labels
        result["probability"] = probs
        result["outlier_score"] = scores
        result["is_noise"] = (labels == -1).astype(int)
        result["is_anomaly"] = (scores >= 0.5).astype(int)

        for col in feature_cols:
            result[col] = pair_agg[col].values

        self.pair_results_ = result

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        n_anomaly = result["is_anomaly"].sum()

        print(f"  Clusters: {n_clusters}")
        print(f"  Noise (outliers): {n_noise:,} ({n_noise/len(result)*100:.1f}%)")
        print(f"  Anomalies (score>=0.5): {n_anomaly:,} ({n_anomaly/len(result)*100:.1f}%)")

        return result

    def get_suspicious_buyers(self, min_score: float = 0.5) -> pd.DataFrame:
        """Get buyers with high outlier scores."""
        if self.buyer_results_ is None:
            raise ValueError("Run cluster_buyers() first")
        return self.buyer_results_[self.buyer_results_["outlier_score"] >= min_score]

    def get_suspicious_suppliers(self, min_score: float = 0.5) -> pd.DataFrame:
        """Get suppliers with high outlier scores."""
        if self.supplier_results_ is None:
            raise ValueError("Run cluster_suppliers() first")
        return self.supplier_results_[self.supplier_results_["outlier_score"] >= min_score]

    def get_suspicious_pairs(self, min_score: float = 0.5) -> pd.DataFrame:
        """Get buyer-supplier pairs with high outlier scores."""
        if self.pair_results_ is None:
            raise ValueError("Run cluster_pairs() first")
        return self.pair_results_[self.pair_results_["outlier_score"] >= min_score]

    def summary(self) -> Dict[str, pd.DataFrame]:
        """Get summary of all clustering results."""
        summaries = {}

        for name, results in [
            ("buyers", self.buyer_results_),
            ("suppliers", self.supplier_results_),
            ("pairs", self.pair_results_),
        ]:
            if results is not None:
                labels = results["cluster"]
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                summaries[name] = pd.DataFrame([
                    {"metric": "total", "value": len(results)},
                    {"metric": "clusters", "value": n_clusters},
                    {"metric": "noise", "value": results["is_noise"].sum()},
                    {"metric": "noise_pct", "value": results["is_noise"].mean() * 100},
                    {"metric": "anomalies", "value": results["is_anomaly"].sum()},
                    {"metric": "anomaly_pct", "value": results["is_anomaly"].mean() * 100},
                ])

        return summaries
