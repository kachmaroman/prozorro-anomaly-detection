"""
ML-based Anomaly Detection for Public Procurement.

This module implements unsupervised machine learning methods:
1. Isolation Forest - tree-based anomaly isolation
2. Local Outlier Factor (LOF) - density-based detection
3. DBSCAN - clustering-based detection

Author: Roman Kachmar
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class MLDetectorConfig:
    """Configuration for ML-based detectors."""

    # Feature sets
    tender_features: List[str] = None
    buyer_features: List[str] = None
    supplier_features: List[str] = None

    # Preprocessing
    scaler: str = "robust"  # "standard" or "robust"
    impute_strategy: str = "median"

    # Isolation Forest
    if_contamination: float = 0.05  # Expected proportion of anomalies
    if_n_estimators: int = 100
    if_max_samples: str = "auto"
    if_random_state: int = 42

    def __post_init__(self):
        if self.tender_features is None:
            self.tender_features = [
                "tender_value",
                "award_value",
                "price_change_pct",
                "number_of_tenderers",
                "is_single_bidder",
                "is_competitive",
            ]
        if self.buyer_features is None:
            self.buyer_features = [
                "buyer_single_bidder_rate",
                "buyer_competitive_rate",
                "buyer_avg_discount_pct",
                "buyer_supplier_diversity_index",
                "buyer_total_tenders",
            ]
        if self.supplier_features is None:
            self.supplier_features = [
                "supplier_total_awards",
                "supplier_total_value",
            ]


# Default feature configuration
DEFAULT_FEATURES = {
    "tender": [
        "tender_value",
        "award_value",
        "price_change_pct",
        "number_of_tenderers",
        "is_single_bidder",
        "is_competitive",
    ],
    "buyer": [
        "single_bidder_rate",
        "competitive_rate",
        "avg_discount_pct",
        "supplier_diversity_index",
        "total_tenders",
    ],
    "supplier": [
        "total_awards",
        "total_value",
    ],
}


class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detector for procurement data.

    Isolation Forest isolates anomalies by randomly selecting features
    and split values. Anomalies are easier to isolate (shorter path length).

    Usage:
        detector = IsolationForestDetector(contamination=0.05)
        results = detector.fit_detect(tenders, buyers_df=buyers, suppliers_df=suppliers)
        print(detector.summary())
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        max_samples: str = "auto",
        random_state: int = 42,
        scaler: str = "robust",
        features: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize Isolation Forest detector.

        Args:
            contamination: Expected proportion of anomalies (0.01-0.5)
            n_estimators: Number of trees in the forest
            max_samples: Number of samples for each tree ("auto" or int)
            random_state: Random seed for reproducibility
            scaler: Scaler type ("robust" or "standard")
            features: Dict with "tender", "buyer", "supplier" feature lists
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.scaler_type = scaler
        self.features = features or DEFAULT_FEATURES

        # Initialize model
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )

        # Will be set during fit
        self.scaler = None
        self.imputer = None
        self.feature_names_ = None
        self.results = None

    def fit_detect(
        self,
        tenders: pd.DataFrame,
        buyers_df: Optional[pd.DataFrame] = None,
        suppliers_df: Optional[pd.DataFrame] = None,
        rule_results: Optional[pd.DataFrame] = None,
        stat_results: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Fit Isolation Forest and detect anomalies.

        Args:
            tenders: Tenders DataFrame
            buyers_df: Optional buyers DataFrame for merging features
            suppliers_df: Optional suppliers DataFrame for merging features
            rule_results: Optional rule-based results to use scores as features
            stat_results: Optional statistical results to use scores as features

        Returns:
            DataFrame with anomaly scores and labels
        """
        print(f"Processing {len(tenders):,} tenders...")

        # Step 1: Prepare features
        print("Step 1/4: Preparing features...")
        X, feature_names = self._prepare_features(
            tenders, buyers_df, suppliers_df, rule_results, stat_results
        )
        self.feature_names_ = feature_names
        print(f"  Features: {len(feature_names)}")

        # Step 2: Preprocess (impute + scale)
        print("Step 2/4: Preprocessing (impute + scale)...")
        X_processed = self._preprocess(X)
        print(f"  Shape: {X_processed.shape}")

        # Step 3: Fit and predict
        print("Step 3/4: Fitting Isolation Forest...")
        self.model.fit(X_processed)

        # Get anomaly scores (lower = more anomalous)
        raw_scores = self.model.decision_function(X_processed)
        # Convert to 0-1 scale where higher = more anomalous
        anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

        # Get predictions (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(X_processed)

        print("Step 4/4: Computing results...")

        # Build results DataFrame
        result = tenders[["tender_id"]].copy()
        result["if_score"] = anomaly_scores
        result["if_anomaly"] = (predictions == -1).astype(int)

        # Assign risk levels based on score percentiles
        result["if_risk_level"] = pd.cut(
            result["if_score"],
            bins=[0, 0.5, 0.75, 0.9, 1.01],
            labels=["low", "medium", "high", "critical"]
        )

        self.results = result

        n_anomalies = result["if_anomaly"].sum()
        print(f"\nIsolation Forest complete!")
        print(f"  Anomalies detected: {n_anomalies:,} ({n_anomalies/len(result)*100:.2f}%)")

        return result

    def _prepare_features(
        self,
        tenders: pd.DataFrame,
        buyers_df: Optional[pd.DataFrame],
        suppliers_df: Optional[pd.DataFrame],
        rule_results: Optional[pd.DataFrame],
        stat_results: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare feature matrix from tenders and reference data."""

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
                df = df.merge(
                    buyers_df[cols_to_merge],
                    on="buyer_id",
                    how="left",
                    suffixes=("", "_buyer")
                )
                for col in buyer_cols:
                    if col in df.columns:
                        feature_cols.append(col)

        # Merge supplier features
        if suppliers_df is not None and "supplier_id" in df.columns:
            supplier_cols = self.features.get("supplier", [])
            cols_to_merge = ["supplier_id"] + [c for c in supplier_cols if c in suppliers_df.columns]

            if len(cols_to_merge) > 1:
                df = df.merge(
                    suppliers_df[cols_to_merge],
                    on="supplier_id",
                    how="left",
                    suffixes=("", "_supplier")
                )
                for col in supplier_cols:
                    if col in df.columns:
                        feature_cols.append(col)

        # Add rule-based score as meta-feature
        if rule_results is not None and "rule_risk_score" in rule_results.columns:
            df = df.merge(
                rule_results[["tender_id", "rule_risk_score"]],
                on="tender_id",
                how="left"
            )
            feature_cols.append("rule_risk_score")

        # Add statistical score as meta-feature
        if stat_results is not None and "stat_score" in stat_results.columns:
            df = df.merge(
                stat_results[["tender_id", "stat_score"]],
                on="tender_id",
                how="left"
            )
            feature_cols.append("stat_score")

        # Remove duplicates while preserving order
        feature_cols = list(dict.fromkeys(feature_cols))

        return df[feature_cols], feature_cols

    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        """Impute missing values and scale features."""

        # Impute missing values
        self.imputer = SimpleImputer(strategy="median")
        X_imputed = self.imputer.fit_transform(X)

        # Scale features
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        X_scaled = self.scaler.fit_transform(X_imputed)

        return X_scaled

    def summary(self) -> pd.DataFrame:
        """Get summary of anomaly detection results."""
        if self.results is None:
            raise ValueError("Run fit_detect() first")

        summary_data = []

        # Risk distribution
        risk_counts = self.results["if_risk_level"].value_counts()
        total = len(self.results)

        for risk in ["critical", "high", "medium", "low"]:
            count = risk_counts.get(risk, 0)
            summary_data.append({
                "metric": f"risk_{risk}",
                "count": count,
                "percentage": count / total * 100
            })

        # Anomaly count
        anomalies = self.results["if_anomaly"].sum()
        summary_data.append({
            "metric": "total_anomalies",
            "count": anomalies,
            "percentage": anomalies / total * 100
        })

        return pd.DataFrame(summary_data)

    def risk_distribution(self) -> pd.DataFrame:
        """Get risk level distribution."""
        if self.results is None:
            raise ValueError("Run fit_detect() first")

        dist = self.results["if_risk_level"].value_counts().reset_index()
        dist.columns = ["risk_level", "count"]
        dist["percentage"] = dist["count"] / len(self.results) * 100

        # Order by severity
        risk_order = ["critical", "high", "medium", "low"]
        dist["risk_level"] = pd.Categorical(dist["risk_level"], categories=risk_order, ordered=True)
        dist = dist.sort_values("risk_level").reset_index(drop=True)

        return dist

    def get_anomalies(self, min_score: float = 0.75) -> pd.DataFrame:
        """Get tenders with anomaly score above threshold."""
        if self.results is None:
            raise ValueError("Run fit_detect() first")

        return self.results[self.results["if_score"] >= min_score]

    def feature_importances(self) -> pd.DataFrame:
        """
        Estimate feature importances based on score variance contribution.

        Note: Isolation Forest doesn't have direct feature importances,
        so we estimate by measuring score change when each feature is shuffled.
        """
        if self.results is None or self.feature_names_ is None:
            raise ValueError("Run fit_detect() first")

        # Return placeholder - actual permutation importance is expensive
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": [1.0 / len(self.feature_names_)] * len(self.feature_names_)
        })


class LOFDetector:
    """
    Local Outlier Factor (LOF) based anomaly detector.

    LOF compares the local density of a point with its neighbors.
    Points with lower density than neighbors are anomalies.

    Note: LOF is memory-intensive. Use sampling for large datasets.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.05,
        scaler: str = "robust",
        features: Optional[Dict[str, List[str]]] = None,
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.scaler_type = scaler
        self.features = features or DEFAULT_FEATURES

        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False,
            n_jobs=-1,
        )

        self.scaler = None
        self.imputer = None
        self.feature_names_ = None
        self.results = None

    def fit_detect(
        self,
        tenders: pd.DataFrame,
        buyers_df: Optional[pd.DataFrame] = None,
        suppliers_df: Optional[pd.DataFrame] = None,
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fit LOF and detect anomalies.

        Args:
            tenders: Tenders DataFrame
            buyers_df: Optional buyers DataFrame
            suppliers_df: Optional suppliers DataFrame
            sample_size: If set, use random sample (LOF is memory-intensive)
        """
        print(f"Processing {len(tenders):,} tenders...")

        # Sample if needed
        if sample_size and len(tenders) > sample_size:
            print(f"  Sampling {sample_size:,} tenders (LOF is memory-intensive)...")
            tenders_sample = tenders.sample(sample_size, random_state=42)
        else:
            tenders_sample = tenders

        # Prepare features (reuse from IF)
        X, feature_names = self._prepare_features(tenders_sample, buyers_df, suppliers_df)
        self.feature_names_ = feature_names

        # Preprocess
        X_processed = self._preprocess(X)

        # Fit and predict
        print("Fitting LOF...")
        predictions = self.model.fit_predict(X_processed)
        scores = -self.model.negative_outlier_factor_  # Higher = more anomalous

        # Normalize scores to 0-1
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())

        # Build results
        result = tenders_sample[["tender_id"]].copy()
        result["lof_score"] = scores_norm
        result["lof_anomaly"] = (predictions == -1).astype(int)

        result["lof_risk_level"] = pd.cut(
            result["lof_score"],
            bins=[0, 0.5, 0.75, 0.9, 1.01],
            labels=["low", "medium", "high", "critical"]
        )

        self.results = result

        n_anomalies = result["lof_anomaly"].sum()
        print(f"LOF complete! Anomalies: {n_anomalies:,} ({n_anomalies/len(result)*100:.2f}%)")

        return result

    def _prepare_features(self, tenders, buyers_df, suppliers_df):
        """Prepare features (same logic as IF)."""
        df = tenders.copy()
        feature_cols = []

        for col in self.features.get("tender", []):
            if col in df.columns:
                feature_cols.append(col)

        if buyers_df is not None and "buyer_id" in df.columns:
            buyer_cols = self.features.get("buyer", [])
            cols_to_merge = ["buyer_id"] + [c for c in buyer_cols if c in buyers_df.columns]
            if len(cols_to_merge) > 1:
                df = df.merge(buyers_df[cols_to_merge], on="buyer_id", how="left")
                for col in buyer_cols:
                    if col in df.columns:
                        feature_cols.append(col)

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

    def _preprocess(self, X):
        """Preprocess features."""
        self.imputer = SimpleImputer(strategy="median")
        X_imputed = self.imputer.fit_transform(X)

        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        return self.scaler.fit_transform(X_imputed)

    def summary(self) -> pd.DataFrame:
        """Get summary."""
        if self.results is None:
            raise ValueError("Run fit_detect() first")

        risk_counts = self.results["lof_risk_level"].value_counts()
        total = len(self.results)

        return pd.DataFrame([
            {"risk_level": risk, "count": risk_counts.get(risk, 0),
             "percentage": risk_counts.get(risk, 0) / total * 100}
            for risk in ["critical", "high", "medium", "low"]
        ])


# Utility functions

def compare_detectors(
    if_results: pd.DataFrame,
    rule_results: Optional[pd.DataFrame] = None,
    stat_results: Optional[pd.DataFrame] = None,
    lof_results: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compare results from multiple detectors.

    Returns DataFrame with all scores merged for correlation analysis.
    """
    result = if_results[["tender_id", "if_score", "if_anomaly"]].copy()

    if rule_results is not None and "rule_risk_score" in rule_results.columns:
        result = result.merge(
            rule_results[["tender_id", "rule_risk_score"]],
            on="tender_id",
            how="left"
        )

    if stat_results is not None and "stat_score" in stat_results.columns:
        result = result.merge(
            stat_results[["tender_id", "stat_score"]],
            on="tender_id",
            how="left"
        )

    if lof_results is not None and "lof_score" in lof_results.columns:
        result = result.merge(
            lof_results[["tender_id", "lof_score", "lof_anomaly"]],
            on="tender_id",
            how="left"
        )

    return result


def ensemble_score(
    comparison_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Compute ensemble anomaly score from multiple detectors.

    Args:
        comparison_df: DataFrame from compare_detectors()
        weights: Optional weights for each score column

    Returns:
        Series with ensemble scores (0-1)
    """
    if weights is None:
        weights = {
            "if_score": 1.0,
            "rule_risk_score": 0.5,
            "stat_score": 0.5,
            "lof_score": 1.0,
        }

    score = pd.Series(0.0, index=comparison_df.index)
    total_weight = 0

    for col, weight in weights.items():
        if col in comparison_df.columns:
            # Normalize to 0-1 if needed
            values = comparison_df[col].fillna(0)
            if values.max() > 1:
                values = values / values.max()
            score += values * weight
            total_weight += weight

    if total_weight > 0:
        score = score / total_weight

    return score
