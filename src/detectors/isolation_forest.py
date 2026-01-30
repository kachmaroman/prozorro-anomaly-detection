"""
Isolation Forest Anomaly Detection for Public Procurement.

Isolation Forest isolates anomalies by randomly selecting features
and split values. Anomalies are easier to isolate (shorter path length).

Author: Roman Kachmar
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Optional, List, Dict, Tuple


# Features to log-transform (monetary and count variables with skewed distributions)
LOG_TRANSFORM_FEATURES = [
    "total_value", "tender_value", "award_value", "avg_value", "avg_tender_value",
    "avg_award_value", "total_savings", "median_value",
    "total_awards", "total_tenders", "contracts_count", "buyer_count",
]


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
    ) -> pd.DataFrame:
        """
        Fit Isolation Forest and detect anomalies.

        Args:
            tenders: Tenders DataFrame
            buyers_df: Optional buyers DataFrame for merging features
            suppliers_df: Optional suppliers DataFrame for merging features

        Returns:
            DataFrame with anomaly scores and labels
        """
        print(f"Processing {len(tenders):,} tenders...")

        # Step 1: Prepare features
        print("Step 1/4: Preparing features...")
        X, feature_names = self._prepare_features(tenders, buyers_df, suppliers_df)
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

        # Remove duplicates while preserving order
        feature_cols = list(dict.fromkeys(feature_cols))

        return df[feature_cols], feature_cols

    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        """Log-transform skewed features, impute missing values, and scale."""
        X_work = X.copy()

        # Log-transform skewed features (monetary and count variables)
        for col in X_work.columns:
            if col in LOG_TRANSFORM_FEATURES:
                X_work[col] = np.log1p(X_work[col].clip(lower=0))  # log(1+x), clip negatives

        # Impute missing values
        self.imputer = SimpleImputer(strategy="median")
        X_imputed = self.imputer.fit_transform(X_work)

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
