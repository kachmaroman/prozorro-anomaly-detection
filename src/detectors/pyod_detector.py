"""
PyOD-based Anomaly Detection for Public Procurement.

Two core algorithms (both used in ensemble):
- Isolation Forest — global anomaly detection (isolates outliers)
- LOF — local anomaly detection (contextual outliers)

Legacy: ECOD (notebook 06 only).

- Tender-level (PyODDetector): IForest (default)
- Aggregated-level (AggregatedPyOD): IForest + LOF (both core ensemble methods)

Author: Roman Kachmar
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from typing import Optional, List, Dict, Literal

from src.config import LOG_TRANSFORM_FEATURES, DEFAULT_ML_FEATURES, DEFAULT_CONTAMINATION

# PyOD imports (core methods only)
from pyod.models.iforest import IForest
from pyod.models.lof import LOF


# Algorithm configurations for TENDER-LEVEL (fast algorithms only)
# LOF is O(n²) - too slow for 13M tenders, available only in AggregatedPyOD
ALGORITHMS = {
    "iforest": {
        "class": IForest,
        "params": {"n_estimators": 100, "behaviour": "new", "n_jobs": -1},
        "description": "Isolation Forest - isolates anomalies using random trees",
    },
}


DEFAULT_FEATURES = DEFAULT_ML_FEATURES


class PyODDetector:
    """
    Unified anomaly detector using PyOD library for TENDER-LEVEL analysis.

    Uses Isolation Forest for 13M+ tenders.

    Note: LOF is O(n²) - only available at aggregated level
    via AggregatedPyOD (36K buyers instead of 13M tenders).

    Usage:
        detector = PyODDetector(algorithm="iforest", contamination=0.05)
        results = detector.fit_detect(tenders, buyers_df=buyers)
        print(detector.summary())
    """

    def __init__(
        self,
        algorithm: Literal["iforest"] = "iforest",
        contamination: float = 0.05,
        features: Optional[Dict[str, List[str]]] = None,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize PyOD-based detector.

        Args:
            algorithm: Algorithm to use (iforest)
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

        # Normalize to 0-1 with percentile clipping to prevent
        # extreme outliers from compressing all other scores to ~0
        lo = np.percentile(scores, 1)
        hi = np.percentile(scores, 99)
        scores_clipped = np.clip(scores, lo, hi)
        scores_norm = (scores_clipped - lo) / (hi - lo + 1e-10)

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

    def feature_importances(self) -> Optional[Dict[str, float]]:
        """
        Get feature importances for IForest model.

        Computed as mean of individual tree feature importances.
        Computed as mean of individual tree feature importances.

        Returns:
            Dict of {feature_name: importance} sorted by importance descending,
            or None if model not fitted.
        """
        if self.algorithm != "iforest" or self.model is None:
            return None

        # IsolationForest doesn't expose feature_importances_ directly,
        # but individual trees (ExtraTreeRegressor) do
        trees = self.model.detector_.estimators_
        importances = np.mean([t.feature_importances_ for t in trees], axis=0)
        result = dict(zip(self.feature_names_, importances))
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def _prepare_features(
        self,
        tenders: pd.DataFrame,
        buyers_df: Optional[pd.DataFrame],
        suppliers_df: Optional[pd.DataFrame],
    ):
        """Prepare feature matrix."""
        df = tenders.copy()
        feature_cols = []

        # Compute value_vs_cpv_median if requested
        if "value_vs_cpv_median" in self.features.get("tender", []):
            cpv_col = "main_cpv_2_digit" if "main_cpv_2_digit" in df.columns else "main_cpv_code"
            cpv_median = df.groupby(cpv_col)["award_value"].transform("median")
            df["value_vs_cpv_median"] = df["award_value"].fillna(0) / cpv_median.clip(lower=1)

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



# Algorithms available for AGGREGATED level (includes O(n²) algorithms)
# All tender-level algorithms + LOF (too slow for 13M tenders)
AGGREGATED_ALGORITHMS = {
    **ALGORITHMS,
    "lof": {
        "class": LOF,
        "params": {"n_neighbors": 20, "n_jobs": -1},
        "description": "Local Outlier Factor - density-based local anomalies",
    },
}


class AggregatedPyOD:
    """
    PyOD-based anomaly detection at aggregated levels: Buyer, Supplier, Pair.

    Works with any PyOD algorithm including LOF (which is too slow for tender-level).
    More effective for detecting:
    - Anomalous buyers with unusual procurement patterns
    - Suspicious suppliers with abnormal winning patterns
    - Collusive buyer-supplier relationships

    Usage:
        detector = AggregatedPyOD(algorithm="lof", contamination=0.05)

        # Three levels of analysis
        buyer_results = detector.detect_buyers(tenders, buyers_df)
        supplier_results = detector.detect_suppliers(tenders)
        pair_results = detector.detect_pairs(tenders, min_contracts=3)

        # Get anomalies
        suspicious_buyers = detector.get_anomalies("buyers", min_score=0.5)
    """

    def __init__(
        self,
        algorithm: Literal["iforest", "lof"] = "lof",
        contamination: float = 0.05,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize aggregated PyOD detector.

        Args:
            algorithm: Algorithm to use (lof recommended for aggregated data)
            contamination: Expected proportion of anomalies (0.01-0.5)
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters passed to the algorithm
        """
        if algorithm not in AGGREGATED_ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: {list(AGGREGATED_ALGORITHMS.keys())}")

        self.algorithm = algorithm
        self.contamination = contamination
        self.random_state = random_state
        self.extra_params = kwargs

        self.buyer_results_ = None
        self.supplier_results_ = None
        self.pair_results_ = None
        self.buyer_model_ = None
        self.supplier_model_ = None
        self.pair_model_ = None
        self.buyer_features_ = None
        self.supplier_features_ = None
        self.pair_features_ = None

    def _create_model(self):
        """Create PyOD model instance."""
        algo_config = AGGREGATED_ALGORITHMS[self.algorithm]
        params = {**algo_config["params"], **self.extra_params}

        if "contamination" in algo_config["class"].__init__.__code__.co_varnames:
            params["contamination"] = self.contamination
        if "random_state" in algo_config["class"].__init__.__code__.co_varnames:
            params["random_state"] = self.random_state

        return algo_config["class"](**params)

    def _preprocess(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Log-transform skewed features, impute, and scale."""
        X = df[feature_cols].copy()

        for col in X.columns:
            if col in LOG_TRANSFORM_FEATURES:
                X[col] = np.log1p(X[col].clip(lower=0))

        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        return X_scaled

    def _fit_and_score(self, X: np.ndarray) -> tuple:
        """Fit model and return normalized scores, labels, and fitted model."""
        model = self._create_model()
        model.fit(X)

        scores = model.decision_scores_
        labels = model.labels_

        # Normalize to 0-1 with percentile clipping to prevent
        # extreme outliers from compressing all other scores to ~0
        lo = np.percentile(scores, 1)
        hi = np.percentile(scores, 99)
        scores_clipped = np.clip(scores, lo, hi)
        scores_norm = (scores_clipped - lo) / (hi - lo + 1e-10)

        return scores_norm, labels, model

    def detect_buyers(
        self,
        tenders: pd.DataFrame,
        buyers_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Detect anomalous buyers based on their procurement patterns.

        Features used:
        - single_bidder_rate, competitive_rate
        - avg_discount_pct, supplier_diversity_index
        - total_tenders, avg_value, total_value

        Returns:
            DataFrame with buyer_id, score, anomaly, risk_level
        """
        from ..data_loader import aggregate_by_buyer

        print(f"AggregatedPyOD ({self.algorithm.upper()}): Detecting anomalous BUYERS...")

        # Get or compute buyer features
        if buyers_df is not None and all(col in buyers_df.columns for col in
            ["single_bidder_rate", "competitive_rate", "supplier_diversity_index"]):
            buyer_agg = buyers_df.copy()
            print(f"  Using pre-computed buyer features")
        else:
            print(f"  Computing buyer features from tenders...")
            buyer_agg = aggregate_by_buyer(tenders, return_polars=False)

        # Select features
        feature_cols = []
        for col in ["single_bidder_rate", "competitive_rate", "avg_discount_pct",
                    "supplier_diversity_index", "total_tenders", "avg_value", "total_value",
                    "cpv_concentration", "avg_award_days", "weekend_rate",
                    "value_variance_coeff", "q4_rate"]:
            if col in buyer_agg.columns:
                feature_cols.append(col)

        print(f"  Features: {feature_cols}")
        print(f"  Buyers: {len(buyer_agg):,}")

        # Fit and detect
        X = self._preprocess(buyer_agg, feature_cols)
        scores, labels, model = self._fit_and_score(X)
        self.buyer_model_ = model
        self.buyer_features_ = feature_cols

        # Build results
        result = buyer_agg[["buyer_id"]].copy()
        result["score"] = scores
        result["anomaly"] = labels
        result["risk_level"] = pd.cut(
            scores, bins=[0, 0.5, 0.75, 0.9, 1.01],
            labels=["low", "medium", "high", "critical"]
        )

        # Add features for analysis
        for col in feature_cols:
            result[col] = buyer_agg[col].values

        self.buyer_results_ = result

        n_anomalies = result["anomaly"].sum()
        print(f"  Anomalies: {n_anomalies:,} ({n_anomalies/len(result)*100:.1f}%)")

        return result

    def detect_suppliers(
        self,
        tenders: pd.DataFrame,
        suppliers_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Detect anomalous suppliers based on their winning patterns.

        Features used:
        - total_awards, total_value, avg_award_value
        - buyer_count, single_bidder_rate, avg_competitors

        Returns:
            DataFrame with supplier_id, score, anomaly, risk_level
        """
        from ..data_loader import aggregate_by_supplier

        print(f"AggregatedPyOD ({self.algorithm.upper()}): Detecting anomalous SUPPLIERS...")

        print(f"  Computing supplier features from tenders...")
        supplier_agg = aggregate_by_supplier(tenders, return_polars=False)

        # Select features
        feature_cols = []
        for col in ["total_awards", "total_value", "avg_award_value",
                    "buyer_count", "single_bidder_rate", "avg_competitors",
                    "cpv_diversity"]:
            if col in supplier_agg.columns:
                feature_cols.append(col)

        print(f"  Features: {feature_cols}")
        print(f"  Suppliers: {len(supplier_agg):,}")

        # Fit and detect
        X = self._preprocess(supplier_agg, feature_cols)
        scores, labels, model = self._fit_and_score(X)
        self.supplier_model_ = model
        self.supplier_features_ = feature_cols

        # Build results
        result = supplier_agg[["supplier_id"]].copy()
        result["score"] = scores
        result["anomaly"] = labels
        result["risk_level"] = pd.cut(
            scores, bins=[0, 0.5, 0.75, 0.9, 1.01],
            labels=["low", "medium", "high", "critical"]
        )

        for col in feature_cols:
            result[col] = supplier_agg[col].values

        self.supplier_results_ = result

        n_anomalies = result["anomaly"].sum()
        print(f"  Anomalies: {n_anomalies:,} ({n_anomalies/len(result)*100:.1f}%)")

        return result

    def detect_pairs(
        self,
        tenders: pd.DataFrame,
        min_contracts: int = 3,
    ) -> pd.DataFrame:
        """
        Detect anomalous buyer-supplier pairs.

        Features used:
        - contracts_count, total_value, avg_value
        - single_bidder_rate
        - exclusivity_buyer, exclusivity_supplier

        Returns:
            DataFrame with buyer_id, supplier_id, score, anomaly, risk_level
        """
        from ..data_loader import aggregate_by_pair

        print(f"AggregatedPyOD ({self.algorithm.upper()}): Detecting anomalous PAIRS...")

        print(f"  Computing pair features from tenders...")
        pair_agg = aggregate_by_pair(tenders, min_contracts=min_contracts, return_polars=False)
        print(f"  Pairs with {min_contracts}+ contracts: {len(pair_agg):,}")

        if len(pair_agg) < 10:
            print(f"  Not enough pairs for detection. Returning empty.")
            return pd.DataFrame()

        # Select features
        feature_cols = []
        for col in ["contracts_count", "total_value", "avg_value",
                    "single_bidder_rate", "exclusivity_buyer", "exclusivity_supplier",
                    "temporal_concentration"]:
            if col in pair_agg.columns:
                feature_cols.append(col)

        print(f"  Features: {feature_cols}")

        # Fit and detect
        X = self._preprocess(pair_agg, feature_cols)
        scores, labels, model = self._fit_and_score(X)
        self.pair_model_ = model
        self.pair_features_ = feature_cols

        # Build results
        result = pair_agg[["buyer_id", "supplier_id"]].copy()
        result["score"] = scores
        result["anomaly"] = labels
        result["risk_level"] = pd.cut(
            scores, bins=[0, 0.5, 0.75, 0.9, 1.01],
            labels=["low", "medium", "high", "critical"]
        )

        for col in feature_cols:
            result[col] = pair_agg[col].values

        self.pair_results_ = result

        n_anomalies = result["anomaly"].sum()
        print(f"  Anomalies: {n_anomalies:,} ({n_anomalies/len(result)*100:.1f}%)")

        return result

    def get_anomalies(
        self,
        level: Literal["buyers", "suppliers", "pairs"],
        min_score: float = 0.5,
    ) -> pd.DataFrame:
        """Get entities with score above threshold."""
        results_map = {
            "buyers": self.buyer_results_,
            "suppliers": self.supplier_results_,
            "pairs": self.pair_results_,
        }

        results = results_map.get(level)
        if results is None:
            raise ValueError(f"Run detect_{level}() first")

        return results[results["score"] >= min_score]

    def summary(self) -> Dict[str, pd.DataFrame]:
        """Get summary of all detection results."""
        summaries = {}

        for name, results in [
            ("buyers", self.buyer_results_),
            ("suppliers", self.supplier_results_),
            ("pairs", self.pair_results_),
        ]:
            if results is not None:
                summaries[name] = pd.DataFrame([
                    {"metric": "algorithm", "value": self.algorithm},
                    {"metric": "total", "value": len(results)},
                    {"metric": "anomalies", "value": results["anomaly"].sum()},
                    {"metric": "anomaly_pct", "value": results["anomaly"].mean() * 100},
                    {"metric": "critical", "value": (results["risk_level"] == "critical").sum()},
                    {"metric": "high", "value": (results["risk_level"] == "high").sum()},
                ])

        return summaries

    def feature_importances(
        self,
        level: str = "buyers",
    ) -> Optional[Dict[str, float]]:
        """
        Get feature importances for IForest models.

        Only available for algorithm='iforest'. Returns None for LOF.

        Args:
            level: Which level's model to use ('buyers', 'suppliers', 'pairs')

        Returns:
            Dict of {feature_name: importance} sorted by importance descending,
            or None if not available.
        """
        if self.algorithm != "iforest":
            return None

        model_map = {
            "buyers": (self.buyer_model_, self.buyer_features_),
            "suppliers": (self.supplier_model_, self.supplier_features_),
            "pairs": (self.pair_model_, self.pair_features_),
        }

        model, features = model_map.get(level, (None, None))
        if model is None or features is None:
            return None

        # IsolationForest doesn't expose feature_importances_ directly,
        # but individual trees (ExtraTreeRegressor) do
        trees = model.detector_.estimators_
        importances = np.mean([t.feature_importances_ for t in trees], axis=0)
        result = dict(zip(features, importances))
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
