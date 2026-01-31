"""
Ensemble Anomaly Detection for Public Procurement.

This module combines multiple detection methods:
1. Rule-based (red flags)
2. Statistical screens
3. Isolation Forest
4. HDBSCAN
5. Network Analysis

Cross-method agreement provides stronger anomaly signals.

Author: Roman Kachmar
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class EnsembleConfig:
    """Configuration for ensemble detector."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        "rule": 1.0,
        "stat": 0.8,
        "if": 1.0,
        "hdbscan": 0.8,
        "network": 1.0,
    })
    consensus_threshold: int = 2  # Minimum methods to flag as anomaly


class EnsembleDetector:
    """
    Ensemble anomaly detector combining multiple methods.

    Provides:
    1. Weighted ensemble score
    2. Consensus voting (how many methods flag a tender)
    3. Risk levels based on cross-method agreement

    Usage:
        detector = EnsembleDetector()
        results = detector.combine(
            rule_results=rule_df,
            stat_results=stat_df,
            if_results=if_df,
            hdbscan_results=hdbscan_df,
            network_results=network_df,
        )
        print(detector.summary())
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        consensus_threshold: int = 2,
    ):
        """
        Initialize Ensemble detector.

        Args:
            weights: Dict with weights for each method
            consensus_threshold: Minimum methods to consider high-risk
        """
        self.weights = weights or {
            "rule": 1.0,
            "stat": 0.8,
            "if": 1.0,
            "hdbscan": 0.8,
            "network": 1.0,
        }
        self.consensus_threshold = consensus_threshold

        self.results = None
        self.method_stats = None

    def combine(
        self,
        rule_results: Optional[pd.DataFrame] = None,
        stat_results: Optional[pd.DataFrame] = None,
        if_results: Optional[pd.DataFrame] = None,
        hdbscan_results: Optional[pd.DataFrame] = None,
        network_results: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Combine results from multiple detectors.

        Args:
            rule_results: Results from RuleBasedDetector (with rule_risk_score, rule_anomaly)
            stat_results: Results from StatisticalDetector (with stat_score, stat_anomaly)
            if_results: Results from PyODDetector/IsolationForest (with score/anomaly or if_score/if_anomaly)
            hdbscan_results: Results from HDBSCANDetector (with hdbscan_score, hdbscan_anomaly)
            network_results: Results from NetworkAnalysisDetector (with network_score, network_anomaly)

        Returns:
            DataFrame with ensemble scores and consensus
        """
        # Collect all results
        all_dfs = []
        methods_used = []

        if rule_results is not None and "tender_id" in rule_results.columns:
            df = rule_results[["tender_id"]].copy()
            df["rule_score"] = rule_results.get("rule_risk_score", rule_results.get("rule_score", 0))
            # Normalize to 0-1 if needed
            if df["rule_score"].max() > 1:
                df["rule_score"] = df["rule_score"] / df["rule_score"].max()
            df["rule_anomaly"] = rule_results.get("rule_anomaly", (df["rule_score"] > 0.5).astype(int))
            all_dfs.append(df)
            methods_used.append("rule")
            print(f"  Rule-based: {len(df):,} tenders")

        if stat_results is not None and "tender_id" in stat_results.columns:
            df = stat_results[["tender_id"]].copy()
            df["stat_score"] = stat_results.get("stat_score", 0)
            if df["stat_score"].max() > 1:
                df["stat_score"] = df["stat_score"] / df["stat_score"].max()
            df["stat_anomaly"] = stat_results.get("stat_anomaly", (df["stat_score"] > 0.5).astype(int))
            all_dfs.append(df)
            methods_used.append("stat")
            print(f"  Statistical: {len(df):,} tenders")

        if if_results is not None and "tender_id" in if_results.columns:
            df = if_results[["tender_id"]].copy()
            # Support both legacy (if_score) and PyOD (score) column names
            df["if_score"] = if_results.get("if_score", if_results.get("score", 0))
            df["if_anomaly"] = if_results.get("if_anomaly", if_results.get("anomaly", 0))
            all_dfs.append(df)
            methods_used.append("if")
            print(f"  Isolation Forest: {len(df):,} tenders")

        if hdbscan_results is not None and "tender_id" in hdbscan_results.columns:
            df = hdbscan_results[["tender_id"]].copy()
            df["hdbscan_score"] = hdbscan_results.get("hdbscan_score", 0)
            df["hdbscan_anomaly"] = hdbscan_results.get("hdbscan_anomaly", 0)
            all_dfs.append(df)
            methods_used.append("hdbscan")
            print(f"  HDBSCAN: {len(df):,} tenders")

        if network_results is not None and "tender_id" in network_results.columns:
            df = network_results[["tender_id"]].copy()
            df["network_score"] = network_results.get("network_score", 0)
            df["network_anomaly"] = network_results.get("network_anomaly", 0)
            all_dfs.append(df)
            methods_used.append("network")
            print(f"  Network: {len(df):,} tenders")

        if len(all_dfs) == 0:
            raise ValueError("No valid results provided")

        print(f"\nCombining {len(methods_used)} methods: {methods_used}")

        # Merge all results
        result = all_dfs[0]
        for df in all_dfs[1:]:
            result = result.merge(df, on="tender_id", how="outer")

        # Fill missing values
        for method in methods_used:
            score_col = f"{method}_score"
            anomaly_col = f"{method}_anomaly"
            if score_col in result.columns:
                result[score_col] = result[score_col].fillna(0)
            if anomaly_col in result.columns:
                result[anomaly_col] = result[anomaly_col].fillna(0).astype(int)

        # Compute ensemble score (weighted average)
        weighted_sum = pd.Series(0.0, index=result.index)
        total_weight = 0

        for method in methods_used:
            score_col = f"{method}_score"
            if score_col in result.columns:
                weight = self.weights.get(method, 1.0)
                weighted_sum += result[score_col] * weight
                total_weight += weight

        result["ensemble_score"] = weighted_sum / total_weight if total_weight > 0 else 0

        # Compute consensus (how many methods flag this tender)
        anomaly_cols = [f"{m}_anomaly" for m in methods_used if f"{m}_anomaly" in result.columns]
        result["consensus_count"] = result[anomaly_cols].sum(axis=1)
        result["consensus_pct"] = result["consensus_count"] / len(methods_used)

        # Ensemble anomaly (consensus >= threshold)
        result["ensemble_anomaly"] = (result["consensus_count"] >= self.consensus_threshold).astype(int)

        # Risk level based on consensus
        def get_risk_level(count, total):
            pct = count / total
            if pct >= 0.75:
                return "critical"
            elif pct >= 0.5:
                return "high"
            elif pct >= 0.25:
                return "medium"
            else:
                return "low"

        result["ensemble_risk_level"] = result.apply(
            lambda x: get_risk_level(x["consensus_count"], len(methods_used)),
            axis=1
        )

        self.results = result
        self.methods_used = methods_used
        self._compute_method_stats()

        # Summary
        n_critical = (result["ensemble_risk_level"] == "critical").sum()
        n_high = (result["ensemble_risk_level"] == "high").sum()
        print(f"\nEnsemble complete!")
        print(f"  Critical ({len(methods_used)}/{len(methods_used)} methods): {n_critical:,}")
        print(f"  High ({self.consensus_threshold}+/{len(methods_used)} methods): {n_high + n_critical:,}")

        return result

    def _compute_method_stats(self) -> None:
        """Compute statistics for each method."""
        if self.results is None:
            return

        stats = []
        for method in self.methods_used:
            anomaly_col = f"{method}_anomaly"
            score_col = f"{method}_score"

            stat = {"method": method}

            if anomaly_col in self.results.columns:
                stat["anomalies"] = self.results[anomaly_col].sum()
                stat["anomaly_pct"] = self.results[anomaly_col].mean() * 100

            if score_col in self.results.columns:
                stat["mean_score"] = self.results[score_col].mean()
                stat["median_score"] = self.results[score_col].median()

            stats.append(stat)

        self.method_stats = pd.DataFrame(stats)

    def get_consensus_anomalies(self, min_consensus: int = 2) -> pd.DataFrame:
        """Get tenders flagged by at least min_consensus methods."""
        if self.results is None:
            raise ValueError("Run combine() first")

        return self.results[self.results["consensus_count"] >= min_consensus]

    def get_critical_tenders(self) -> pd.DataFrame:
        """Get tenders flagged as critical (all methods agree)."""
        if self.results is None:
            raise ValueError("Run combine() first")

        return self.results[self.results["ensemble_risk_level"] == "critical"]

    def correlation_matrix(self) -> pd.DataFrame:
        """Compute correlation between method scores."""
        if self.results is None:
            raise ValueError("Run combine() first")

        score_cols = [f"{m}_score" for m in self.methods_used if f"{m}_score" in self.results.columns]
        return self.results[score_cols].corr()

    def agreement_matrix(self) -> pd.DataFrame:
        """Compute Jaccard similarity between method anomaly flags."""
        if self.results is None:
            raise ValueError("Run combine() first")

        anomaly_cols = [f"{m}_anomaly" for m in self.methods_used if f"{m}_anomaly" in self.results.columns]

        # Compute Jaccard similarity
        matrix = pd.DataFrame(index=anomaly_cols, columns=anomaly_cols, dtype=float)

        for col1 in anomaly_cols:
            for col2 in anomaly_cols:
                set1 = set(self.results[self.results[col1] == 1]["tender_id"])
                set2 = set(self.results[self.results[col2] == 1]["tender_id"])

                if len(set1 | set2) > 0:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                else:
                    jaccard = 0

                matrix.loc[col1, col2] = jaccard

        return matrix

    def summary(self) -> pd.DataFrame:
        """Get summary of ensemble results."""
        if self.results is None:
            raise ValueError("Run combine() first")

        total = len(self.results)

        summary_data = [
            {"metric": "total_tenders", "value": total, "percentage": 100.0},
        ]

        # Risk distribution
        for risk in ["critical", "high", "medium", "low"]:
            count = (self.results["ensemble_risk_level"] == risk).sum()
            summary_data.append({
                "metric": f"risk_{risk}",
                "value": count,
                "percentage": count / total * 100
            })

        # Consensus distribution
        for i in range(len(self.methods_used) + 1):
            count = (self.results["consensus_count"] == i).sum()
            summary_data.append({
                "metric": f"consensus_{i}_methods",
                "value": count,
                "percentage": count / total * 100
            })

        return pd.DataFrame(summary_data)

    def method_summary(self) -> pd.DataFrame:
        """Get summary per method."""
        if self.method_stats is None:
            raise ValueError("Run combine() first")

        return self.method_stats
