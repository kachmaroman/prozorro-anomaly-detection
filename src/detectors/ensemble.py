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
        self.methods_used = []
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

        Each detector uses its native column names:
        - RuleBasedDetector: rule_risk_score, rule_flags_count, rule_risk_level
        - StatisticalDetector: stat_score, stat_anomaly, stat_flags_count
        - PyODDetector (IForest): score, anomaly
        - HDBSCANDetector: hdbscan_score, hdbscan_anomaly
        - NetworkAnalysisDetector: network_score, network_anomaly

        Returns:
            DataFrame with ensemble scores and consensus
        """
        all_dfs = []
        methods_used = []

        # --- Rule-based ---
        if rule_results is not None and "tender_id" in rule_results.columns:
            df = rule_results[["tender_id"]].copy()
            # Get raw score and normalize to 0-1
            raw_score = rule_results.get("rule_risk_score", rule_results.get("rule_score", 0))
            if isinstance(raw_score, (int, float)):
                df["rule_score"] = raw_score
            else:
                df["rule_score"] = raw_score.values
            max_score = df["rule_score"].max()
            if max_score > 1:
                df["rule_score"] = df["rule_score"] / max_score
            # Anomaly = high or critical risk level (score >= 6)
            if "rule_risk_level" in rule_results.columns:
                df["rule_anomaly"] = rule_results["rule_risk_level"].isin(["high", "critical"]).astype(int).values
            elif "rule_risk_score" in rule_results.columns:
                df["rule_anomaly"] = (rule_results["rule_risk_score"].values >= 6).astype(int)
            elif "rule_flags_count" in rule_results.columns:
                df["rule_anomaly"] = (rule_results["rule_flags_count"].values >= 3).astype(int)
            else:
                df["rule_anomaly"] = (df["rule_score"] > 0.5).astype(int)
            all_dfs.append(df)
            methods_used.append("rule")
            print(f"  Rule-based: {len(df):,} tenders, {df['rule_anomaly'].sum():,} flagged")

        # --- Statistical ---
        if stat_results is not None and "tender_id" in stat_results.columns:
            df = stat_results[["tender_id"]].copy()
            raw_score = stat_results.get("stat_score", 0)
            if isinstance(raw_score, (int, float)):
                df["stat_score"] = raw_score
            else:
                df["stat_score"] = raw_score.values
            max_score = df["stat_score"].max()
            if max_score > 1:
                df["stat_score"] = df["stat_score"] / max_score
            # Use native stat_anomaly if available
            if "stat_anomaly" in stat_results.columns:
                df["stat_anomaly"] = stat_results["stat_anomaly"].values
            elif "stat_flags_count" in stat_results.columns:
                df["stat_anomaly"] = (stat_results["stat_flags_count"].values > 0).astype(int)
            else:
                df["stat_anomaly"] = (df["stat_score"] > 0.3).astype(int)
            all_dfs.append(df)
            methods_used.append("stat")
            print(f"  Statistical: {len(df):,} tenders, {df['stat_anomaly'].sum():,} flagged")

        # --- Isolation Forest ---
        if if_results is not None and "tender_id" in if_results.columns:
            df = if_results[["tender_id"]].copy()
            # PyODDetector returns 'score' and 'anomaly'
            if "if_score" in if_results.columns:
                df["if_score"] = if_results["if_score"].values
            elif "score" in if_results.columns:
                df["if_score"] = if_results["score"].values
            else:
                df["if_score"] = 0
            if "if_anomaly" in if_results.columns:
                df["if_anomaly"] = if_results["if_anomaly"].values
            elif "anomaly" in if_results.columns:
                df["if_anomaly"] = if_results["anomaly"].values
            else:
                df["if_anomaly"] = 0
            all_dfs.append(df)
            methods_used.append("if")
            print(f"  Isolation Forest: {len(df):,} tenders, {df['if_anomaly'].sum():,} flagged")

        # --- HDBSCAN ---
        if hdbscan_results is not None and "tender_id" in hdbscan_results.columns:
            df = hdbscan_results[["tender_id"]].copy()
            # HDBSCANDetector returns 'hdbscan_score' and 'hdbscan_anomaly'
            if "hdbscan_score" in hdbscan_results.columns:
                df["hdbscan_score"] = hdbscan_results["hdbscan_score"].values
            elif "score" in hdbscan_results.columns:
                df["hdbscan_score"] = hdbscan_results["score"].values
            else:
                df["hdbscan_score"] = 0
            if "hdbscan_anomaly" in hdbscan_results.columns:
                df["hdbscan_anomaly"] = hdbscan_results["hdbscan_anomaly"].values
            elif "anomaly" in hdbscan_results.columns:
                df["hdbscan_anomaly"] = hdbscan_results["anomaly"].values
            else:
                df["hdbscan_anomaly"] = 0
            all_dfs.append(df)
            methods_used.append("hdbscan")
            print(f"  HDBSCAN: {len(df):,} tenders, {df['hdbscan_anomaly'].sum():,} flagged")

        # --- Network ---
        if network_results is not None and "tender_id" in network_results.columns:
            df = network_results[["tender_id"]].copy()
            if "network_score" in network_results.columns:
                df["network_score"] = network_results["network_score"].values
                max_score = df["network_score"].max()
                if max_score > 1:
                    df["network_score"] = df["network_score"] / max_score
            else:
                df["network_score"] = 0
            if "network_anomaly" in network_results.columns:
                df["network_anomaly"] = network_results["network_anomaly"].values
            else:
                df["network_anomaly"] = 0
            all_dfs.append(df)
            methods_used.append("network")
            print(f"  Network: {len(df):,} tenders, {df['network_anomaly'].sum():,} flagged")

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
                stat["anomalies"] = int(self.results[anomaly_col].sum())
                stat["anomaly_pct"] = round(self.results[anomaly_col].mean() * 100, 2)

            if score_col in self.results.columns:
                stat["mean_score"] = round(self.results[score_col].mean(), 4)
                stat["median_score"] = round(self.results[score_col].median(), 4)

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
                "percentage": round(count / total * 100, 2)
            })

        # Consensus distribution
        for i in range(len(self.methods_used) + 1):
            count = (self.results["consensus_count"] == i).sum()
            summary_data.append({
                "metric": f"consensus_{i}_methods",
                "value": count,
                "percentage": round(count / total * 100, 2)
            })

        return pd.DataFrame(summary_data)

    def method_summary(self) -> pd.DataFrame:
        """Get summary per method."""
        if self.method_stats is None:
            raise ValueError("Run combine() first")

        return self.method_stats
