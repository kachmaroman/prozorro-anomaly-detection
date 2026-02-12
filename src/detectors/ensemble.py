"""
Ensemble Anomaly Detection for Public Procurement.

This module combines multiple detection methods:
1. Rule-based (red flags)
2. Statistical screens
3. Isolation Forest
4. LOF (Local Outlier Factor)
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
            lof_results=lof_df,
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
            "lof": 0.8,
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
        lof_results: Optional[pd.DataFrame] = None,
        network_results: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Combine results from multiple detectors.

        Each detector uses its native column names:
        - RuleBasedDetector: rule_risk_score, rule_flags_count, rule_risk_level
        - StatisticalDetector: stat_score, stat_anomaly, stat_flags_count
        - PyODDetector (IForest): score, anomaly
        - AggregatedPyOD (LOF): score, anomaly
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

        # --- LOF ---
        if lof_results is not None and "tender_id" in lof_results.columns:
            df = lof_results[["tender_id"]].copy()
            # AggregatedPyOD (LOF) returns 'lof_score' and 'lof_anomaly'
            if "lof_score" in lof_results.columns:
                df["lof_score"] = lof_results["lof_score"].values
            elif "score" in lof_results.columns:
                df["lof_score"] = lof_results["score"].values
            else:
                df["lof_score"] = 0
            # Rank-based normalization: PyOD LOF scores are compressed near zero
            # (median anomalous ~0.0008, normal ~0.0002), making them useless
            # for weighted averaging and display. Percentile rank preserves
            # ordering while mapping to interpretable 0-1 range.
            df["lof_score"] = df["lof_score"].rank(pct=True)
            if "lof_anomaly" in lof_results.columns:
                df["lof_anomaly"] = lof_results["lof_anomaly"].values
            elif "anomaly" in lof_results.columns:
                df["lof_anomaly"] = lof_results["anomaly"].values
            else:
                df["lof_anomaly"] = 0
            all_dfs.append(df)
            methods_used.append("lof")
            print(f"  LOF: {len(df):,} tenders, {df['lof_anomaly'].sum():,} flagged")

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

    def generate_explanations(
        self,
        tenders_df: pd.DataFrame,
        rule_results: Optional[pd.DataFrame] = None,
        network_results: Optional[pd.DataFrame] = None,
        buyer_portraits: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Generate human-readable explanations for flagged tenders.

        Args:
            tenders_df: Original tenders with tender_id, buyer_id, etc.
            rule_results: Rule detector output with flag_* columns.
            network_results: Network detector output with network_* columns.
            buyer_portraits: Buyer-level portraits (from buyers.csv or aggregation).

        Returns:
            Series of explanation strings indexed like self.results.
        """
        if self.results is None:
            raise ValueError("Run combine() first")

        print("Generating explanations for flagged tenders...")
        result = self.results
        explanations = pd.Series("", index=result.index)

        # Only generate for tenders with consensus >= 2
        mask = result["consensus_count"] >= 2
        if mask.sum() == 0:
            return explanations

        # --- Methods summary ---
        method_names = {
            "rule": "Правила",
            "stat": "Статистика",
            "if": "IForest",
            "lof": "LOF",
            "network": "Мережа",
        }
        parts_methods = []
        for method in self.methods_used:
            anomaly_col = f"{method}_anomaly"
            score_col = f"{method}_score"
            if anomaly_col in result.columns:
                name = method_names.get(method, method)
                flagged = result[anomaly_col] == 1
                score_str = result[score_col].apply(lambda s: f"{s:.2f}") if score_col in result.columns else ""
                parts_methods.append((method, name, flagged, score_str))

        def build_methods_line(idx):
            active = []
            for method, name, flagged_series, score_str in parts_methods:
                if flagged_series.iloc[idx]:
                    s = score_str.iloc[idx] if isinstance(score_str, pd.Series) else ""
                    active.append(f"{name}({s})")
            return active

        # --- Rule flags (batch) ---
        rule_flag_map = {}
        if rule_results is not None:
            flag_cols = [c for c in rule_results.columns if c.startswith("flag_")]
            if flag_cols:
                rule_indexed = rule_results.set_index("tender_id")
                for col in flag_cols:
                    rule_flag_map[col] = rule_indexed[col]

        # --- Network flags (batch) ---
        net_cols_map = {}
        if network_results is not None:
            for col in ["network_rotation", "network_monopolistic", "network_suspicious_supplier"]:
                if col in network_results.columns:
                    net_cols_map[col] = network_results.set_index("tender_id")[col]

        # --- Buyer portraits (batch) ---
        buyer_data = {}
        if buyer_portraits is not None:
            buyer_indexed = buyer_portraits.set_index("buyer_id")
            portrait_cols = ["single_bidder_rate", "competitive_rate", "supplier_diversity_index",
                             "avg_discount_pct", "total_tenders", "total_value"]
            portrait_cols = [c for c in portrait_cols if c in buyer_indexed.columns]
            buyer_data = {col: buyer_indexed[col] for col in portrait_cols}

        # --- Tender buyer_id mapping ---
        tender_buyer = {}
        if buyer_data and tenders_df is not None and "buyer_id" in tenders_df.columns:
            tender_buyer = tenders_df.set_index("tender_id")["buyer_id"]

        # Vectorized explanation building
        flagged_indices = result.index[mask]
        flagged_tender_ids = result.loc[mask, "tender_id"]

        print(f"  Building explanations for {len(flagged_indices):,} tenders...")

        explanations_list = []
        for idx, tid in zip(flagged_indices, flagged_tender_ids):
            parts = []

            # 1. Methods line
            active = build_methods_line(idx)
            consensus = int(result.at[idx, "consensus_count"])
            parts.append(f"Консенсус: {consensus}/5 ({', '.join(active)})")

            # 2. Rule flags
            if rule_flag_map:
                fired = []
                for col, series in rule_flag_map.items():
                    if tid in series.index and series[tid] == 1:
                        fired.append(col.replace("flag_", ""))
                if fired:
                    parts.append(f"Правила: {', '.join(fired[:8])}" +
                                 (f" (+{len(fired)-8})" if len(fired) > 8 else ""))

            # 3. Network flags
            if net_cols_map:
                net_flags = []
                net_labels = {
                    "network_rotation": "ротація перемог",
                    "network_monopolistic": "монополія",
                    "network_suspicious_supplier": "підозріла зв'язність",
                }
                for col, series in net_cols_map.items():
                    if tid in series.index and series[tid] == 1:
                        net_flags.append(net_labels.get(col, col))
                if net_flags:
                    parts.append(f"Мережа: {', '.join(net_flags)}")

            # 4. Buyer portrait
            if buyer_data and tid in tender_buyer.index:
                bid = tender_buyer[tid]
                portrait_parts = []
                if "single_bidder_rate" in buyer_data and bid in buyer_data["single_bidder_rate"].index:
                    val = buyer_data["single_bidder_rate"][bid]
                    if val > 0.5:
                        portrait_parts.append(f"single_bidder={val:.0%}")
                if "supplier_diversity_index" in buyer_data and bid in buyer_data["supplier_diversity_index"].index:
                    val = buyer_data["supplier_diversity_index"][bid]
                    if val < 0.3:
                        portrait_parts.append(f"diversity={val:.2f}")
                if "competitive_rate" in buyer_data and bid in buyer_data["competitive_rate"].index:
                    val = buyer_data["competitive_rate"][bid]
                    if val < 0.1:
                        portrait_parts.append(f"competitive={val:.0%}")
                if portrait_parts:
                    parts.append(f"Портрет: {', '.join(portrait_parts)}")

            explanations_list.append(" | ".join(parts))

        explanations.loc[mask] = explanations_list
        print(f"  Done. {len(explanations_list):,} explanations generated.")
        return explanations
