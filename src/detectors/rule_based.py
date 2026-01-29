"""
Rule-Based Anomaly Detection (Level 1 - Baseline)

Implements 40+ red flags based on:
- Red_Flags_Prozorro.xlsx (36 rules)
- Additional domain knowledge rules

Categories:
1. Process Quality (Якість процесу)
2. Competition Quality (Якість конкуренції)
3. Price Quality (Якість цін)
4. Procedure Manipulation (Маніпуляції з процедурою)
5. Reputation (Репутація)
6. Additional Flags (Додаткові)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from scipy import stats

from ..config import Thresholds, RiskLevel, ProcurementMethod


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RedFlag:
    """Represents a detected red flag."""
    id: str
    name: str
    name_ua: str
    category: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    affected_records: int = 0
    percentage: float = 0.0


@dataclass
class RuleConfig:
    """Configuration for a single rule."""
    id: str
    name: str
    name_ua: str
    category: str
    severity: str
    description: str
    weight: int  # Score weight for risk calculation
    requires_bids: bool = False
    requires_aggregation: bool = False


# =============================================================================
# Rule Definitions
# =============================================================================

RULE_DEFINITIONS: Dict[str, RuleConfig] = {
    # === Process Quality (Якість процесу) ===
    "R005": RuleConfig(
        id="R005",
        name="missing_documents",
        name_ua="Відсутні документи",
        category="process_quality",
        severity="medium",
        description="Key tender information and documents are not available",
        weight=1
    ),
    "R013": RuleConfig(
        id="R013",
        name="high_limited_usage",
        name_ua="Надмірне використання limited",
        category="process_quality",
        severity="high",
        description="Buyer has high use of non-competitive methods",
        weight=2,
        requires_aggregation=True
    ),
    "R039": RuleConfig(
        id="R039",
        name="no_enquiries",
        name_ua="Без запитань учасників",
        category="process_quality",
        severity="low",
        description="Open tender without any bidder questions",
        weight=1
    ),

    # === Competition Quality (Якість конкуренції) ===
    "R018": RuleConfig(
        id="R018",
        name="single_bidder",
        name_ua="Один учасник",
        category="competition_quality",
        severity="high",
        description="Single bid received in competitive tender",
        weight=2
    ),
    "R019": RuleConfig(
        id="R019",
        name="low_bidders_for_cpv",
        name_ua="Мало учасників для категорії",
        category="competition_quality",
        severity="medium",
        description="Low number of bidders compared to category average",
        weight=1,
        requires_aggregation=True
    ),
    "R040": RuleConfig(
        id="R040",
        name="buyer_supplier_dominance",
        name_ua="Домінування постачальника",
        category="competition_quality",
        severity="high",
        description="Supplier wins high share of buyer's contracts",
        weight=2,
        requires_aggregation=True
    ),
    "R050": RuleConfig(
        id="R050",
        name="high_market_share",
        name_ua="Висока частка ринку",
        category="competition_quality",
        severity="medium",
        description="Supplier has high market share in CPV category",
        weight=1,
        requires_aggregation=True
    ),
    "R051": RuleConfig(
        id="R051",
        name="market_concentration",
        name_ua="Концентрація ринку",
        category="competition_quality",
        severity="medium",
        description="High market concentration (HHI) in CPV category",
        weight=1,
        requires_aggregation=True
    ),

    # === Price Quality (Якість цін) ===
    "R017": RuleConfig(
        id="R017",
        name="price_outlier",
        name_ua="Аномальна ціна",
        category="price_quality",
        severity="high",
        description="Tender value is unreasonably low or high for category",
        weight=2,
        requires_aggregation=True
    ),
    "R022": RuleConfig(
        id="R022",
        name="wide_bid_disparity",
        name_ua="Великий розкид ставок",
        category="price_quality",
        severity="medium",
        description="Wide disparity in bid prices within tender",
        weight=1,
        requires_bids=True
    ),
    "R023": RuleConfig(
        id="R023",
        name="round_bid_prices",
        name_ua="Круглі ставки",
        category="price_quality",
        severity="low",
        description="Bid prices are suspiciously round numbers",
        weight=1,
        requires_bids=True
    ),
    "R024": RuleConfig(
        id="R024",
        name="close_to_winner",
        name_ua="Ставка близька до переможця",
        category="price_quality",
        severity="medium",
        description="Second bid price very close to winning bid",
        weight=1,
        requires_bids=True
    ),
    "R025": RuleConfig(
        id="R025",
        name="low_win_rate",
        name_ua="Низький win rate",
        category="price_quality",
        severity="medium",
        description="Bidder has excessively low win rate (professional loser)",
        weight=1,
        requires_bids=True
    ),
    "R027": RuleConfig(
        id="R027",
        name="missing_regular_bidders",
        name_ua="Відсутні регулярні учасники",
        category="price_quality",
        severity="medium",
        description="Regular bidders in category are missing from tender",
        weight=1,
        requires_bids=True,
        requires_aggregation=True
    ),
    "R028": RuleConfig(
        id="R028",
        name="identical_bids",
        name_ua="Ідентичні ставки",
        category="price_quality",
        severity="critical",
        description="Multiple bidders submitted identical bid prices",
        weight=3,
        requires_bids=True
    ),
    "R034": RuleConfig(
        id="R034",
        name="same_bid_order",
        name_ua="Однаковий порядок ставок",
        category="price_quality",
        severity="high",
        description="Bids consistently submitted in same order across tenders",
        weight=2,
        requires_bids=True
    ),
    "R053": RuleConfig(
        id="R053",
        name="cobidding_same_winner",
        name_ua="Спільні ставки з постійним переможцем",
        category="price_quality",
        severity="critical",
        description="Co-bidding pairs always have same winner",
        weight=3,
        requires_bids=True,
        requires_aggregation=True
    ),
    "R057": RuleConfig(
        id="R057",
        name="bid_rotation",
        name_ua="Ротація ставок",
        category="price_quality",
        severity="critical",
        description="Suppliers rotate winning in same CPV category",
        weight=3,
        requires_aggregation=True
    ),
    "R058": RuleConfig(
        id="R058",
        name="extreme_discount",
        name_ua="Екстремальна знижка",
        category="price_quality",
        severity="medium",
        description="Winning bid has unusually high discount (>50%)",
        weight=1
    ),

    # === Procedure Manipulation (Маніпуляції з процедурою) ===
    "R002": RuleConfig(
        id="R002",
        name="threshold_manipulation",
        name_ua="Маніпуляція порогами",
        category="procedure_manipulation",
        severity="high",
        description="Tender value suspiciously close to threshold",
        weight=2
    ),
    "R011": RuleConfig(
        id="R011",
        name="contract_splitting",
        name_ua="Дроблення закупівель",
        category="procedure_manipulation",
        severity="critical",
        description="Multiple tenders split to avoid competitive threshold",
        weight=3,
        requires_aggregation=True
    ),
    "R016": RuleConfig(
        id="R016",
        name="value_zscore_outlier",
        name_ua="Z-score вартості",
        category="procedure_manipulation",
        severity="medium",
        description="Tender value significantly deviates from CPV average",
        weight=1,
        requires_aggregation=True
    ),
    "R021": RuleConfig(
        id="R021",
        name="discretionary_criteria",
        name_ua="Дискреційні критерії",
        category="procedure_manipulation",
        severity="medium",
        description="Use of discretionary evaluation criteria instead of lowest cost",
        weight=1
    ),
    "R029": RuleConfig(
        id="R029",
        name="benford_violation",
        name_ua="Порушення закону Бенфорда",
        category="procedure_manipulation",
        severity="medium",
        description="Bid prices deviate from Benford's Law distribution",
        weight=1,
        requires_bids=True
    ),
    "R031": RuleConfig(
        id="R031",
        name="bid_near_estimate",
        name_ua="Ставка близька до оцінки",
        category="procedure_manipulation",
        severity="high",
        description="Winning bid very close to or higher than estimated price",
        weight=2
    ),
    "R035": RuleConfig(
        id="R035",
        name="all_except_winner_disqualified",
        name_ua="Всі крім переможця дискваліфіковані",
        category="procedure_manipulation",
        severity="critical",
        description="All bids except winner were disqualified",
        weight=3,
        requires_bids=True
    ),
    "R036": RuleConfig(
        id="R036",
        name="lowest_bid_disqualified",
        name_ua="Найнижча ставка дискваліфікована",
        category="procedure_manipulation",
        severity="high",
        description="Lowest bid was disqualified",
        weight=2,
        requires_bids=True
    ),
    "R038": RuleConfig(
        id="R038",
        name="excessive_disqualifications",
        name_ua="Надмірні дискваліфікації",
        category="procedure_manipulation",
        severity="high",
        description="Buyer or bidder has excessive disqualification rate",
        weight=2,
        requires_bids=True,
        requires_aggregation=True
    ),
    "R049": RuleConfig(
        id="R049",
        name="direct_awards_pattern",
        name_ua="Патерн прямих закупівель",
        category="procedure_manipulation",
        severity="high",
        description="Pattern of direct awards just below threshold",
        weight=2,
        requires_aggregation=True
    ),
    "R052": RuleConfig(
        id="R052",
        name="small_then_large",
        name_ua="Малий контракт потім великий",
        category="procedure_manipulation",
        severity="high",
        description="Small initial purchase followed by much larger purchases",
        weight=2,
        requires_aggregation=True
    ),
    "R055": RuleConfig(
        id="R055",
        name="multiple_near_threshold",
        name_ua="Багато закупівель біля порогу",
        category="procedure_manipulation",
        severity="high",
        description="Multiple direct awards near competitive threshold",
        weight=2,
        requires_aggregation=True
    ),
    "R059": RuleConfig(
        id="R059",
        name="award_contract_difference",
        name_ua="Різниця award/contract",
        category="procedure_manipulation",
        severity="high",
        description="Large difference between award value and contract amount",
        weight=2
    ),

    # === Reputation (Репутація) ===
    "R048": RuleConfig(
        id="R048",
        name="heterogeneous_supplier",
        name_ua="Гетерогенний постачальник",
        category="reputation",
        severity="medium",
        description="Supplier operates in many unrelated CPV categories",
        weight=1,
        requires_aggregation=True
    ),
    "R069": RuleConfig(
        id="R069",
        name="price_increase",
        name_ua="Збільшення ціни",
        category="reputation",
        severity="critical",
        description="Contract amendments increased price above tender value",
        weight=3
    ),

    # === Additional Flags (Додаткові - мої ідеї) ===
    "X001": RuleConfig(
        id="X001",
        name="weekend_publication",
        name_ua="Публікація у вихідний",
        category="additional",
        severity="low",
        description="Tender published on weekend (reduced visibility)",
        weight=1
    ),
    "X002": RuleConfig(
        id="X002",
        name="q4_rush",
        name_ua="Q4 rush",
        category="additional",
        severity="low",
        description="High-value tender in Q4 (budget spending pressure)",
        weight=1
    ),
    "X003": RuleConfig(
        id="X003",
        name="december_rush",
        name_ua="Грудневий rush",
        category="additional",
        severity="medium",
        description="High-value tender in December (year-end pressure)",
        weight=1
    ),
    "X004": RuleConfig(
        id="X004",
        name="cross_region_single_bidder",
        name_ua="Міжрегіональний single bidder",
        category="additional",
        severity="high",
        description="Single bidder from different region",
        weight=2
    ),
    "X005": RuleConfig(
        id="X005",
        name="masked_data",
        name_ua="Замасковані дані",
        category="additional",
        severity="low",
        description="Buyer or supplier data is masked",
        weight=1
    ),
    "X006": RuleConfig(
        id="X006",
        name="award_issues",
        name_ua="Проблеми з awards",
        category="additional",
        severity="medium",
        description="Has unsuccessful or cancelled awards",
        weight=1
    ),
    "X007": RuleConfig(
        id="X007",
        name="new_supplier_large_contract",
        name_ua="Новий постачальник великий контракт",
        category="additional",
        severity="high",
        description="New supplier immediately wins large contract",
        weight=2,
        requires_aggregation=True
    ),
    "X008": RuleConfig(
        id="X008",
        name="captive_supplier",
        name_ua="Залежний постачальник",
        category="additional",
        severity="high",
        description="Supplier wins only from one buyer",
        weight=2,
        requires_aggregation=True
    ),
    "X009": RuleConfig(
        id="X009",
        name="single_bidder_low_discount",
        name_ua="Single bidder низька знижка",
        category="additional",
        severity="critical",
        description="Single bidder with discount less than 2%",
        weight=3
    ),
    "X010": RuleConfig(
        id="X010",
        name="same_day_same_supplier",
        name_ua="Той самий день той самий постачальник",
        category="additional",
        severity="high",
        description="Multiple awards to same supplier on same day",
        weight=2,
        requires_aggregation=True
    ),
}


# =============================================================================
# Main Detector Class
# =============================================================================

class RuleBasedDetector:
    """
    Comprehensive rule-based anomaly detector.

    Implements 40+ red flags organized by category:
    - Process Quality
    - Competition Quality
    - Price Quality
    - Procedure Manipulation
    - Reputation
    - Additional Flags

    Usage:
        detector = RuleBasedDetector()
        results = detector.detect(tenders_df)
        print(detector.summary())
    """

    def __init__(
        self,
        thresholds: Optional[Thresholds] = None,
        enabled_rules: Optional[List[str]] = None,
        disabled_rules: Optional[List[str]] = None,
    ):
        """
        Initialize detector.

        Args:
            thresholds: Custom thresholds. None = use defaults.
            enabled_rules: Only run these rules. None = run all.
            disabled_rules: Skip these rules.
        """
        self.thresholds = thresholds or Thresholds()
        self.rule_configs = RULE_DEFINITIONS.copy()

        # Filter rules
        if enabled_rules:
            self.rule_configs = {k: v for k, v in self.rule_configs.items() if k in enabled_rules}
        if disabled_rules:
            self.rule_configs = {k: v for k, v in self.rule_configs.items() if k not in disabled_rules}

        self.flags_detected: List[RedFlag] = []
        self.results: Optional[pd.DataFrame] = None
        self._cpv_stats: Optional[pd.DataFrame] = None
        self._buyer_stats: Optional[pd.DataFrame] = None
        self._supplier_stats: Optional[pd.DataFrame] = None

    def detect(
        self,
        df: pd.DataFrame,
        buyers_df: Optional[pd.DataFrame] = None,
        suppliers_df: Optional[pd.DataFrame] = None,
        bids_df: Optional[pd.DataFrame] = None,
        compute_aggregations: bool = True,
    ) -> pd.DataFrame:
        """
        Apply all rule-based checks to the data.

        Args:
            df: Tender DataFrame
            buyers_df: Buyers reference table (optional, for enrichment)
            suppliers_df: Suppliers reference table (optional)
            bids_df: Bids data (optional, enables bid-level rules)
            compute_aggregations: Whether to compute aggregations for context-aware rules

        Returns:
            DataFrame with added flag columns and risk scores
        """
        result = df.copy()

        # Compute aggregations if needed
        if compute_aggregations:
            self._compute_aggregations(result)

        # Merge reference data if provided
        if buyers_df is not None:
            result = self._merge_buyers(result, buyers_df)
        if suppliers_df is not None:
            result = self._merge_suppliers(result, suppliers_df)

        # Initialize score
        result["rule_risk_score"] = 0
        result["rule_flags_count"] = 0

        # Apply all rules
        for rule_id, config in self.rule_configs.items():
            # Skip rules that need bids if not provided
            if config.requires_bids and bids_df is None:
                continue

            method_name = f"_check_{config.name}"
            if hasattr(self, method_name):
                try:
                    flag_col = f"flag_{config.name}"
                    if config.requires_bids:
                        result[flag_col] = getattr(self, method_name)(result, bids_df)
                    else:
                        result[flag_col] = getattr(self, method_name)(result)

                    # Update scores
                    mask = result[flag_col] == 1
                    result.loc[mask, "rule_risk_score"] += config.weight
                    result.loc[mask, "rule_flags_count"] += 1
                except Exception as e:
                    print(f"Warning: Rule {rule_id} failed: {e}")
                    result[f"flag_{config.name}"] = 0

        # Calculate risk levels
        result["rule_risk_level"] = self._calculate_risk_level(result["rule_risk_score"])

        # Compute summary
        self._compute_flags_summary(result)
        self.results = result

        return result

    # =========================================================================
    # Aggregation Methods
    # =========================================================================

    def _compute_aggregations(self, df: pd.DataFrame):
        """Compute aggregated statistics for context-aware rules."""
        # CPV statistics
        self._cpv_stats = df.groupby("main_cpv_2_digit").agg({
            "tender_value": ["mean", "median", "std", "count"],
            "number_of_tenderers": ["mean", "median"],
            "price_change_pct": ["mean", "median"],
        }).reset_index()
        self._cpv_stats.columns = [
            "cpv", "cpv_value_mean", "cpv_value_median", "cpv_value_std", "cpv_count",
            "cpv_tenderers_mean", "cpv_tenderers_median",
            "cpv_discount_mean", "cpv_discount_median"
        ]

        # Buyer statistics
        self._buyer_stats = df.groupby("buyer_id").agg({
            "tender_id": "count",
            "is_single_bidder": "mean",
            "procurement_method": lambda x: (x == "limited").mean(),
            "tender_value": "sum",
        }).reset_index()
        self._buyer_stats.columns = [
            "buyer_id", "buyer_tender_count", "buyer_single_bidder_rate",
            "buyer_limited_rate", "buyer_total_value"
        ]

        # Supplier statistics
        self._supplier_stats = df.groupby("supplier_id").agg({
            "tender_id": "count",
            "tender_value": "sum",
            "main_cpv_2_digit": "nunique",
            "buyer_id": "nunique",
        }).reset_index()
        self._supplier_stats.columns = [
            "supplier_id", "supplier_win_count", "supplier_total_value",
            "supplier_cpv_count", "supplier_buyer_count"
        ]

        # Buyer-Supplier pair statistics
        self._pair_stats = df.groupby(["buyer_id", "supplier_id"]).agg({
            "tender_id": "count",
            "tender_value": "sum",
        }).reset_index()
        self._pair_stats.columns = ["buyer_id", "supplier_id", "pair_count", "pair_value"]

    def _merge_buyers(self, df: pd.DataFrame, buyers_df: pd.DataFrame) -> pd.DataFrame:
        """Merge buyer reference data."""
        return df.merge(
            buyers_df[["buyer_id", "single_bidder_rate", "competitive_rate",
                      "supplier_diversity_index", "buyer_region"]],
            on="buyer_id",
            how="left",
            suffixes=("", "_ref")
        )

    def _merge_suppliers(self, df: pd.DataFrame, suppliers_df: pd.DataFrame) -> pd.DataFrame:
        """Merge supplier reference data."""
        return df.merge(
            suppliers_df[["supplier_id", "total_awards", "total_value"]],
            on="supplier_id",
            how="left",
            suffixes=("", "_ref")
        )

    # =========================================================================
    # Process Quality Rules
    # =========================================================================

    def _check_missing_documents(self, df: pd.DataFrame) -> pd.Series:
        """R005: Missing documents."""
        return (df["number_of_documents"] == 0).astype(int)

    def _check_high_limited_usage(self, df: pd.DataFrame) -> pd.Series:
        """R013: Buyer has high use of limited procurement."""
        if self._buyer_stats is None:
            return pd.Series(0, index=df.index)

        merged = df.merge(self._buyer_stats[["buyer_id", "buyer_limited_rate"]],
                         on="buyer_id", how="left")
        return (merged["buyer_limited_rate"].fillna(0) > 0.95).astype(int)

    def _check_no_enquiries(self, df: pd.DataFrame) -> pd.Series:
        """R039: Open tender without enquiries (questions)."""
        return (
            (df["procurement_method"] == ProcurementMethod.OPEN) &
            (df["has_enquiries"] == 0)
        ).astype(int)

    # =========================================================================
    # Competition Quality Rules
    # =========================================================================

    def _check_single_bidder(self, df: pd.DataFrame) -> pd.Series:
        """R018: Single bidder in competitive tender."""
        return (
            (df["is_single_bidder"] == 1) &
            (df["procurement_method"].isin([ProcurementMethod.OPEN, ProcurementMethod.SELECTIVE]))
        ).astype(int)

    def _check_low_bidders_for_cpv(self, df: pd.DataFrame) -> pd.Series:
        """R019: Low number of bidders compared to CPV average."""
        if self._cpv_stats is None:
            return pd.Series(0, index=df.index)

        merged = df.merge(
            self._cpv_stats[["cpv", "cpv_tenderers_median"]],
            left_on="main_cpv_2_digit", right_on="cpv", how="left"
        )
        return (
            (merged["number_of_tenderers"] > 0) &
            (merged["number_of_tenderers"] < merged["cpv_tenderers_median"] * 0.5)
        ).astype(int)

    def _check_buyer_supplier_dominance(self, df: pd.DataFrame) -> pd.Series:
        """R040: Supplier wins high share of buyer's contracts."""
        if self._pair_stats is None or self._buyer_stats is None:
            return pd.Series(0, index=df.index)

        # Calculate share
        pair_with_buyer = self._pair_stats.merge(
            self._buyer_stats[["buyer_id", "buyer_tender_count"]],
            on="buyer_id"
        )
        pair_with_buyer["share"] = pair_with_buyer["pair_count"] / pair_with_buyer["buyer_tender_count"]

        # Flag pairs with >50% share and at least 5 contracts
        suspicious_pairs = pair_with_buyer[
            (pair_with_buyer["share"] > 0.5) &
            (pair_with_buyer["pair_count"] >= 5)
        ][["buyer_id", "supplier_id"]]

        # Mark tenders
        merged = df.merge(suspicious_pairs, on=["buyer_id", "supplier_id"], how="left", indicator=True)
        return (merged["_merge"] == "both").astype(int)

    def _check_high_market_share(self, df: pd.DataFrame) -> pd.Series:
        """R050: Supplier has high market share in CPV."""
        if self._cpv_stats is None:
            return pd.Series(0, index=df.index)

        # Calculate supplier share per CPV
        cpv_supplier = df.groupby(["main_cpv_2_digit", "supplier_id"]).agg({
            "tender_value": "sum"
        }).reset_index()

        cpv_total = df.groupby("main_cpv_2_digit")["tender_value"].sum().reset_index()
        cpv_total.columns = ["main_cpv_2_digit", "cpv_total_value"]

        cpv_supplier = cpv_supplier.merge(cpv_total, on="main_cpv_2_digit")
        cpv_supplier["market_share"] = cpv_supplier["tender_value"] / cpv_supplier["cpv_total_value"]

        # Flag suppliers with >30% share
        high_share = cpv_supplier[cpv_supplier["market_share"] > 0.3][["main_cpv_2_digit", "supplier_id"]]

        merged = df.merge(high_share, on=["main_cpv_2_digit", "supplier_id"], how="left", indicator=True)
        return (merged["_merge"] == "both").astype(int)

    def _check_market_concentration(self, df: pd.DataFrame) -> pd.Series:
        """R051: High market concentration (HHI) in CPV."""
        # Calculate HHI per CPV
        cpv_supplier = df.groupby(["main_cpv_2_digit", "supplier_id"]).agg({
            "tender_value": "sum"
        }).reset_index()

        cpv_total = df.groupby("main_cpv_2_digit")["tender_value"].sum().reset_index()
        cpv_total.columns = ["main_cpv_2_digit", "cpv_total"]

        cpv_supplier = cpv_supplier.merge(cpv_total, on="main_cpv_2_digit")
        cpv_supplier["share"] = cpv_supplier["tender_value"] / cpv_supplier["cpv_total"]
        cpv_supplier["share_sq"] = cpv_supplier["share"] ** 2

        hhi = cpv_supplier.groupby("main_cpv_2_digit")["share_sq"].sum().reset_index()
        hhi.columns = ["main_cpv_2_digit", "hhi"]

        # Flag CPVs with HHI > 0.25 (concentrated market)
        high_hhi_cpv = hhi[hhi["hhi"] > 0.25]["main_cpv_2_digit"].tolist()

        return df["main_cpv_2_digit"].isin(high_hhi_cpv).astype(int)

    # =========================================================================
    # Price Quality Rules
    # =========================================================================

    def _check_price_outlier(self, df: pd.DataFrame) -> pd.Series:
        """R017: Tender value is outlier for CPV category."""
        if self._cpv_stats is None:
            return pd.Series(0, index=df.index)

        merged = df.merge(
            self._cpv_stats[["cpv", "cpv_value_mean", "cpv_value_std"]],
            left_on="main_cpv_2_digit", right_on="cpv", how="left"
        )

        # Z-score
        zscore = (merged["tender_value"] - merged["cpv_value_mean"]) / merged["cpv_value_std"].replace(0, np.nan)
        return (zscore.abs() > 3).fillna(False).astype(int)

    def _check_wide_bid_disparity(self, df: pd.DataFrame, bids_df: pd.DataFrame) -> pd.Series:
        """R022: Wide disparity in bid prices."""
        bid_stats = bids_df.groupby("tender_id").agg({
            "bid_amount": ["min", "max", "mean", "std"]
        }).reset_index()
        bid_stats.columns = ["tender_id", "bid_min", "bid_max", "bid_mean", "bid_std"]
        bid_stats["bid_range_ratio"] = (bid_stats["bid_max"] - bid_stats["bid_min"]) / bid_stats["bid_mean"]

        merged = df.merge(bid_stats[["tender_id", "bid_range_ratio"]], on="tender_id", how="left")
        return (merged["bid_range_ratio"].fillna(0) > 1.0).astype(int)  # Range > mean

    def _check_round_bid_prices(self, df: pd.DataFrame, bids_df: pd.DataFrame) -> pd.Series:
        """R023: Suspiciously round bid prices."""
        # Check if bid amounts are round (divisible by 1000 or 10000)
        bids_df = bids_df.copy()
        bids_df["is_round"] = (
            (bids_df["bid_amount"] % 10000 == 0) |
            (bids_df["bid_amount"] % 1000 == 0)
        )

        round_rate = bids_df.groupby("tender_id")["is_round"].mean().reset_index()
        round_rate.columns = ["tender_id", "round_rate"]

        merged = df.merge(round_rate, on="tender_id", how="left")
        return (merged["round_rate"].fillna(0) > 0.8).astype(int)  # >80% round bids

    def _check_close_to_winner(self, df: pd.DataFrame, bids_df: pd.DataFrame) -> pd.Series:
        """R024: Second bid very close to winning bid."""
        # Get winning and second bid
        bids_sorted = bids_df.sort_values(["tender_id", "bid_amount"])

        def get_price_diff(group):
            if len(group) < 2:
                return np.nan
            sorted_bids = group.sort_values("bid_amount")
            first = sorted_bids.iloc[0]["bid_amount"]
            second = sorted_bids.iloc[1]["bid_amount"]
            if first == 0:
                return np.nan
            return (second - first) / first

        price_diffs = bids_df.groupby("tender_id").apply(get_price_diff).reset_index()
        price_diffs.columns = ["tender_id", "price_diff"]

        merged = df.merge(price_diffs, on="tender_id", how="left")
        # Flag if difference < 1% (suspiciously close)
        return ((merged["price_diff"].fillna(1) < 0.01) & (merged["price_diff"].fillna(1) >= 0)).astype(int)

    def _check_low_win_rate(self, df: pd.DataFrame, bids_df: pd.DataFrame = None) -> pd.Series:
        """R025: Bidder with excessively low win rate (professional loser)."""
        if bids_df is None:
            return pd.Series(0, index=df.index)

        # Calculate win rate per bidder
        bidder_stats = bids_df.groupby("bidder_id").agg({
            "is_winner": ["sum", "count"]
        }).reset_index()
        bidder_stats.columns = ["bidder_id", "wins", "total_bids"]
        bidder_stats["win_rate"] = bidder_stats["wins"] / bidder_stats["total_bids"]

        # Flag bidders with many bids but very low win rate (<5% with 10+ bids)
        # These might be "professional losers" in bid rigging schemes
        suspicious_bidders = bidder_stats[
            (bidder_stats["total_bids"] >= 10) &
            (bidder_stats["win_rate"] < 0.05)
        ]["bidder_id"]

        # Get tenders where these suspicious bidders participated
        suspicious_tenders = bids_df[bids_df["bidder_id"].isin(suspicious_bidders)]["tender_id"].unique()

        return df["tender_id"].isin(suspicious_tenders).astype(int)

    def _check_identical_bids(self, df: pd.DataFrame, bids_df: pd.DataFrame) -> pd.Series:
        """R028: Multiple bidders with identical bid amounts."""
        # Count duplicate bid amounts per tender
        bid_counts = bids_df.groupby(["tender_id", "bid_amount"]).size().reset_index(name="count")
        has_duplicates = bid_counts[bid_counts["count"] > 1]["tender_id"].unique()

        return df["tender_id"].isin(has_duplicates).astype(int)

    def _check_same_bid_order(self, df: pd.DataFrame, bids_df: pd.DataFrame) -> pd.Series:
        """R034: Bids consistently submitted in same order."""
        # This requires tracking bid submission order across multiple tenders
        # Simplified: flag if all bids submitted within very short time
        if "bid_date" not in bids_df.columns:
            return pd.Series(0, index=df.index)

        bids_df = bids_df.copy()
        bids_df["bid_date"] = pd.to_datetime(bids_df["bid_date"], errors="coerce")

        time_spread = bids_df.groupby("tender_id").agg({
            "bid_date": lambda x: (x.max() - x.min()).total_seconds() / 3600 if len(x) > 1 else np.nan
        }).reset_index()
        time_spread.columns = ["tender_id", "hours_spread"]

        merged = df.merge(time_spread, on="tender_id", how="left")
        # Flag if all bids within 1 hour (suspicious coordination)
        return ((merged["hours_spread"].fillna(999) < 1) & (merged["number_of_bids"] > 1)).astype(int)

    def _check_cobidding_same_winner(self, df: pd.DataFrame, bids_df: pd.DataFrame) -> pd.Series:
        """R053: Co-bidding pairs always have same winner (bid rigging indicator)."""
        try:
            # Find pairs of bidders who frequently bid together and one always wins
            # Filter out rows with NA bidder_id
            bids_clean = bids_df[bids_df["bidder_id"].notna()].copy()

            # Get all bidder pairs per tender
            tender_bidders = bids_clean.groupby("tender_id")["bidder_id"].apply(list).reset_index()
            tender_winners = bids_clean[bids_clean["is_winner"] == 1].groupby("tender_id")["bidder_id"].first().reset_index()
            tender_winners.columns = ["tender_id", "winner_id"]

            tender_bidders = tender_bidders.merge(tender_winners, on="tender_id", how="left")

            # Count co-bidding patterns (check if same winner when 2 bidders)
            from collections import defaultdict
            pair_wins = defaultdict(lambda: defaultdict(int))
            pair_total = defaultdict(int)

            for _, row in tender_bidders.iterrows():
                bidders = row["bidder_id"]
                winner = row["winner_id"]
                # Skip if winner is NA/None or bidders list is wrong size
                if isinstance(bidders, list) and len(bidders) == 2 and pd.notna(winner):
                    # Filter out None values in bidders
                    bidders = [b for b in bidders if pd.notna(b)]
                    if len(bidders) == 2:
                        pair = tuple(sorted(bidders))
                        pair_total[pair] += 1
                        pair_wins[pair][winner] += 1

            # Find suspicious pairs: same winner > 80% of time, at least 5 co-bids
            suspicious_pairs = set()
            for pair, total in pair_total.items():
                if total >= 5:
                    max_wins = max(pair_wins[pair].values()) if pair_wins[pair] else 0
                    if max_wins / total >= 0.8:
                        suspicious_pairs.add(pair)

            if not suspicious_pairs:
                return pd.Series(0, index=df.index)

            # Flag tenders with these suspicious pairs
            def has_suspicious_pair(bidders):
                if not isinstance(bidders, list) or len(bidders) != 2:
                    return False
                bidders = [b for b in bidders if pd.notna(b)]
                if len(bidders) != 2:
                    return False
                pair = tuple(sorted(bidders))
                return pair in suspicious_pairs

            tender_bidders["is_suspicious"] = tender_bidders["bidder_id"].apply(has_suspicious_pair)
            suspicious_tenders = tender_bidders[tender_bidders["is_suspicious"]]["tender_id"]

            return df["tender_id"].isin(suspicious_tenders).astype(int)

        except Exception as e:
            # If anything fails, return zeros
            return pd.Series(0, index=df.index)

    def _check_bid_rotation(self, df: pd.DataFrame) -> pd.Series:
        """R057: Suppliers rotate winning in CPV category (cartel indicator)."""
        # Detect if small group of suppliers take turns winning in a CPV

        # Need datetime for ordering
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy["published_date"]):
            df_copy["published_date"] = pd.to_datetime(df_copy["published_date"], errors="coerce")

        suspicious_tenders = set()

        # Analyze each CPV category
        for cpv in df_copy["main_cpv_2_digit"].dropna().unique():
            cpv_df = df_copy[df_copy["main_cpv_2_digit"] == cpv].sort_values("published_date")

            if len(cpv_df) < 10:
                continue

            # Get sequence of winners
            winners = cpv_df["supplier_id"].dropna().tolist()
            if len(winners) < 10:
                continue

            # Check for rotation pattern: small group alternating wins
            unique_winners = set(winners[:20])  # Look at first 20 wins
            if len(unique_winners) <= 3 and len(unique_winners) >= 2:
                # Very few winners - check if they alternate
                # Count transitions between different winners
                transitions = sum(1 for i in range(len(winners)-1) if winners[i] != winners[i+1])
                transition_rate = transitions / (len(winners) - 1)

                # High transition rate with few winners = rotation
                if transition_rate > 0.6:
                    suspicious_tenders.update(cpv_df["tender_id"].tolist())

        return df["tender_id"].isin(suspicious_tenders).astype(int)

    def _check_extreme_discount(self, df: pd.DataFrame) -> pd.Series:
        """R058: Extreme discount (>50%)."""
        return (df["price_change_pct"].fillna(0) > 50).astype(int)

    # =========================================================================
    # Procedure Manipulation Rules
    # =========================================================================

    def _check_threshold_manipulation(self, df: pd.DataFrame) -> pd.Series:
        """R002: Tender value suspiciously close to threshold."""
        # Common thresholds in Ukraine (UAH)
        thresholds = [50000, 200000, 1000000, 5000000]

        flags = pd.Series(False, index=df.index)
        for threshold in thresholds:
            # Within 5% below threshold
            mask = (df["tender_value"] >= threshold * 0.95) & (df["tender_value"] < threshold)
            flags = flags | mask

        return flags.astype(int)

    def _check_contract_splitting(self, df: pd.DataFrame) -> pd.Series:
        """R011: Multiple tenders split to avoid threshold."""
        if self._buyer_stats is None:
            return pd.Series(0, index=df.index)

        # Look for same buyer, same CPV, same day, values near threshold
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy["published_date"]):
            df_copy["published_date"] = pd.to_datetime(df_copy["published_date"], errors="coerce")

        df_copy["pub_date"] = df_copy["published_date"].dt.date

        # Count same-day same-buyer same-CPV tenders
        daily_counts = df_copy.groupby(["buyer_id", "main_cpv_2_digit", "pub_date"]).agg({
            "tender_id": "count",
            "tender_value": ["sum", "mean"]
        }).reset_index()
        daily_counts.columns = ["buyer_id", "main_cpv_2_digit", "pub_date", "day_count", "day_sum", "day_avg"]

        # Flag if multiple tenders on same day with avg value near threshold
        suspicious = daily_counts[
            (daily_counts["day_count"] >= 3) &
            (daily_counts["day_avg"] >= 40000) &
            (daily_counts["day_avg"] < 50000)
        ][["buyer_id", "main_cpv_2_digit", "pub_date"]]

        merged = df_copy.merge(suspicious, on=["buyer_id", "main_cpv_2_digit", "pub_date"], how="left", indicator=True)
        return (merged["_merge"] == "both").astype(int)

    def _check_value_zscore_outlier(self, df: pd.DataFrame) -> pd.Series:
        """R016: Tender value significantly deviates from CPV average."""
        # Same as R017
        return self._check_price_outlier(df)

    def _check_discretionary_criteria(self, df: pd.DataFrame) -> pd.Series:
        """R021: Use of discretionary evaluation criteria."""
        return (
            (df["award_criteria"] != "lowestCost") &
            (df["procurement_method"] == ProcurementMethod.OPEN)
        ).astype(int)

    def _check_benford_violation(self, df: pd.DataFrame, bids_df: pd.DataFrame) -> pd.Series:
        """R029: Bid prices deviate from Benford's Law."""
        # Get first digit distribution
        bids_df = bids_df.copy()
        bids_df["first_digit"] = bids_df["bid_amount"].astype(str).str[0].astype(int, errors="ignore")

        # Expected Benford distribution
        benford = {1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079,
                   6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046}

        # Calculate per tender
        def check_benford(group):
            if len(group) < 10:
                return 0
            observed = group["first_digit"].value_counts(normalize=True)
            chi_sq = 0
            for digit in range(1, 10):
                obs = observed.get(digit, 0)
                exp = benford[digit]
                chi_sq += ((obs - exp) ** 2) / exp
            return 1 if chi_sq > 15.51 else 0  # Chi-sq critical value for df=8, alpha=0.05

        benford_flags = bids_df.groupby("tender_id").apply(check_benford).reset_index()
        benford_flags.columns = ["tender_id", "benford_flag"]

        merged = df.merge(benford_flags, on="tender_id", how="left")
        return merged["benford_flag"].fillna(0).astype(int)

    def _check_bid_near_estimate(self, df: pd.DataFrame) -> pd.Series:
        """R031: Winning bid very close to estimated price (only for competitive tenders)."""
        # Ratio of award to tender value
        ratio = df["award_value"] / df["tender_value"].replace(0, np.nan)
        # Flag only for Open/Selective where we expect discount from competition
        # ratio >= 0.995 means less than 0.5% discount (suspicious for competitive)
        is_competitive = df["procurement_method"].isin([ProcurementMethod.OPEN, ProcurementMethod.SELECTIVE])
        return (is_competitive & ((ratio >= 0.995) | (ratio > 1))).fillna(False).astype(int)

    def _check_all_except_winner_disqualified(self, df: pd.DataFrame, bids_df: pd.DataFrame) -> pd.Series:
        """R035: All bids except winner disqualified."""
        # Count disqualified per tender
        disq_counts = bids_df.groupby("tender_id").agg({
            "bid_status": lambda x: (x == "disqualified").sum(),
            "bid_id": "count"
        }).reset_index()
        disq_counts.columns = ["tender_id", "disqualified_count", "total_bids"]
        disq_counts["all_but_one_disq"] = (
            (disq_counts["disqualified_count"] == disq_counts["total_bids"] - 1) &
            (disq_counts["total_bids"] > 1)
        )

        merged = df.merge(disq_counts[["tender_id", "all_but_one_disq"]], on="tender_id", how="left")
        return merged["all_but_one_disq"].fillna(False).astype(int)

    def _check_lowest_bid_disqualified(self, df: pd.DataFrame, bids_df: pd.DataFrame) -> pd.Series:
        """R036: Lowest bid was disqualified."""
        # Find lowest bid per tender and check if disqualified
        idx_min = bids_df.groupby("tender_id")["bid_amount"].idxmin()
        lowest_bids = bids_df.loc[idx_min, ["tender_id", "bid_status"]]
        lowest_bids["lowest_disqualified"] = (lowest_bids["bid_status"] == "disqualified")

        merged = df.merge(lowest_bids[["tender_id", "lowest_disqualified"]], on="tender_id", how="left")
        return merged["lowest_disqualified"].fillna(False).astype(int)

    def _check_excessive_disqualifications(self, df: pd.DataFrame, bids_df: pd.DataFrame) -> pd.Series:
        """R038: High disqualification rate."""
        disq_rate = bids_df.groupby("tender_id").agg({
            "bid_status": lambda x: (x == "disqualified").mean()
        }).reset_index()
        disq_rate.columns = ["tender_id", "disq_rate"]

        merged = df.merge(disq_rate, on="tender_id", how="left")
        return (merged["disq_rate"].fillna(0) > 0.5).astype(int)  # >50% disqualified

    def _check_direct_awards_pattern(self, df: pd.DataFrame) -> pd.Series:
        """R049: Pattern of direct awards just below threshold."""
        # Count limited tenders near 50K threshold per buyer
        near_threshold = df[
            (df["procurement_method"] == ProcurementMethod.LIMITED) &
            (df["tender_value"] >= 40000) &
            (df["tender_value"] < 50000)
        ]

        buyer_counts = near_threshold.groupby("buyer_id").size().reset_index(name="near_threshold_count")
        suspicious_buyers = buyer_counts[buyer_counts["near_threshold_count"] >= 5]["buyer_id"]

        return (
            df["buyer_id"].isin(suspicious_buyers) &
            (df["tender_value"] >= 40000) &
            (df["tender_value"] < 50000)
        ).astype(int)

    def _check_small_then_large(self, df: pd.DataFrame) -> pd.Series:
        """R052: Small initial purchase followed by much larger purchases."""
        # Pattern: buyer starts with small contract to supplier, then gives large ones
        # This can be used to establish "relationship" before big awards

        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy["published_date"]):
            df_copy["published_date"] = pd.to_datetime(df_copy["published_date"], errors="coerce")

        suspicious_tenders = set()

        # Analyze each buyer-supplier pair
        for (buyer_id, supplier_id), group in df_copy.groupby(["buyer_id", "supplier_id"]):
            if len(group) < 3:
                continue

            group = group.sort_values("published_date")
            values = group["tender_value"].tolist()

            # Check if first contract was small and later ones much larger
            first_value = values[0]
            max_later = max(values[1:]) if len(values) > 1 else 0
            avg_later = sum(values[1:]) / len(values[1:]) if len(values) > 1 else 0

            # Flag if first was <20% of average later and later contracts are 5x larger
            if first_value > 0 and avg_later > 0:
                if first_value < avg_later * 0.2 and max_later > first_value * 5:
                    suspicious_tenders.update(group["tender_id"].tolist())

        return df["tender_id"].isin(suspicious_tenders).astype(int)

    def _check_multiple_near_threshold(self, df: pd.DataFrame) -> pd.Series:
        """R055: Multiple direct awards near threshold to same supplier."""
        # Same supplier gets multiple near-threshold contracts from same buyer
        near_threshold = df[
            (df["procurement_method"] == ProcurementMethod.LIMITED) &
            (df["tender_value"] >= 40000) &
            (df["tender_value"] < 50000)
        ]

        pair_counts = near_threshold.groupby(["buyer_id", "supplier_id"]).size().reset_index(name="pair_near_count")
        suspicious_pairs = pair_counts[pair_counts["pair_near_count"] >= 3][["buyer_id", "supplier_id"]]

        merged = df.merge(suspicious_pairs, on=["buyer_id", "supplier_id"], how="left", indicator=True)
        return (
            (merged["_merge"] == "both") &
            (df["tender_value"] >= 40000) &
            (df["tender_value"] < 50000)
        ).astype(int)

    def _check_award_contract_difference(self, df: pd.DataFrame) -> pd.Series:
        """R059: Large difference between tender and award value."""
        # Using price_change_pct - negative means price increased
        return (df["price_change_pct"].fillna(0) < -10).astype(int)  # >10% increase

    # =========================================================================
    # Reputation Rules
    # =========================================================================

    def _check_heterogeneous_supplier(self, df: pd.DataFrame) -> pd.Series:
        """R048: Supplier operates in many unrelated CPV categories."""
        if self._supplier_stats is None:
            return pd.Series(0, index=df.index)

        # Suppliers with >10 different CPV categories
        diverse_suppliers = self._supplier_stats[
            self._supplier_stats["supplier_cpv_count"] > 10
        ]["supplier_id"]

        return df["supplier_id"].isin(diverse_suppliers).astype(int)

    def _check_price_increase(self, df: pd.DataFrame) -> pd.Series:
        """R069: Contract amendments increased price."""
        return (df["price_change_pct"].fillna(0) < 0).astype(int)

    # =========================================================================
    # Additional Rules (My Ideas)
    # =========================================================================

    def _check_weekend_publication(self, df: pd.DataFrame) -> pd.Series:
        """X001: Published on weekend."""
        return (df["is_weekend"] == 1).astype(int)

    def _check_q4_rush(self, df: pd.DataFrame) -> pd.Series:
        """X002: High-value tender in Q4."""
        median_value = df["tender_value"].median()
        return (
            (df["is_q4"] == 1) &
            (df["tender_value"] > median_value * 2)
        ).astype(int)

    def _check_december_rush(self, df: pd.DataFrame) -> pd.Series:
        """X003: High-value tender in December."""
        median_value = df["tender_value"].median()
        return (
            (df["is_december"] == 1) &
            (df["tender_value"] > median_value * 2)
        ).astype(int)

    def _check_cross_region_single_bidder(self, df: pd.DataFrame) -> pd.Series:
        """X004: Single bidder from different region."""
        return (
            (df["is_single_bidder"] == 1) &
            (df["is_cross_region"] == 1)
        ).astype(int)

    def _check_masked_data(self, df: pd.DataFrame) -> pd.Series:
        """X005: Masked buyer or supplier data."""
        return (
            (df["is_buyer_masked"] == 1) |
            (df["is_supplier_masked"] == 1)
        ).astype(int)

    def _check_award_issues(self, df: pd.DataFrame) -> pd.Series:
        """X006: Has unsuccessful or cancelled awards."""
        return (
            (df["has_unsuccessful_awards"] == 1) |
            (df["has_cancelled_awards"] == 1)
        ).astype(int)

    def _check_new_supplier_large_contract(self, df: pd.DataFrame) -> pd.Series:
        """X007: New supplier wins large contract."""
        if self._supplier_stats is None:
            return pd.Series(0, index=df.index)

        # Suppliers with only 1-2 wins
        new_suppliers = self._supplier_stats[
            self._supplier_stats["supplier_win_count"] <= 2
        ]["supplier_id"]

        # Large = top 10% by value
        large_threshold = df["tender_value"].quantile(0.9)

        return (
            df["supplier_id"].isin(new_suppliers) &
            (df["tender_value"] > large_threshold)
        ).astype(int)

    def _check_captive_supplier(self, df: pd.DataFrame) -> pd.Series:
        """X008: Supplier only works with one buyer."""
        if self._supplier_stats is None:
            return pd.Series(0, index=df.index)

        captive = self._supplier_stats[
            (self._supplier_stats["supplier_buyer_count"] == 1) &
            (self._supplier_stats["supplier_win_count"] >= 5)
        ]["supplier_id"]

        return df["supplier_id"].isin(captive).astype(int)

    def _check_single_bidder_low_discount(self, df: pd.DataFrame) -> pd.Series:
        """X009: Single bidder with very low discount."""
        return (
            (df["is_single_bidder"] == 1) &
            (df["discount_percentage_avg"].fillna(0) < 2)
        ).astype(int)

    def _check_same_day_same_supplier(self, df: pd.DataFrame) -> pd.Series:
        """X010: Multiple awards to same supplier on same day."""
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy["published_date"]):
            df_copy["published_date"] = pd.to_datetime(df_copy["published_date"], errors="coerce")

        df_copy["pub_date"] = df_copy["published_date"].dt.date

        daily_supplier = df_copy.groupby(["buyer_id", "supplier_id", "pub_date"]).size().reset_index(name="daily_count")
        suspicious = daily_supplier[daily_supplier["daily_count"] >= 3][["buyer_id", "supplier_id", "pub_date"]]

        merged = df_copy.merge(suspicious, on=["buyer_id", "supplier_id", "pub_date"], how="left", indicator=True)
        return (merged["_merge"] == "both").astype(int)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _calculate_risk_level(self, scores: pd.Series) -> pd.Series:
        """Calculate risk level from score."""
        return pd.cut(
            scores,
            bins=[-1, 2, 5, 10, 1000],
            labels=[RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL],
        )

    def _compute_flags_summary(self, df: pd.DataFrame):
        """Compute summary of detected flags."""
        flag_cols = [col for col in df.columns if col.startswith("flag_")]
        total = len(df)

        self.flags_detected = []
        for col in flag_cols:
            rule_name = col.replace("flag_", "")

            # Find matching rule config
            config = None
            for rule_id, rc in self.rule_configs.items():
                if rc.name == rule_name:
                    config = rc
                    break

            if config is None:
                continue

            count = int(df[col].sum())
            pct = round(count / total * 100, 2)

            flag = RedFlag(
                id=config.id,
                name=config.name,
                name_ua=config.name_ua,
                category=config.category,
                severity=config.severity,
                description=config.description,
                affected_records=count,
                percentage=pct,
            )
            self.flags_detected.append(flag)

        # Sort by affected_records descending
        self.flags_detected.sort(key=lambda x: x.affected_records, reverse=True)

    def summary(self) -> pd.DataFrame:
        """Get summary of detected flags."""
        if not self.flags_detected:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "id": f.id,
                "name": f.name,
                "name_ua": f.name_ua,
                "category": f.category,
                "severity": f.severity,
                "count": f.affected_records,
                "percentage": f.percentage,
            }
            for f in self.flags_detected
        ])

    def summary_by_category(self) -> pd.DataFrame:
        """Get summary grouped by category."""
        summary_df = self.summary()
        if summary_df.empty:
            return summary_df

        return summary_df.groupby("category").agg({
            "count": "sum",
            "id": "count"
        }).rename(columns={"id": "rules_triggered"}).reset_index()

    def summary_by_severity(self) -> pd.DataFrame:
        """Get summary grouped by severity."""
        summary_df = self.summary()
        if summary_df.empty:
            return summary_df

        return summary_df.groupby("severity").agg({
            "count": "sum",
            "id": "count"
        }).rename(columns={"id": "rules_triggered"}).reset_index()

    def risk_distribution(self) -> pd.DataFrame:
        """Get distribution of risk levels."""
        if self.results is None:
            raise ValueError("Run detect() first")

        dist = self.results["rule_risk_level"].value_counts()
        total = len(self.results)

        return pd.DataFrame({
            "risk_level": dist.index,
            "count": dist.values,
            "percentage": (dist.values / total * 100).round(2),
        })

    def get_flagged(
        self,
        min_score: Optional[int] = None,
        min_flags: Optional[int] = None,
        risk_level: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get flagged tenders.

        Args:
            min_score: Minimum risk score
            min_flags: Minimum number of flags
            risk_level: Filter by risk level

        Returns:
            Filtered DataFrame
        """
        if self.results is None:
            raise ValueError("Run detect() first")

        result = self.results.copy()

        if min_score is not None:
            result = result[result["rule_risk_score"] >= min_score]
        if min_flags is not None:
            result = result[result["rule_flags_count"] >= min_flags]
        if risk_level is not None:
            result = result[result["rule_risk_level"] == risk_level]

        return result

    def get_critical(self) -> pd.DataFrame:
        """Get critical risk tenders."""
        return self.get_flagged(risk_level=RiskLevel.CRITICAL)

    def get_high_risk(self) -> pd.DataFrame:
        """Get high risk or above tenders."""
        return self.get_flagged(min_score=6)

    def explain(self, tender_id: str) -> Dict:
        """
        Explain why a tender was flagged.

        Args:
            tender_id: Tender ID to explain

        Returns:
            Dictionary with explanation
        """
        if self.results is None:
            raise ValueError("Run detect() first")

        row = self.results[self.results["tender_id"] == tender_id]
        if len(row) == 0:
            return {"error": "Tender not found"}

        row = row.iloc[0]

        # Get triggered flags
        flag_cols = [col for col in self.results.columns if col.startswith("flag_")]
        triggered = []

        for col in flag_cols:
            if row[col] == 1:
                rule_name = col.replace("flag_", "")
                for rule_id, config in self.rule_configs.items():
                    if config.name == rule_name:
                        triggered.append({
                            "id": rule_id,
                            "name": config.name_ua,
                            "severity": config.severity,
                            "description": config.description,
                        })
                        break

        return {
            "tender_id": tender_id,
            "risk_score": int(row["rule_risk_score"]),
            "risk_level": str(row["rule_risk_level"]),
            "flags_count": int(row["rule_flags_count"]),
            "flags": triggered,
        }


# =============================================================================
# Standalone Functions
# =============================================================================

def quick_scan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick scan with basic rules only (no aggregations).

    Useful for fast initial screening.
    """
    basic_rules = [
        "R018", "R058", "R069", "R021", "R002",
        "X001", "X005", "X006", "X009"
    ]

    detector = RuleBasedDetector(enabled_rules=basic_rules)
    return detector.detect(df, compute_aggregations=False)


def full_scan(
    df: pd.DataFrame,
    buyers_df: Optional[pd.DataFrame] = None,
    suppliers_df: Optional[pd.DataFrame] = None,
    bids_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Full scan with all rules.
    """
    detector = RuleBasedDetector()
    return detector.detect(df, buyers_df, suppliers_df, bids_df)
