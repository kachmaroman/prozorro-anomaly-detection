"""
Statistical Screens for Anomaly Detection (Level 2)

Statistical methods for detecting anomalies:
1. Benford's Law - digit distribution analysis
2. Z-score outliers - standard deviation based
3. IQR outliers - interquartile range based
4. Distribution tests - normality, uniformity
5. HHI concentration - market concentration index
6. Bid pattern analysis - clustering, spread

Bid-level Statistical Screens (from methodology_plan):
- CV (Coefficient of Variation): < 5% = підозріло схожі ставки
- DIFFP (Price Difference): > 5% = переможець "знав" скільки ставити
- RDNOR (Relative Distance): > 1.5 = переможець відірвався, решта скупчені
- KS-stat (Uniformity): > 0.3 = неприродний розподіл ставок
- Skewness: |skew| > 0.5 = ставки зміщені в один бік
- Kurtosis: > 2 = ставки або дуже схожі, або є різкі викиди

All bid-level screens apply only to competitive tenders (open/selective) with 3+ bidders.
"""

import warnings
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.stats import chi2_contingency, kstest, shapiro


# =============================================================================
# Constants
# =============================================================================

# Benford's Law expected frequencies for first digit
BENFORD_EXPECTED = {
    1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
    5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
}

# Chi-square critical values (df=8, common alpha levels)
CHI2_CRITICAL = {
    0.10: 13.36,
    0.05: 15.51,
    0.01: 20.09,
    0.001: 26.12
}


@dataclass
class StatisticalResult:
    """Result of a statistical test."""
    name: str
    statistic: float
    p_value: Optional[float]
    is_anomaly: bool
    threshold: float
    description: str


# =============================================================================
# Main Statistical Detector Class
# =============================================================================

class StatisticalDetector:
    """
    Statistical anomaly detection using various statistical tests.

    Methods:
    - Benford's Law analysis for prices
    - Z-score and IQR outlier detection
    - Distribution tests (KS, Shapiro)
    - HHI market concentration
    - Bid spread analysis

    Usage:
        detector = StatisticalDetector()
        results = detector.detect(tenders_df, bids_df)
    """

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        benford_alpha: float = 0.05,
        min_samples: int = 30
    ):
        """
        Initialize detector with thresholds.

        Args:
            zscore_threshold: Z-score threshold for outliers (default 3.0)
            iqr_multiplier: IQR multiplier for outliers (default 1.5)
            benford_alpha: Significance level for Benford test (default 0.05)
            min_samples: Minimum samples for statistical tests
        """
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.benford_alpha = benford_alpha
        self.min_samples = min_samples

        self.results: Optional[pd.DataFrame] = None
        self.tender_stats: Dict = {}
        self.buyer_stats: Dict = {}
        self.cpv_stats: Dict = {}

    def detect(
        self,
        df: pd.DataFrame,
        bids_df: Optional[pd.DataFrame] = None,
        group_by_cpv: bool = True
    ) -> pd.DataFrame:
        """
        Run all statistical screens on the data.

        Args:
            df: Tender DataFrame
            bids_df: Bids DataFrame (optional, for bid-level analysis)
            group_by_cpv: Whether to compute CPV-relative statistics

        Returns:
            DataFrame with statistical anomaly flags
        """
        print(f"Processing {len(df):,} tenders...")
        result = df.copy()

        # Step 1: Value-based outliers
        print("Step 1/5: Computing value outliers (Z-score, IQR)...")
        result = self._detect_value_outliers(result, group_by_cpv)

        # Step 2: Price pattern analysis
        print("Step 2/5: Analyzing price patterns...")
        result = self._detect_price_patterns(result)

        # Step 3: Benford's Law (requires bids)
        if bids_df is not None:
            print("Step 3/5: Running Benford's Law analysis...")
            result = self._detect_benford_anomalies(result, bids_df)
        else:
            print("Step 3/5: Skipping Benford (no bids data)...")
            result["stat_benford_anomaly"] = 0

        # Step 4: Bid spread analysis (requires bids)
        if bids_df is not None:
            print("Step 4/5: Analyzing bid spreads...")
            result = self._detect_bid_spread_anomalies(result, bids_df)
        else:
            print("Step 4/5: Skipping bid spread (no bids data)...")
            result["stat_bid_spread_anomaly"] = 0
            result["stat_bid_clustering"] = 0

        # Step 5: Concentration analysis
        print("Step 5/5: Computing market concentration...")
        result = self._detect_concentration_anomalies(result)

        # Compute aggregate score
        result = self._compute_statistical_score(result)

        self.results = result
        print("Statistical screening complete!")

        return result

    # =========================================================================
    # Value Outlier Detection
    # =========================================================================

    def _detect_value_outliers(
        self,
        df: pd.DataFrame,
        group_by_cpv: bool = True
    ) -> pd.DataFrame:
        """Detect outliers using Z-score and IQR methods."""
        result = df.copy()

        # Global outliers
        result["stat_zscore_value"] = self._zscore_outlier(df["tender_value"])
        result["stat_iqr_value"] = self._iqr_outlier(df["tender_value"])

        # Discount outliers
        if "price_change_pct" in df.columns:
            result["stat_zscore_discount"] = self._zscore_outlier(df["price_change_pct"])
            result["stat_iqr_discount"] = self._iqr_outlier(df["price_change_pct"])

        # CPV-relative outliers (compare within category)
        if group_by_cpv and "main_cpv_2_digit" in df.columns:
            result["stat_zscore_value_cpv"] = df.groupby("main_cpv_2_digit")["tender_value"].transform(
                lambda x: self._zscore_outlier(x)
            )
            result["stat_iqr_value_cpv"] = df.groupby("main_cpv_2_digit")["tender_value"].transform(
                lambda x: self._iqr_outlier(x)
            )

        return result

    def _zscore_outlier(self, series: pd.Series) -> pd.Series:
        """Flag outliers based on Z-score."""
        mean = series.mean()
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0, index=series.index)
        zscore = (series - mean) / std
        return (zscore.abs() > self.zscore_threshold).astype(int)

    def _iqr_outlier(self, series: pd.Series) -> pd.Series:
        """Flag outliers based on IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return pd.Series(0, index=series.index)
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr
        return ((series < lower) | (series > upper)).astype(int)

    # =========================================================================
    # Price Pattern Detection
    # =========================================================================

    def _detect_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect suspicious price patterns (competitive tenders only for some)."""
        result = df.copy()

        # Mask for competitive tenders with 3+ bidders
        is_competitive = df["procurement_method"].isin(["open", "selective"])
        has_multiple_bidders = df["number_of_tenderers"] >= 3
        competitive_mask = is_competitive & has_multiple_bidders

        # Round number prices (ends in 000) - all tenders
        result["stat_round_price"] = df["tender_value"].apply(
            lambda x: 1 if pd.notna(x) and x > 0 and x % 1000 == 0 else 0
        )

        # Very round prices (ends in 00000) - all tenders
        result["stat_very_round_price"] = df["tender_value"].apply(
            lambda x: 1 if pd.notna(x) and x > 0 and x % 100000 == 0 else 0
        )

        # Suspicious 99 ending (psychological pricing) - all tenders
        result["stat_99_ending"] = df["tender_value"].apply(
            lambda x: 1 if pd.notna(x) and x > 0 and int(x) % 100 == 99 else 0
        )

        # Award very close to tender value (>99%) - ONLY competitive with 3+ bidders
        # In limited procurement, award ≈ tender value is normal (no competition)
        if "award_value" in df.columns:
            ratio = df["award_value"] / df["tender_value"].replace(0, np.nan)
            result["stat_award_ratio_suspicious"] = (
                competitive_mask & (ratio > 0.99) & (ratio <= 1.0)
            ).fillna(0).astype(int)
        else:
            result["stat_award_ratio_suspicious"] = 0

        return result

    # =========================================================================
    # Benford's Law Analysis
    # =========================================================================

    def _detect_benford_anomalies(
        self,
        df: pd.DataFrame,
        bids_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect Benford's Law violations at buyer and supplier level.

        Benford's Law requires large samples (30+), so we test:
        - Per buyer: all bids received by this buyer
        - Per supplier (bidder): all bids submitted by this bidder

        Tender is flagged if its buyer OR winning supplier fails Benford test.
        """
        result = df.copy()

        # Filter valid bids
        valid_bids = bids_df[bids_df["bid_amount"] > 0].copy()
        if len(valid_bids) == 0:
            result["stat_benford_buyer"] = 0
            result["stat_benford_supplier"] = 0
            return result

        # Extract first digit
        valid_bids["first_digit"] = valid_bids["bid_amount"].apply(self._get_first_digit)
        valid_bids = valid_bids[valid_bids["first_digit"].between(1, 9)]

        # Merge with tender data to get buyer_id
        bids_with_info = valid_bids.merge(
            df[["tender_id", "buyer_id"]].drop_duplicates(),
            on="tender_id",
            how="left"
        )

        # Test Benford per BUYER (all bids received by this buyer)
        print("    Testing Benford per buyer...")
        buyer_benford = bids_with_info.groupby("buyer_id").apply(
            self._test_benford, include_groups=False
        ).reset_index()
        buyer_benford.columns = ["buyer_id", "stat_benford_buyer"]

        buyers_tested = len(buyer_benford)
        buyers_anomaly = buyer_benford["stat_benford_buyer"].sum()
        print(f"    Buyers tested: {buyers_tested:,}, anomalies: {buyers_anomaly:,}")

        # Test Benford per BIDDER/SUPPLIER (all bids from this bidder)
        print("    Testing Benford per supplier...")
        supplier_benford = bids_with_info.groupby("bidder_id").apply(
            self._test_benford, include_groups=False
        ).reset_index()
        supplier_benford.columns = ["bidder_id", "stat_benford_supplier"]

        suppliers_tested = len(supplier_benford)
        suppliers_anomaly = supplier_benford["stat_benford_supplier"].sum()
        print(f"    Suppliers tested: {suppliers_tested:,}, anomalies: {suppliers_anomaly:,}")

        # Merge buyer Benford back to tenders
        result = result.merge(buyer_benford, on="buyer_id", how="left")

        # For supplier, match via winning bidder
        winners = bids_df[bids_df["is_winner"] == 1][["tender_id", "bidder_id"]].drop_duplicates()
        result = result.merge(winners, on="tender_id", how="left")
        result = result.merge(supplier_benford, on="bidder_id", how="left")

        # Fill NaN and convert to int
        result["stat_benford_buyer"] = result["stat_benford_buyer"].fillna(0).astype(int)
        result["stat_benford_supplier"] = result["stat_benford_supplier"].fillna(0).astype(int)

        # Drop helper column
        if "bidder_id" in result.columns:
            result = result.drop(columns=["bidder_id"])

        return result

    def _get_first_digit(self, value: float) -> int:
        """Extract first significant digit from a number."""
        if pd.isna(value) or value <= 0:
            return 0
        return int(str(abs(value)).lstrip('0').replace('.', '')[0])

    def _test_benford(self, group: pd.DataFrame) -> int:
        """Test if a group of values follows Benford's Law."""
        if len(group) < self.min_samples:
            return 0

        observed = group["first_digit"].value_counts(normalize=True)

        chi_sq = 0
        for digit in range(1, 10):
            obs = observed.loc[digit] if digit in observed.index else 0
            exp = BENFORD_EXPECTED[digit]
            chi_sq += ((obs - exp) ** 2) / exp

        # Scale by sample size
        chi_sq *= len(group)

        # Compare to critical value
        critical = CHI2_CRITICAL[self.benford_alpha]
        return 1 if chi_sq > critical else 0

    # =========================================================================
    # Bid Spread Analysis (Statistical Screens from Methodology)
    # =========================================================================

    def _detect_bid_spread_anomalies(
        self,
        df: pd.DataFrame,
        bids_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect suspicious bid spread patterns using statistical screens.

        Screens implemented (from methodology_plan):
        1. CV (Coefficient of Variation) - подозріло схожі ставки
        2. DIFFP (Price Difference) - різниця між 1-м і 2-м місцем
        3. RDNOR (Relative Distance) - відносна відстань переможця
        4. KS-stat (Uniformity) - рівномірність розподілу
        5. Skewness - асиметрія розподілу
        6. Kurtosis - "хвости" розподілу

        All screens apply only to competitive tenders (open/selective) with 3+ bidders.
        """
        result = df.copy()

        # Calculate comprehensive bid statistics per tender
        bid_stats = self._calculate_bid_statistics(bids_df)

        # Merge with tenders
        result = result.merge(bid_stats, on="tender_id", how="left")

        # Mask for competitive tenders (Open/Selective) with 3+ bidders
        is_competitive = result["procurement_method"].isin(["open", "selective"])
        has_multiple_bidders = result["bid_count"].fillna(0) >= 3
        competitive_mask = is_competitive & has_multiple_bidders

        # =====================================================================
        # Screen 1: CV (Coefficient of Variation)
        # Норма: > 10% (ставки різні, конкуренція реальна)
        # Аномалія: < 5-6% (ставки підозріло схожі)
        # =====================================================================
        result["stat_cv_anomaly"] = (
            competitive_mask &
            (result["bid_cv"].fillna(999) < 0.05)  # CV < 5%
        ).astype(int)

        # =====================================================================
        # Screen 2: DIFFP (Price Difference)
        # (друга найнижча ставка - найнижча) / найнижча
        # Норма: < 3% (щільна боротьба за перемогу)
        # Аномалія: > 5% (переможець "знав" скільки ставити)
        # =====================================================================
        result["stat_diffp_anomaly"] = (
            competitive_mask &
            (result["bid_diffp"].fillna(0) > 0.05)  # DIFFP > 5%
        ).astype(int)

        # =====================================================================
        # Screen 3: RDNOR (Relative Distance to Nearest Other Record)
        # Δ₁₂ / mean(Δᵢ) - відстань переможця до 2-го vs середня відстань
        # Норма: ≈ 1 (розриви між ставками рівномірні)
        # Аномалія: > 1.5 (переможець відірвався, решта скупчені)
        # =====================================================================
        result["stat_rdnor_anomaly"] = (
            competitive_mask &
            (result["bid_rdnor"].fillna(0) > 1.5)
        ).astype(int)

        # =====================================================================
        # Screen 4: KS-stat (Kolmogorov-Smirnov Uniformity Test)
        # max|F(x) - U(x)| - відхилення від рівномірного розподілу
        # Норма: < 0.2 (ставки розподілені рівномірно)
        # Аномалія: > 0.3 (ставки згруповані, неприродний розподіл)
        # =====================================================================
        result["stat_ks_anomaly"] = (
            competitive_mask &
            (result["bid_ks_stat"].fillna(0) > 0.3)
        ).astype(int)

        # =====================================================================
        # Screen 5: Skewness (Asymmetry)
        # μ₃ / σ³ - асиметрія розподілу ставок
        # Норма: ≈ 0 (симетричний розподіл ставок)
        # Аномалія: < -0.5 або > 0.5 (ставки зміщені в один бік)
        # =====================================================================
        result["stat_skewness_anomaly"] = (
            competitive_mask &
            (result["bid_skewness"].fillna(0).abs() > 0.5)
        ).astype(int)

        # =====================================================================
        # Screen 6: Kurtosis (Tail heaviness)
        # μ₄ / σ⁴ - 3 - "хвости" розподілу
        # Норма: ≈ 0 (нормальний "хвіст" розподілу)
        # Аномалія: > 2 (ставки або дуже схожі, або є різкі викиди)
        # =====================================================================
        result["stat_kurtosis_anomaly"] = (
            competitive_mask &
            (result["bid_kurtosis"].fillna(0) > 2)
        ).astype(int)

        # =====================================================================
        # Legacy screens (kept for backwards compatibility)
        # =====================================================================

        # Very low spread (< 1%) - suspiciously similar bids
        result["stat_bid_spread_anomaly"] = (
            competitive_mask &
            (result["bid_spread"].fillna(999) < 1.01)
        ).astype(int)

        # Very low CV (< 1%) - extremely clustered bids (stricter than cv_anomaly)
        result["stat_bid_clustering"] = (
            competitive_mask &
            (result["bid_cv"].fillna(999) < 0.01)
        ).astype(int)

        # Very high spread (> 10x) - potential manipulation
        result["stat_bid_spread_high"] = (
            competitive_mask &
            (result["bid_spread"].fillna(0) > 10)
        ).astype(int)

        return result

    def _calculate_bid_statistics(self, bids_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive bid statistics per tender.

        Returns DataFrame with columns:
        - bid_count, bid_min, bid_max, bid_mean, bid_std
        - bid_cv (Coefficient of Variation)
        - bid_spread (max/min ratio)
        - bid_diffp (Price Difference: (2nd - 1st) / 1st)
        - bid_rdnor (Relative Distance)
        - bid_ks_stat (KS uniformity test)
        - bid_skewness, bid_kurtosis
        """
        # Basic statistics
        basic_stats = bids_df.groupby("tender_id").agg({
            "bid_amount": ["count", "min", "max", "mean", "std"]
        }).reset_index()
        basic_stats.columns = ["tender_id", "bid_count", "bid_min", "bid_max", "bid_mean", "bid_std"]

        # CV and spread
        basic_stats["bid_cv"] = basic_stats["bid_std"] / basic_stats["bid_mean"].replace(0, np.nan)
        basic_stats["bid_spread"] = basic_stats["bid_max"] / basic_stats["bid_min"].replace(0, np.nan)

        # Advanced statistics per tender (requires groupby apply)
        # Use a list to collect results for better pandas compatibility
        advanced_results = []
        for tender_id, group in bids_df.groupby("tender_id")["bid_amount"]:
            stats = self._compute_advanced_bid_stats(group)
            stats["tender_id"] = tender_id
            advanced_results.append(stats)

        advanced_df = pd.DataFrame(advanced_results)

        # Merge basic and advanced
        bid_stats = basic_stats.merge(advanced_df, on="tender_id", how="left")

        return bid_stats

    def _compute_advanced_bid_stats(self, bids: pd.Series) -> dict:
        """
        Compute advanced bid statistics for a single tender.

        Args:
            bids: Series of bid amounts for one tender

        Returns:
            dict with diffp, rdnor, ks_stat, skewness, kurtosis
        """
        result = {
            "bid_diffp": np.nan,
            "bid_rdnor": np.nan,
            "bid_ks_stat": np.nan,
            "bid_skewness": np.nan,
            "bid_kurtosis": np.nan,
        }

        bids = bids.dropna().sort_values()
        n = len(bids)

        if n < 3:
            return result

        bids_arr = bids.values

        # DIFFP: (2nd lowest - 1st lowest) / 1st lowest
        if bids_arr[0] > 0:
            result["bid_diffp"] = (bids_arr[1] - bids_arr[0]) / bids_arr[0]

        # RDNOR: Δ₁₂ / mean(Δᵢ)
        # Δ₁₂ = difference between 1st and 2nd place
        # mean(Δᵢ) = mean of all consecutive differences
        if n >= 3:
            delta_12 = bids_arr[1] - bids_arr[0]
            all_deltas = np.diff(bids_arr)
            mean_delta = np.mean(all_deltas)
            if mean_delta > 0:
                result["bid_rdnor"] = delta_12 / mean_delta

        # KS-stat: Kolmogorov-Smirnov test against uniform distribution
        # Normalize bids to [0, 1] range
        if bids_arr[-1] > bids_arr[0]:
            normalized = (bids_arr - bids_arr[0]) / (bids_arr[-1] - bids_arr[0])
            try:
                ks_stat, _ = kstest(normalized, 'uniform')
                result["bid_ks_stat"] = ks_stat
            except Exception:
                pass

        # Skewness and Kurtosis
        # Only compute if there's enough variance (avoid precision loss warning)
        if n >= 3:
            std = np.std(bids_arr)
            # Check coefficient of variation - if bids are too similar, skip
            mean_val = np.mean(bids_arr)
            cv = std / mean_val if mean_val > 0 else 0
            if cv > 0.001:  # At least 0.1% variation required
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    try:
                        result["bid_skewness"] = stats.skew(bids_arr)
                        result["bid_kurtosis"] = stats.kurtosis(bids_arr)  # Fisher's definition (normal = 0)
                    except Exception:
                        pass
            # If CV ≈ 0, all bids are nearly identical (suspicious! but keep NaN for stats)

        return result

    # =========================================================================
    # Market Concentration Analysis
    # =========================================================================

    def _detect_concentration_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market concentration anomalies using HHI."""
        result = df.copy()

        # Calculate HHI per buyer (concentration of suppliers)
        buyer_supplier = df.groupby(["buyer_id", "supplier_id"]).agg({
            "tender_value": "sum"
        }).reset_index()

        # Calculate market share per buyer
        buyer_total = buyer_supplier.groupby("buyer_id")["tender_value"].transform("sum")
        buyer_supplier["share"] = buyer_supplier["tender_value"] / buyer_total.replace(0, np.nan)
        buyer_supplier["share_squared"] = buyer_supplier["share"] ** 2

        # HHI = sum of squared market shares (0-1, higher = more concentrated)
        buyer_hhi = buyer_supplier.groupby("buyer_id")["share_squared"].sum().reset_index()
        buyer_hhi.columns = ["buyer_id", "buyer_hhi"]

        result = result.merge(buyer_hhi, on="buyer_id", how="left")

        # Flag high concentration (HHI > 0.5 = highly concentrated)
        result["stat_high_concentration"] = (
            result["buyer_hhi"].fillna(0) > 0.5
        ).astype(int)

        # Flag monopoly (HHI > 0.9 = near monopoly)
        result["stat_monopoly"] = (
            result["buyer_hhi"].fillna(0) > 0.9
        ).astype(int)

        return result

    # =========================================================================
    # Aggregate Score
    # =========================================================================

    def _compute_statistical_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute aggregate statistical anomaly score."""
        result = df.copy()

        # Define statistical flag columns and weights
        # NOTE: benford_buyer, benford_supplier, round_price removed from scoring
        # (too noisy - 52% flagged). They are still computed for analysis.
        stat_flags = {
            # Value outliers
            "stat_zscore_value": 1,
            "stat_iqr_value": 1,
            "stat_zscore_value_cpv": 2,  # CPV-relative is more meaningful
            "stat_iqr_value_cpv": 2,
            "stat_zscore_discount": 1,
            "stat_iqr_discount": 1,
            # Price patterns (only suspicious ones, not round prices)
            "stat_99_ending": 0.5,
            "stat_award_ratio_suspicious": 1,
            # Bid distribution screens (from methodology_plan) - primary signals
            "stat_cv_anomaly": 2,        # CV < 5% - підозріло схожі ставки
            "stat_diffp_anomaly": 2,     # DIFFP > 5% - переможець "знав" ціну
            "stat_rdnor_anomaly": 2,     # RDNOR > 1.5 - переможець відірвався
            "stat_ks_anomaly": 1.5,      # KS > 0.3 - неприродний розподіл
            "stat_skewness_anomaly": 1,  # |Skew| > 0.5 - асиметрія
            "stat_kurtosis_anomaly": 1,  # Kurt > 2 - аномальні хвости
            # Legacy bid patterns (kept for compatibility)
            "stat_bid_spread_anomaly": 2,
            "stat_bid_clustering": 2,
            "stat_bid_spread_high": 1,
            # Concentration
            "stat_high_concentration": 1,
            "stat_monopoly": 2,
        }

        # Calculate score
        result["stat_score"] = 0
        result["stat_flags_count"] = 0

        for flag, weight in stat_flags.items():
            if flag in result.columns:
                result["stat_score"] += result[flag].fillna(0) * weight
                result["stat_flags_count"] += result[flag].fillna(0)

        # Calculate risk level (adjusted thresholds after removing noisy flags)
        result["stat_risk_level"] = pd.cut(
            result["stat_score"],
            bins=[-0.1, 1, 4, 8, 100],
            labels=["low", "medium", "high", "critical"]
        )

        # Binary anomaly flag for ensemble (score >= 4 = anomaly)
        # Raised from 3 to 4 after removing benford/round_price from scoring
        result["stat_anomaly"] = (result["stat_score"] >= 4).astype(int)

        return result

    # =========================================================================
    # Summary Methods
    # =========================================================================

    def summary(self) -> pd.DataFrame:
        """Get summary of statistical flags."""
        if self.results is None:
            raise ValueError("Run detect() first")

        stat_cols = [c for c in self.results.columns if c.startswith("stat_") and c not in ["stat_score", "stat_flags_count", "stat_risk_level"]]

        summary_data = []
        total = len(self.results)

        for col in stat_cols:
            count = int(self.results[col].sum())
            pct = round(count / total * 100, 2)
            summary_data.append({
                "flag": col.replace("stat_", ""),
                "count": count,
                "percentage": pct
            })

        return pd.DataFrame(summary_data).sort_values("count", ascending=False)

    def risk_distribution(self) -> pd.DataFrame:
        """Get distribution of statistical risk levels."""
        if self.results is None:
            raise ValueError("Run detect() first")

        dist = self.results["stat_risk_level"].value_counts()
        total = len(self.results)

        return pd.DataFrame({
            "risk_level": dist.index,
            "count": dist.values,
            "percentage": (dist.values / total * 100).round(2)
        })

    def get_anomalies(self, min_score: float = 3.0) -> pd.DataFrame:
        """Get tenders with statistical anomaly score above threshold."""
        if self.results is None:
            raise ValueError("Run detect() first")
        return self.results[self.results["stat_score"] >= min_score]


# =============================================================================
# Standalone Functions for Quick Analysis
# =============================================================================

def benford_test(values: pd.Series, alpha: float = 0.05) -> StatisticalResult:
    """
    Test if a series of values follows Benford's Law.

    Args:
        values: Series of positive numbers
        alpha: Significance level

    Returns:
        StatisticalResult with test outcome
    """
    # Extract first digits
    first_digits = values[values > 0].apply(
        lambda x: int(str(abs(x)).lstrip('0').replace('.', '')[0])
    )
    first_digits = first_digits[first_digits.between(1, 9)]

    if len(first_digits) < 30:
        return StatisticalResult(
            name="Benford's Law",
            statistic=0,
            p_value=None,
            is_anomaly=False,
            threshold=CHI2_CRITICAL[alpha],
            description="Insufficient data (need at least 30 samples)"
        )

    observed = first_digits.value_counts(normalize=True)

    chi_sq = 0
    for digit in range(1, 10):
        obs = observed.loc[digit] if digit in observed.index else 0
        exp = BENFORD_EXPECTED[digit]
        chi_sq += ((obs - exp) ** 2) / exp

    chi_sq *= len(first_digits)
    critical = CHI2_CRITICAL[alpha]

    return StatisticalResult(
        name="Benford's Law",
        statistic=chi_sq,
        p_value=1 - stats.chi2.cdf(chi_sq, df=8),
        is_anomaly=chi_sq > critical,
        threshold=critical,
        description=f"Chi-square = {chi_sq:.2f}, critical = {critical:.2f}"
    )


def hhi_index(shares: pd.Series) -> float:
    """
    Calculate Herfindahl-Hirschman Index.

    Args:
        shares: Series of market shares (should sum to 1)

    Returns:
        HHI value (0-1)
    """
    return (shares ** 2).sum()


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0
) -> Tuple[pd.Series, float, float]:
    """
    Detect outliers using Z-score method.

    Returns:
        (boolean mask, mean, std)
    """
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(False, index=series.index), mean, std
    zscore = (series - mean) / std
    return zscore.abs() > threshold, mean, std


def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = 1.5
) -> Tuple[pd.Series, float, float]:
    """
    Detect outliers using IQR method.

    Returns:
        (boolean mask, lower_bound, upper_bound)
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper), lower, upper
