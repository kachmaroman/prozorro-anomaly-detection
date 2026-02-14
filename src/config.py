"""
Configuration and constants for anomaly detection.
"""

from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ANOMALIES_DIR = RESULTS_DIR / "anomalies"

# === Data Files ===
YEARS = [2022, 2023, 2024, 2025]

TENDER_FILES = {year: DATA_DIR / f"tenders_{year}.csv" for year in YEARS}
BID_FILES = {year: DATA_DIR / f"bids_{year}.csv" for year in YEARS}

BUYERS_FILE = DATA_DIR / "buyers.csv"
SUPPLIERS_FILE = DATA_DIR / "suppliers.csv"
BIDDERS_FILE = DATA_DIR / "bidders.csv"

# === Procurement Methods ===
class ProcurementMethod:
    LIMITED = "limited"      # 91% - спрощені закупівлі
    OPEN = "open"            # 5.5% - відкриті торги
    SELECTIVE = "selective"  # 3.3% - переговорна процедура

# === Thresholds (from methodology) ===
class Thresholds:
    # Rule-based detection
    SINGLE_BIDDER_LOW_DISCOUNT = 2.0      # % - критична аномалія
    SUPPLIER_MONOPOLY_SHARE = 50.0        # % - висока
    BUYER_SINGLE_BIDDER_RATE = 40.0       # % - висока
    PRICE_ZSCORE = 3.0                    # стандартних відхилень
    PRICE_INCREASE_CRITICAL = 0.0         # % - переплата

    # Statistical screens
    CV_SUSPICIOUS = 5.0                   # % - coefficient of variation
    CV_NORMAL = 10.0                      # %
    DIFFP_SUSPICIOUS = 5.0                # % - price difference
    RDNOR_SUSPICIOUS = 1.5                # relative distance
    KS_SUSPICIOUS = 0.3                   # KS statistic
    SKEWNESS_SUSPICIOUS = 0.5             # abs value
    KURTOSIS_SUSPICIOUS = 2.0             # excess kurtosis

    # Isolation Forest / LOF
    IF_CONTAMINATION = 0.05               # 5% anomalies expected

# === Risk Levels ===
class RiskLevel:
    CRITICAL = "critical"    # 5/5 methods agree
    HIGH = "high"            # 4/5 methods agree
    MEDIUM = "medium"        # 2-3/5 methods agree
    LOW = "low"              # 1 method

# === Feature Groups ===
NUMERIC_FEATURES = [
    "tender_value",
    "award_value",
    "price_change_pct",
    "number_of_tenderers",
    "number_of_bids",
    "number_of_documents",
    "award_concentration",
    "discount_percentage_avg",
]

CATEGORICAL_FEATURES = [
    "procurement_method",
    "main_procurement_category",
    "main_cpv_2_digit",
]

FLAG_FEATURES = [
    "is_single_bidder",
    "is_competitive",
    "is_cross_region",
    "has_enquiries",
    "is_weekend",
    "is_q4",
    "is_december",
]

# === CPV Categories (top) ===
CPV_NAMES = {
    33: "Медичне обладнання",
    45: "Будівництво",
    9: "Паливо та енергія",
    34: "Транспорт",
    15: "Продукти харчування",
    50: "Ремонт та обслуговування",
    44: "Будматеріали",
    90: "Послуги з відходів",
    72: "IT послуги",
    30: "Офісна техніка",
}

# === ML Preprocessing ===
# Features to log-transform (monetary and count variables with skewed distributions)
LOG_TRANSFORM_FEATURES = [
    "total_value", "tender_value", "award_value", "avg_value",
    "avg_award_value", "total_savings", "median_value",
    "total_awards", "total_tenders", "contracts_count", "buyer_count",
    "value_vs_cpv_median",
    "avg_award_days",
    "cpv_diversity",
]

# Default ML features per analysis level
DEFAULT_ML_FEATURES = {
    "tender": [
        "tender_value",
        "price_change_pct",
        "number_of_tenderers",
        "is_single_bidder",
        "is_competitive",
        "is_weekend",
        "is_q4",
        "is_december",
        "value_vs_cpv_median",
    ],
    "buyer": [
        "single_bidder_rate",
        "competitive_rate",
        "avg_discount_pct",
        "supplier_diversity_index",
        "total_tenders",
        "avg_value",
        "total_value",
        "cpv_concentration",
        "avg_award_days",
        "weekend_rate",
        "value_variance_coeff",
        "q4_rate",
    ],
    "supplier": [
        "total_awards",
        "total_value",
        "avg_award_value",
        "buyer_count",
        "single_bidder_rate",
        "avg_competitors",
        "cpv_diversity",
    ],
    "pair": [
        "contracts_count",
        "total_value",
        "avg_value",
        "single_bidder_rate",
        "exclusivity_buyer",
        "exclusivity_supplier",
        "temporal_concentration",
    ],
}

# Default contamination rate for anomaly detection
DEFAULT_CONTAMINATION = 0.05

# === Random State ===
RANDOM_STATE = 42
